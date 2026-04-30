import os
import cv2
import webcolors  # 用于颜色名称匹配
from PEACE.dependencies.ultralytics import YOLOv10

class map_legend_detector:

    def __init__(self, model_path=os.path.join(os.path.dirname(__file__), "..", "models", "det_legend", "weights", "best.pt")):
        self.model = YOLOv10(model_path)
        # 初始化颜色库映射表 (CSS3标准)
        self.color_db = self._prepare_color_db()

    def _prepare_color_db(self):
        """兼容新旧版本 webcolors 的颜色库预处理"""
        db = {}
        try:
            # 尝试新版本 (v24.0+) 的 API
            for name in webcolors.names(spec='css3'):
                rgb = webcolors.name_to_rgb(name)
                db[name] = (rgb.red, rgb.green, rgb.blue)
        except (AttributeError, ValueError):
            # 备选方案：尝试旧版本的常量
            try:
                # 有些版本使用这种形式
                source = getattr(webcolors, 'CSS3_NAMES_TO_HEX',
                                 getattr(webcolors, 'css3_names_to_hex', {}))
                for name, hex_val in source.items():
                    rgb = webcolors.hex_to_rgb(hex_val)
                    db[name] = (rgb.red, rgb.green, rgb.blue)
            except:
                # 如果都失败了，手动硬编码几种常用颜色作为保底，防止程序崩溃
                db = {"white": (255, 255, 255), "black": (0, 0, 0), "red": (255, 0, 0)}
        return db

    def _get_color_name(self, bgr):
        """通过计算欧几里得距离找到最接近的颜色名称"""
        # OpenCV 是 BGR，转换为 RGB
        b, g, r = bgr
        min_dist = float("inf")
        closest_name = "unknown"

        for name, rgb_val in self.color_db.items():
            # 计算 RGB 空间的欧几里得距离
            rd = (r - rgb_val[0]) ** 2
            gd = (g - rgb_val[1]) ** 2
            bd = (b - rgb_val[2]) ** 2
            dist = (rd + gd + bd) ** 0.5

            if dist < min_dist:
                min_dist = dist
                closest_name = name
        return closest_name

    def overlap(self, anchor_col, bndbox):
        x0, y0, x1, y1 = bndbox
        x = (x0 + x1) / 2
        return anchor_col[0] < x < anchor_col[1]

    def distance(self, color_bndbox, text_bndbox):
        c_x0, c_y0, c_x1, c_y1 = color_bndbox
        t_x0, t_y0, t_x1, t_y1 = text_bndbox
        c_x, c_y = c_x1, (c_y0 + c_y1) / 2
        t_x, t_y = t_x0, (t_y0 + t_y1) / 2
        return ((c_x - t_x) ** 2 + (c_y - t_y) ** 2) ** 0.5

    def clamp_bndbox(self, bndbox, width, height):
        x0, y0, x1, y1 = bndbox
        x0, x1 = int(max(0, min(x0, width))), int(max(0, min(x1, width)))
        y0, y1 = int(max(0, min(y0, height))), int(max(0, min(y1, height)))
        return (x0, y0, x1, y1)

    def shrink_bndbox(self, image, bndbox):
        h, w = image.shape[:2]
        x0, y0, x1, y1 = self.clamp_bndbox(bndbox, w, h)
        if x1 <= x0 or y1 <= y0: return (x0, y0, x1, y1)
        roi = image[y0:y1, x0:x1]
        if roi.size == 0: return (x0, y0, x1, y1)
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        s = gray.min(axis=0)
        thred, width_roi = 32, s.shape[0]
        dx0 = int(width_roi * 0.01)
        while dx0 < width_roi and s[dx0] >= 256 - thred: dx0 += 1
        dx1 = int(width_roi * 0.99) - 1
        while dx1 > dx0 and s[dx1] >= 256 - thred: dx1 -= 1
        new_x0, new_x1 = x0 + max(dx0 - 1, 0), x0 + min(dx1 + 1, width_roi)
        return self.clamp_bndbox((new_x0, y0, new_x1, y1), w, h)

    def mean_bgr_in_box(self, image, bndbox):
        h, w = image.shape[:2]
        x0, y0, x1, y1 = self.clamp_bndbox(bndbox, w, h)
        if x1 <= x0 or y1 <= y0: return None, 0
        patch = image[y0:y1, x0:x1]
        if patch.size == 0: return None, 0
        area = int(patch.shape[0] * patch.shape[1])
        mean_bgr = patch.mean(axis=(0, 1)).astype(int).tolist()
        return mean_bgr, area

    def detect(self, image_path):
        objs = self.model.predict(source=image_path)[0]
        image = cv2.imread(image_path)
        if image is None: raise ValueError(f"Failed to read image: {image_path}")
        height, width = image.shape[:2]

        color_bndboxes = objs["color_bndbox"]
        text_bndboxes = list(objs["text_bndbox"])
        legends, used_text, idx = {}, [False] * len(text_bndboxes), 0

        for color_bndbox in color_bndboxes:
            color_bndbox = self.clamp_bndbox(color_bndbox, width, height)
            thred, paired_i, min_dist = color_bndbox[3] - color_bndbox[1], -1, float("inf")

            for i, text_bndbox in enumerate(text_bndboxes):
                if used_text[i]: continue
                dist = self.distance(color_bndbox, text_bndbox)
                if dist < min_dist:
                    min_dist, paired_i = dist, i

            if paired_i == -1 or min_dist > thred: continue

            used_text[paired_i] = True
            paired_text_bndbox = self.shrink_bndbox(image, text_bndboxes[paired_i])
            mean_bgr, area = self.mean_bgr_in_box(image, color_bndbox)
            if mean_bgr is None or area == 0: continue

            # --- 仅优化 color_name ---
            color_name_str = self._get_color_name(mean_bgr)
            # ------------------------

            legends[idx] = {
                "color_bndbox": color_bndbox,
                "text_bndbox": paired_text_bndbox,
                "color": mean_bgr,
                "color_name": color_name_str, # ✅ 已填充
                "text": "",                   # 保持为空
                "area": area,
            }
            idx += 1

        return legends

if __name__ == "__main__":
    det = map_legend_detector()
    # 替换为你实际的路径
    results = det.detect(os.path.join(os.getcwd(), "E4901.JPG"))
    print(results)
