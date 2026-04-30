import os
import sys
import base64
import json
import re
import time
import csv
import numpy as np
import cv2
from typing import List, Optional, TypedDict, Dict, Any, Union
from pydantic import BaseModel, Field
from langchain_core.messages import HumanMessage
from langchain_community.chat_models import ChatTongyi
from langgraph.graph import StateGraph, START, END

# === 1. 依赖库导入与环境配置 (双轨制与 YOLO 前置探针) ===

current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

try:
    from skimage import io as sio
    from skimage.feature import local_binary_pattern
    from shapely.geometry import Polygon

    HAS_CV_LIBS = True
except Exception as e:
    HAS_CV_LIBS = False

try:
    from PEACE.tool_pool.map_legend_detector import map_legend_detector
    from PEACE.tool_pool.map_component_detector import map_component_detector

    HAS_PEACE = True
except Exception as e:
    HAS_PEACE = False

try:
    from ultralytics import YOLO

    HAS_YOLO = True
except ImportError as e:
    HAS_YOLO = False

# ==========================================
# 2. 常量配置
# ==========================================
DOWNSCALE_TARGET = 2500
API_IMAGE_MAX_DIM = 2048
WHITE_THRESH = 245


# ==========================================
# 3. 数据结构 (严格对齐上传文件的 JSON Schema)
# ==========================================

class GeoSymbol(BaseModel):
    base: Optional[str] = Field(None)
    superscript: Optional[str] = Field(None)
    subscript: Optional[str] = Field(None)
    final: Optional[str] = Field("Unknown", alias="symbol_final")
    terminal: Optional[str] = Field("Unknown", alias="symbol_terminal")
    html: Optional[str] = Field("", alias="symbol_html")
    latex: Optional[str] = Field("Unknown", alias="symbol_latex")
    confidence: float = Field(0.0, alias="symbol_conf")

    class Config:
        populate_by_name = True


class LegendItem(BaseModel):
    id: int
    color_bbox: List[int] = Field(default_factory=list)
    text_bbox: List[int] = Field(default_factory=list)
    avg_color: List[int] = Field(...)
    color_name: str = Field("unknown")
    area: float = Field(0.0)
    color_img: str = Field("")
    text_img: str = Field("")
    hist: Optional[List[float]] = Field(None)

    legend_text: str = Field("")
    symbol: GeoSymbol = Field(default_factory=GeoSymbol)
    is_recognized: bool = Field(False)

    color_patch_base64: Optional[str] = Field(None, exclude=True)
    text_patch_base64: Optional[str] = Field(None, exclude=True)


class RegionGeoInfo(BaseModel):
    symbol: GeoSymbol = Field(default_factory=GeoSymbol)
    unit_name: str = Field("")
    age: Optional[str] = Field(None)
    genesis: Optional[str] = Field(None)
    legend_id: int = Field(-1)
    match_score: float = Field(0.0)
    legend_color_name: str = Field("unknown")
    legend_color_rgb: List[int] = Field(default_factory=list)


class RegionMatch(BaseModel):
    legend_id: int
    match_degree: float
    raw: float
    dE: float
    hist: float
    legend_rgb: List[int]
    legend_name: str


class RegionItem(BaseModel):
    id: int
    contour: List[List[float]] = Field(default_factory=list)
    centroid: List[float] = Field(default_factory=lambda: [-1.0, -1.0])
    area: float = Field(-1.0)
    matched_legend_id: int = Field(-1)
    match_score: float = Field(0.0)
    geo: RegionGeoInfo = Field(default_factory=RegionGeoInfo)
    region_color_rgb: List[int] = Field(default_factory=lambda: [0, 0, 0])
    top_matches: List[RegionMatch] = Field(default_factory=list)


class ImageAnalysisResult(BaseModel):
    summary: str
    features: List[Dict[str, Any]] = []
    legends: List[LegendItem] = []
    regions: List[RegionItem] = []
    annotated_map_base64: Optional[str] = None
    cropped_legend_base64: Optional[str] = None


class AgentState(TypedDict):
    image_path: str
    instruction: str
    analysis: Optional[ImageAnalysisResult]


# ==========================================
# 4. 辅助函数
# ==========================================

def safe_parse_json(text: str) -> Optional[dict]:
    if not text: return None
    text = re.sub(r"```json|```", "", text, flags=re.IGNORECASE).strip()
    try:
        return json.loads(text)
    except:
        pass
    match = re.search(r"\{[\s\S]*\}", text)
    if match:
        try:
            return json.loads(match.group())
        except:
            pass
    return None


def _ensure_text_fallback(txt):
    return txt if isinstance(txt, dict) else {"legend_text": "", "confidence": 0.0}


def _ensure_symbol_fallback(sym):
    if isinstance(sym, dict): return sym
    return {"final": "Unknown", "confidence": 0.0, "base": None, "superscript": None, "subscript": None,
            "terminal": "Unknown", "html": "", "latex": "Unknown"}


def convert_for_json(obj):
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {k: convert_for_json(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_for_json(v) for v in obj]
    return obj


def bgr_to_rgb(bgr):
    return [int(bgr[2]), int(bgr[1]), int(bgr[0])]


def nonwhite_mask_u8(patch_rgb, white_thresh=245):
    flat = patch_rgb.reshape(-1, 3)
    keep = (np.any(flat < white_thresh, axis=1)).astype(np.uint8) * 255
    return keep.reshape(patch_rgb.shape[0], patch_rgb.shape[1])


def compute_texture_feature(image_rgb, mask=None):
    if image_rgb is None or image_rgb.size == 0: return None
    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
    radius = 1
    n_points = 8 * radius
    lbp = local_binary_pattern(gray, n_points, radius, method='uniform')
    n_bins = n_points + 2
    if mask is not None:
        if mask.dtype != bool: mask = mask.astype(bool)
        if np.sum(mask) == 0: return np.zeros(n_bins, dtype=np.float32)
        hist, _ = np.histogram(lbp[mask], bins=n_bins, range=(0, n_bins), density=True)
    else:
        hist, _ = np.histogram(lbp.ravel(), bins=n_bins, range=(0, n_bins), density=True)
    return hist.astype(np.float32)


# ==========================================
# 6. Image Agent 类
# ==========================================

class ImageAgent:
    def __init__(self, api_key: str, model_name: str = "qwen-vl-max"):
        self.model_name = model_name
        self.llm = ChatTongyi(model=model_name, dashscope_api_key=api_key, temperature=0.01)

        self.output_dir = os.path.join(current_dir, "output")
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

        self.graph = self._build_graph()

    # === JPEG 自适应高速渲染方法 ===
    def _save_fast_vis(self, img_bgr, path, max_dim=4096):
        """跳过缓慢的无损 PNG 编码，将超高清结果强制缩放并在毫秒级编码为 JPG 存储"""
        h, w = img_bgr.shape[:2]
        if max(h, w) > max_dim:
            scale = max_dim / max(h, w)
            new_w, new_h = int(w * scale), int(h * scale)
            img_bgr = cv2.resize(img_bgr, (new_w, new_h), interpolation=cv2.INTER_AREA)
        jpg_path = path.replace(".png", ".jpg")
        cv2.imencode('.jpg', img_bgr, [cv2.IMWRITE_JPEG_QUALITY, 85])[1].tofile(jpg_path)

    def _resize_and_compress(self, img_bgr, max_dim=API_IMAGE_MAX_DIM, quality=80):
        h, w = img_bgr.shape[:2]
        if max(h, w) > max_dim:
            scale = max_dim / max(h, w)
            new_w, new_h = int(w * scale), int(h * scale)
            img_bgr = cv2.resize(img_bgr, (new_w, new_h), interpolation=cv2.INTER_AREA)
        _, buffer = cv2.imencode(".jpg", img_bgr, [cv2.IMWRITE_JPEG_QUALITY, quality])
        return base64.b64encode(buffer.tobytes()).decode("utf-8")

    def _img_to_base64(self, img_rgb):
        try:
            img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
            return self._resize_and_compress(img_bgr)
        except Exception:
            return ""

    def _file_to_base64(self, path):
        try:
            img_bgr = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
            if img_bgr is not None:
                return self._resize_and_compress(img_bgr)
        except Exception:
            pass
        try:
            with open(path, "rb") as f:
                return base64.b64encode(f.read()).decode("utf-8")
        except Exception:
            return ""

    def _extract_content_str(self, content_obj: Any) -> str:
        if isinstance(content_obj, str): return content_obj
        if isinstance(content_obj, list):
            parts = []
            for item in content_obj:
                if isinstance(item, dict) and 'text' in item:
                    parts.append(item['text'])
                elif isinstance(item, str):
                    parts.append(item)
            return "".join(parts)
        return str(content_obj)

    def _call_vlm(self, image_data, prompt):
        try:
            if isinstance(image_data, np.ndarray):
                b64_str = self._img_to_base64(image_data)
                img_content = f"data:image/jpeg;base64,{b64_str}"
            else:
                b64_str = self._file_to_base64(image_data) if (
                        len(image_data) < 1000 and os.path.exists(image_data)) else image_data
                if b64_str.startswith("data:image"):
                    img_content = b64_str
                else:
                    img_content = f"data:image/jpeg;base64,{b64_str}"

            messages = [
                HumanMessage(content=[{"type": "text", "text": prompt}, {"type": "image", "image": img_content}])]
            resp = self.llm.invoke(messages)
            content_str = self._extract_content_str(resp.content)
            return safe_parse_json(content_str) or {}
        except Exception:
            return {}

    def _analyze_node(self, state: AgentState):
        image_path = state["image_path"]
        instruction = state["instruction"]

        if not image_path or not os.path.exists(image_path):
            return {"analysis": ImageAnalysisResult(summary="未找到图片文件", legends=[])}

        filename = os.path.basename(image_path)
        print(f">>> [ImageAgent] 图例理解与区域分割智能体启动，解析文件: {filename}")

        is_fast_mode = any(k in instruction for k in ["图例顺序", "钻孔", "地层代号"])

        if is_fast_mode and HAS_PEACE:
            try:
                comp_det = map_component_detector()
                components = comp_det.detect(image_path)
                legend_regions = components.get("legend", [])

                if legend_regions:
                    legend_bndbox = legend_regions[0]
                    x0, y0, x1, y1 = [int(v) for v in legend_bndbox]
                    orig_img_cv = cv2.imdecode(np.fromfile(image_path, dtype=np.uint8), cv2.IMREAD_COLOR)
                    h, w = orig_img_cv.shape[:2]
                    x0, y0 = max(0, x0), max(0, y0)
                    x1, y1 = min(w, x1), min(h, y1)

                    cropped_legend = orig_img_cv[y0:y1, x0:x1]
                    b64_str = self._resize_and_compress(cropped_legend)

                    fast_prompt = (
                        "你是一个专业的地质专家。这是一张地质图的图例区域截图。\n"
                        "请严格按照图中图例【从上到下】（代表地层由新到老）的排列顺序，"
                        "一次性提取出所有的【地层代号/符号】和对应的【地层名称/岩性】。\n"
                        "请直接以清晰的列表形式输出（如：1. Qp - 更新统），不要遗漏，无需提取颜色与坐标，也不要任何多余废话。"
                    )

                    messages = [
                        HumanMessage(content=[
                            {"type": "text", "text": fast_prompt},
                            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64_str}"}}
                        ])
                    ]
                    resp = self.llm.invoke(messages)
                    summary = self._extract_content_str(resp.content)

                    return {"analysis": ImageAnalysisResult(
                        summary=f"【虚拟钻孔极速图例提取】\n{summary}",
                        legends=[],
                        regions=[]
                    )}
            except Exception:
                pass

        if not (HAS_CV_LIBS and HAS_PEACE):
            return self._simple_analysis(image_path, instruction)

        try:
            comp_det = map_component_detector()
            components = comp_det.detect(image_path)
            legend_regions = components.get("legend", [])

            extracted_legends = []
            basic_legends_list = []

            annotated_map_b64 = None
            cropped_legend_b64 = None
            run_dir = self.output_dir
            folder_name = "output"

            if legend_regions:
                timestamp_str = str(int(time.time()))
                base_name = os.path.splitext(os.path.basename(image_path))[0]
                folder_name = f"{base_name}_{timestamp_str}"

                run_dir = os.path.join(self.output_dir, folder_name)
                os.makedirs(run_dir, exist_ok=True)

                legend_items_dir = os.path.join(run_dir, "legend_items")
                os.makedirs(legend_items_dir, exist_ok=True)

                orig_img_cv = cv2.imdecode(np.fromfile(image_path, dtype=np.uint8), cv2.IMREAD_COLOR)
                if orig_img_cv is None: raise ValueError("无法解码原图像文件。")

                h, w = orig_img_cv.shape[:2]

                map_regions = components.get("map", []) or components.get("Map", []) or components.get("drawing", [])
                if map_regions:
                    mx0, my0, mx1, my1 = [int(v) for v in map_regions[0]]
                    mx0, my0 = max(0, mx0), max(0, my0)
                    mx1, my1 = min(w, mx1), min(h, my1)
                else:
                    mx0, my0, mx1, my1 = 0, 0, w, h

                legend_bndbox = legend_regions[0]
                x0, y0, x1, y1 = [int(v) for v in legend_bndbox]
                x0, y0 = max(0, x0), max(0, y0)
                x1, y1 = min(w, x1), min(h, y1)

                cropped_legend = orig_img_cv[y0:y1, x0:x1]
                temp_legend_path = os.path.join(run_dir, f"temp_legend_{timestamp_str}.png")
                cv2.imencode('.png', cropped_legend)[1].tofile(temp_legend_path)

                leg_det = map_legend_detector()
                legends_dict = leg_det.detect(temp_legend_path)

                if os.path.exists(temp_legend_path): os.remove(temp_legend_path)

                lh, lw = cropped_legend.shape[:2]
                annotated_img = orig_img_cv.copy()
                annotated_legend_img = cropped_legend.copy()

                for k, info in legends_dict.items():
                    cb = info.get("color_bndbox", [])
                    tb = info.get("text_bndbox", [])
                    if not isinstance(cb, (list, tuple)) or len(cb) != 4: cb = [0, 0, 0, 0]
                    if not isinstance(tb, (list, tuple)) or len(tb) != 4: tb = [0, 0, 0, 0]

                    cx0, cy0, cx1, cy1 = [int(v) for v in cb]
                    tx0, ty0, tx1, ty1 = [int(v) for v in tb]

                    valid_c = cx1 > cx0 and cy1 > cy0
                    valid_t = tx1 > tx0 and cy1 > ty0

                    abs_color_bbox = [x0 + cx0, y0 + cy0, x0 + cx1, y0 + cy1] if valid_c else [-1, -1, -1, -1]
                    abs_text_bbox = [x0 + tx0, y0 + ty0, x0 + tx1, y0 + ty1] if valid_t else [-1, -1, -1, -1]

                    info["color_bndbox"] = abs_color_bbox
                    info["text_bndbox"] = abs_text_bbox
                    info["text"] = ""

                    if valid_c: cv2.rectangle(annotated_img, (abs_color_bbox[0], abs_color_bbox[1]),
                                              (abs_color_bbox[2], abs_color_bbox[3]), (0, 0, 255), 2)
                    if valid_t: cv2.rectangle(annotated_img, (abs_text_bbox[0], abs_text_bbox[1]),
                                              (abs_text_bbox[2], abs_text_bbox[3]), (0, 0, 255), 2)

                    color_patch = None
                    avg_color = [127, 127, 127]
                    lbp_hist = None
                    color_area = 0.0

                    if valid_c:
                        mcx0, mcy0 = max(0, cx0), max(0, cy0)
                        mcx1, mcy1 = min(lw, cx1), min(lh, cy1)
                        color_patch = cropped_legend[mcy0:mcy1, mcx0:mcx1]
                        color_area = float((mcx1 - mcx0) * (mcy1 - mcy0))

                        if color_patch.size > 0:
                            color_filename = f"legend_color_{k}.png"
                            color_path = os.path.join(legend_items_dir, color_filename)
                            cv2.imencode('.png', color_patch)[1].tofile(color_path)

                            color_patch_rgb = cv2.cvtColor(color_patch, cv2.COLOR_BGR2RGB)
                            mask_u8 = nonwhite_mask_u8(color_patch_rgb, WHITE_THRESH)
                            valid_pixels = color_patch_rgb[mask_u8 == 255]
                            if len(valid_pixels) > 0:
                                avg_color = np.median(valid_pixels, axis=0).astype(int).tolist()
                            lbp_hist = compute_texture_feature(color_patch_rgb, mask=(mask_u8 == 255))

                    info["color"] = avg_color

                    text_patch = None
                    text_patch_rgb = None

                    if valid_t:
                        pad = 4
                        ptx0, pty0 = max(0, tx0 - pad), max(0, ty0 - pad)
                        ptx1, pty1 = min(lw, tx1 + pad), min(lh, ty1 + pad)
                        text_patch = cropped_legend[pty0:pty1, ptx0:ptx1]
                        if text_patch.size > 0:
                            text_filename = f"legend_text_{k}.png"
                            text_path = os.path.join(legend_items_dir, text_filename)
                            cv2.imencode('.png', text_patch)[1].tofile(text_path)
                        text_patch_rgb = cv2.cvtColor(text_patch, cv2.COLOR_BGR2RGB)

                    rel_color_img = f"output/{folder_name}/legend_items/legend_color_{k}.png" if valid_c else ""
                    rel_text_img = f"output/{folder_name}/legend_items/legend_text_{k}.png" if valid_t else ""

                    basic_legend = {
                        "id": int(k), "color_bbox": abs_color_bbox, "text_bbox": abs_text_bbox,
                        "avg_color": avg_color, "color_name": info.get("color_name", "unknown"),
                        "area": color_area, "color_img": rel_color_img, "text_img": rel_text_img,
                        "hist": lbp_hist.tolist() if lbp_hist is not None else []
                    }
                    basic_legends_list.append(basic_legend)

                    leg_item = LegendItem(
                        id=int(k), color_bbox=abs_color_bbox, text_bbox=abs_text_bbox,
                        color_name=info.get("color_name", "unknown"), avg_color=avg_color,
                        area=color_area, color_img=rel_color_img, text_img=rel_text_img,
                        hist=lbp_hist.tolist() if lbp_hist is not None else [],
                        legend_text="", symbol=GeoSymbol(final="Unknown", terminal="Unknown"),
                        lbp=lbp_hist,
                        color_patch_base64=self._img_to_base64(color_patch_rgb) if color_patch is not None else None,
                        text_patch_base64=self._img_to_base64(text_patch_rgb) if text_patch is not None else None,
                        is_recognized=False
                    )
                    extracted_legends.append(leg_item)

                extracted_legends.sort(key=lambda x: x.color_bbox[1] if (
                            x.color_bbox and len(x.color_bbox) == 4 and x.color_bbox[1] != -1) else float('inf'))

                vis_seg_optimized = orig_img_cv.copy()
                vis_seg_reinforced = orig_img_cv.copy()
                all_regions_ui = []
                all_region_matches = []

                try:
                    if len(extracted_legends) > 0 and mx1 > mx0 and my1 > my0:
                        map_patch = orig_img_cv[my0:my1, mx0:mx1]

                        legend_rgb_colors = []
                        legend_bgr_colors = []
                        for leg in extracted_legends:
                            c = leg.avg_color
                            legend_rgb_colors.append(c)
                            legend_bgr_colors.append([c[2], c[1], c[0]])

                        legend_colors_rgb = np.array(legend_rgb_colors, dtype=np.int32)
                        legend_colors_bgr = np.array(legend_bgr_colors, dtype=np.uint8)

                        yolo_success = False
                        region_id_counter = 1

                        # === 🚀 局部 BBox 极速提取优化（YOLO 分支） ===
                        if HAS_YOLO:
                            try:
                                yolo_model_path = os.getenv("YOLO_SEG_MODEL", "yolov10n-seg.pt")
                                seg_model = YOLO(yolo_model_path)
                                yolo_results = seg_model(map_patch, verbose=False)

                                if yolo_results and len(yolo_results) > 0 and yolo_results[0].masks is not None:
                                    masks_xy = yolo_results[0].masks.xy

                                    map_patch_optimized = np.full(map_patch.shape, 255, dtype=np.uint8)
                                    map_patch_reinforced = map_patch.copy()

                                    for cnt_arr in masks_xy:
                                        if len(cnt_arr) < 3: continue
                                        cnt = cnt_arr.astype(np.int32)
                                        area = cv2.contourArea(cnt)
                                        if area < 800: continue

                                        # 🚀 优化 1：使用 cv2.approxPolyDP 进行多边形抽稀，防止内存及 JSON 爆炸
                                        epsilon = 1.5
                                        cnt = cv2.approxPolyDP(cnt, epsilon, True)

                                        # 🚀 优化 2：禁止全图尺寸 np.zeros，改为局部极小包围盒处理
                                        x_b, y_b, w_b, h_b = cv2.boundingRect(cnt)
                                        if w_b <= 0 or h_b <= 0: continue

                                        small_mask = np.zeros((h_b, w_b), dtype=np.uint8)
                                        shifted_cnt = cnt - [x_b, y_b]
                                        cv2.drawContours(small_mask, [shifted_cnt], -1, 255, -1)
                                        valid_pixels = map_patch[y_b:y_b + h_b, x_b:x_b + w_b][small_mask == 255]

                                        if len(valid_pixels) == 0: continue

                                        region_bgr = np.median(valid_pixels, axis=0).astype(int)
                                        region_rgb = np.array([region_bgr[2], region_bgr[1], region_bgr[0]],
                                                              dtype=np.int32)

                                        dists = np.sum(np.abs(legend_colors_rgb - region_rgb), axis=1)
                                        best_idx = np.argmin(dists)

                                        leg = extracted_legends[best_idx]
                                        target_bgr = legend_bgr_colors[best_idx]

                                        cv2.drawContours(map_patch_optimized, [cnt], -1, target_bgr.tolist(), -1)
                                        cv2.drawContours(map_patch_reinforced, [cnt], -1, target_bgr.tolist(), -1)
                                        cv2.drawContours(map_patch_reinforced, [cnt], -1, (0, 0, 0), 2)

                                        cnt_offset = cnt + np.array([[mx0, my0]])
                                        M = cv2.moments(cnt_offset)
                                        cx = M["m10"] / M["m00"] if M["m00"] != 0 else 0.0
                                        cy = M["m01"] / M["m00"] if M["m00"] != 0 else 0.0

                                        contour_list = cnt_offset.squeeze().tolist() if cnt_offset.ndim == 3 else cnt_offset.tolist()

                                        all_regions_ui.append({
                                            "id": region_id_counter, "contour": contour_list,
                                            "centroid": [cx, cy], "area": float(area),
                                            "matched_legend_id": leg.id, "match_score": 0.95,
                                            "geo": {
                                                "unit_name": leg.legend_text, "legend_id": leg.id,
                                                "legend_color_name": leg.color_name, "legend_color_rgb": leg.avg_color
                                            },
                                            "region_color_rgb": leg.avg_color
                                        })

                                        # === 🚀 恢复完整字段收集 (YOLO 分支) ===
                                        all_region_matches.append({
                                            "region_id": region_id_counter, "matched_legend_id": leg.id,
                                            "match_score": 0.95,
                                            "symbol_final": leg.symbol.final, "symbol_terminal": leg.symbol.terminal,
                                            "symbol_html": leg.symbol.html, "symbol_latex": leg.symbol.latex,
                                            "symbol_conf": leg.symbol.confidence, "legend_text": leg.legend_text,
                                            "legend_color_name": leg.color_name, "legend_color_rgb": leg.avg_color,
                                            "area_px": float(area), "centroid_x": cx, "centroid_y": cy
                                        })
                                        region_id_counter += 1

                                    vis_seg_optimized[my0:my1, mx0:mx1] = map_patch_optimized
                                    vis_seg_reinforced[my0:my1, mx0:mx1] = map_patch_reinforced
                                    yolo_success = True
                            except Exception:
                                yolo_success = False

                        # === 🚀 自适应降维提取优化（曼哈顿色彩回退分支） ===
                        if not yolo_success:
                            print(">>> [ImageAgent] 启用自适应曼哈顿矩阵降维分割提速引擎...")

                            # 🚀 优化 3：工作分辨率降维（最高不超过 3000px，极大缩减内存消耗）
                            scale_ratio = 1.0
                            MAX_WORK_DIM = 3000
                            h_m, w_m = map_patch.shape[:2]
                            if max(h_m, w_m) > MAX_WORK_DIM:
                                scale_ratio = MAX_WORK_DIM / max(h_m, w_m)
                                work_patch = cv2.resize(map_patch, (int(w_m * scale_ratio), int(h_m * scale_ratio)),
                                                        interpolation=cv2.INTER_AREA)
                            else:
                                work_patch = map_patch.copy()

                            color_threds = []
                            for i, c1 in enumerate(legend_colors_rgb):
                                min_d = 256 * 3
                                for j, c2 in enumerate(legend_colors_rgb):
                                    if i == j: continue
                                    d = abs(c1[0] - c2[0]) + abs(c1[1] - c2[1]) + abs(c1[2] - c2[2])
                                    if d < min_d: min_d = d
                                color_threds.append(max(10, min_d / 2.0))
                            color_threds = np.array(color_threds, dtype=np.float32)

                            img_flat_rgb = cv2.cvtColor(work_patch, cv2.COLOR_BGR2RGB).reshape(-1, 3).astype(np.int32)
                            result_flat_bgr = np.full((img_flat_rgb.shape[0], 3), 255, dtype=np.uint8)

                            batch_size = 2000000
                            for i in range(0, img_flat_rgb.shape[0], batch_size):
                                batch = img_flat_rgb[i:i + batch_size]
                                dists = np.sum(np.abs(batch[:, None, :] - legend_colors_rgb[None, :, :]), axis=2)
                                best_idx = np.argmin(dists, axis=1)
                                min_dists = np.min(dists, axis=1)
                                threds_for_best = color_threds[best_idx]
                                valid_mask = min_dists < threds_for_best
                                batch_result = result_flat_bgr[i:i + batch_size]
                                batch_result[valid_mask] = legend_colors_bgr[best_idx[valid_mask]]
                                result_flat_bgr[i:i + batch_size] = batch_result

                            map_patch_optimized_small = result_flat_bgr.reshape(work_patch.shape)
                            map_patch_optimized = cv2.resize(map_patch_optimized_small, (w_m, h_m),
                                                             interpolation=cv2.INTER_NEAREST)
                            vis_seg_optimized[my0:my1, mx0:mx1] = map_patch_optimized

                            # 形态学操作在降采样的小图上进行，提速百倍
                            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
                            map_patch_reinforced_small = cv2.morphologyEx(map_patch_optimized_small, cv2.MORPH_CLOSE,
                                                                          kernel)
                            map_patch_reinforced_small = cv2.morphologyEx(map_patch_reinforced_small, cv2.MORPH_OPEN,
                                                                          kernel)

                            gray_map_small = cv2.cvtColor(work_patch, cv2.COLOR_BGR2GRAY)
                            edges_small = cv2.Canny(gray_map_small, 50, 150)
                            edges_small_dilated = cv2.dilate(edges_small, np.ones((2, 2), np.uint8), iterations=1)
                            map_patch_reinforced_small[edges_small_dilated == 255] = [0, 0, 0]

                            map_patch_reinforced = cv2.resize(map_patch_reinforced_small, (w_m, h_m),
                                                              interpolation=cv2.INTER_NEAREST)
                            vis_seg_reinforced[my0:my1, mx0:mx1] = map_patch_reinforced

                            for idx, leg in enumerate(extracted_legends):
                                target_bgr = legend_colors_bgr[idx]
                                target_int = target_bgr.astype(int)
                                lower = np.clip(target_int - 2, 0, 255).astype(np.uint8)
                                upper = np.clip(target_int + 2, 0, 255).astype(np.uint8)

                                mask = cv2.inRange(map_patch_reinforced_small, lower, upper)
                                cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                                for cnt in cnts:
                                    area_small = cv2.contourArea(cnt)
                                    if area_small < 800 * (scale_ratio ** 2): continue

                                    epsilon = 1.0
                                    approx = cv2.approxPolyDP(cnt, epsilon, True)
                                    if len(approx) < 3: continue

                                    # 🚀 优化 4：将小图多边形完美映射回原图超清坐标系
                                    approx_orig = (approx / scale_ratio).astype(np.int32)
                                    approx_offset = approx_orig + np.array([[[mx0, my0]]])

                                    M = cv2.moments(approx_offset)
                                    cx = M["m10"] / M["m00"] if M["m00"] != 0 else 0.0
                                    cy = M["m01"] / M["m00"] if M["m00"] != 0 else 0.0

                                    contour_list = approx_offset.squeeze().tolist() if approx_offset.ndim == 3 else approx_offset.tolist()

                                    all_regions_ui.append({
                                        "id": region_id_counter, "contour": contour_list,
                                        "centroid": [cx, cy], "area": float(area_small / (scale_ratio ** 2)),
                                        "matched_legend_id": leg.id, "match_score": 0.95,
                                        "geo": {
                                            "unit_name": leg.legend_text, "legend_id": leg.id,
                                            "legend_color_name": leg.color_name, "legend_color_rgb": leg.avg_color
                                        },
                                        "region_color_rgb": leg.avg_color
                                    })

                                    # === 🚀 恢复完整字段收集 (曼哈顿降维分支) ===
                                    all_region_matches.append({
                                        "region_id": region_id_counter, "matched_legend_id": leg.id,
                                        "match_score": 0.95,
                                        "symbol_final": leg.symbol.final, "symbol_terminal": leg.symbol.terminal,
                                        "symbol_html": leg.symbol.html, "symbol_latex": leg.symbol.latex,
                                        "symbol_conf": leg.symbol.confidence, "legend_text": leg.legend_text,
                                        "legend_color_name": leg.color_name, "legend_color_rgb": leg.avg_color,
                                        "area_px": float(area_small / (scale_ratio ** 2)), "centroid_x": cx,
                                        "centroid_y": cy
                                    })
                                    region_id_counter += 1

                except Exception as e:
                    import traceback
                    traceback.print_exc()

                hie_meta = {
                    "date": time.strftime("%Y-%m-%d"),
                    "name": base_name,
                    "size": {"width": int(w), "height": int(h)},
                    "legend": legends_dict
                }
                with open(os.path.join(run_dir, "meta.json"), "w", encoding="utf-8") as f:
                    json.dump(convert_for_json(hie_meta), f, ensure_ascii=False, indent=4)

                # === 🚀 优化 5：使用自研加速管线将 2亿 像素图片瞬间渲染存盘 ===
                self._save_fast_vis(annotated_img, os.path.join(run_dir, "legend_items_detected.jpg"))
                self._save_fast_vis(vis_seg_optimized, os.path.join(run_dir, "vis_seg_optimized.jpg"))
                self._save_fast_vis(vis_seg_reinforced, os.path.join(run_dir, "vis_seg_reinforced.jpg"))

                with open(os.path.join(run_dir, "legend_info.json"), "w", encoding="utf-8") as f:
                    json.dump(convert_for_json(basic_legends_list), f, ensure_ascii=False, indent=4)

                gemini_data = []
                for item in extracted_legends:
                    try:
                        d = item.model_dump(exclude={'color_patch_base64', 'text_patch_base64'},
                                            by_alias=True) if hasattr(item, "model_dump") else item.dict(
                            exclude={'color_patch_base64', 'text_patch_base64'}, by_alias=True)
                        gemini_data.append(d)
                    except:
                        pass

                safe_model_name = re.sub(r'[^a-zA-Z0-9_\-]', '_', self.model_name)
                json_filename = f"legend_info_with_{safe_model_name}.json"
                with open(os.path.join(run_dir, json_filename), "w", encoding="utf-8") as f:
                    json.dump(convert_for_json(gemini_data), f, ensure_ascii=False, indent=4)

                with open(os.path.join(run_dir, "regions_ui.json"), "w", encoding="utf-8") as f:
                    json.dump(convert_for_json(all_regions_ui), f, ensure_ascii=False, indent=2)

                # === 🚀 恢复落盘：region_matches_symbols 的 JSON 与 CSV 文件 ===
                with open(os.path.join(run_dir, "region_matches_symbols.json"), "w", encoding="utf-8") as f:
                    json.dump(convert_for_json(all_region_matches), f, ensure_ascii=False, indent=2)

                with open(os.path.join(run_dir, "region_matches_symbols.csv"), "w", encoding="utf-8-sig",
                          newline="") as f:
                    writer = csv.writer(f)
                    writer.writerow([
                        "region_id", "matched_legend_id", "match_score",
                        "symbol_final", "symbol_terminal", "symbol_html",
                        "symbol_latex", "symbol_conf", "legend_text",
                        "legend_color_name", "legend_color_rgb",
                        "area_px", "centroid_x", "centroid_y"
                    ])
                    for match in all_region_matches:
                        writer.writerow([
                            match["region_id"], match["matched_legend_id"], match["match_score"],
                            match.get("symbol_final", ""), match.get("symbol_terminal", ""),
                            match.get("symbol_html", ""),
                            match.get("symbol_latex", ""), match.get("symbol_conf", 0.0), match["legend_text"],
                            match["legend_color_name"], json.dumps(match.get("legend_color_rgb", [])),
                            match["area_px"], match["centroid_x"], match["centroid_y"]
                        ])

                annotated_map_b64 = self._resize_and_compress(annotated_img)
                cropped_legend_b64 = self._resize_and_compress(cropped_legend) if 'cropped_legend' in locals() else ""

            if not extracted_legends:
                raise ValueError("未提取到有效图例单元")

            summary = f"【空间图斑极速提取完成】共处理 {len(extracted_legends)} 个图例，分割并进行拓扑抽稀获得 {len(all_regions_ui)} 个空间多边形。高清结果渲染已保存。"
            print(
                f">>> [ImageAgent] 处理完成，共识别 {len(extracted_legends)} 个图例，分割 {len(all_regions_ui)} 个空间图斑。")

            return {"analysis": ImageAnalysisResult(
                summary=summary, legends=extracted_legends, regions=[],
                annotated_map_base64=annotated_map_b64, cropped_legend_base64=cropped_legend_b64
            )}

        except Exception as e:
            return self._simple_analysis(image_path, instruction)

    def _recognize_single_legend(self, legend: LegendItem):
        if legend.is_recognized: return legend
        GEOLOGY_PROMPT = "你是一个地质专家。请识别图例色块中的地质符号（包含上下标）。请严格输出纯JSON格式数据：{\"base\": \"\", \"superscript\": \"\", \"subscript\": \"\", \"symbol_final\": \"\", \"symbol_terminal\": \"\", \"confidence\": 0.99}。绝不要输出其他废话。"
        LEGEND_TEXT_PROMPT = "你是一个地质专家。请识别图例文字说明（即地层名称、岩性等）。请严格输出纯JSON格式数据：{\"legend_text\": \"\", \"confidence\": 0.99}。绝不要输出其他废话。"

        if legend.text_patch_base64:
            txt_res = self._call_vlm(legend.text_patch_base64, LEGEND_TEXT_PROMPT)
            legend.legend_text = _ensure_text_fallback(txt_res).get("legend_text", "识别失败")

        if legend.color_patch_base64:
            sym_res = self._call_vlm(legend.color_patch_base64, GEOLOGY_PROMPT)
            legend.symbol = GeoSymbol(**_ensure_symbol_fallback(sym_res))

        legend.is_recognized = True
        return legend

    def _parse_legend_input(self, legends_input: List[Union[Dict, LegendItem]]) -> List[LegendItem]:
        parsed_legends = []
        for item in legends_input:
            if isinstance(item, LegendItem):
                parsed_legends.append(item)
            elif isinstance(item, dict):
                try:
                    parsed_legends.append(LegendItem(**item))
                except Exception:
                    pass
        return parsed_legends

    def identify_clicked_region(self, image_path: str, click_point: tuple,
                                legends_input: List[Union[Dict, LegendItem]]) -> Optional[RegionItem]:
        if not os.path.exists(image_path) or not legends_input: return None

        try:
            legends = self._parse_legend_input(legends_input)
            image_bgr = cv2.imdecode(np.fromfile(image_path, dtype=np.uint8), cv2.IMREAD_COLOR)
            if image_bgr is None: return None
            image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
            h, w = image_rgb.shape[:2]

            cx, cy = int(click_point[0]), int(click_point[1])
            if cx < 0 or cx >= w or cy < 0 or cy >= h: return None

            y1, y2 = max(0, cy - 2), min(h, cy + 3)
            x1, x2 = max(0, cx - 2), min(w, cx + 3)
            patch = image_rgb[y1:y2, x1:x2]
            click_color = np.median(patch.reshape(-1, 3), axis=0).astype(int)

            click_arr = np.array(click_color, dtype=np.float32)
            candidates = []
            for leg in legends:
                leg_rgb = np.array(leg.avg_color, dtype=np.float32)
                dist = np.linalg.norm(click_arr - leg_rgb)
                candidates.append((dist, leg))

            candidates.sort(key=lambda x: x[0])
            final_match = candidates[0][1]
            best_score = float(candidates[0][0])

            if not final_match.is_recognized:
                if image_bgr is not None:
                    cb = final_match.color_bbox
                    tb = final_match.text_bbox
                    if cb and len(cb) == 4 and cb[2] > cb[0] and cb[3] > cb[1]:
                        final_match.color_patch_base64 = self._resize_and_compress(image_bgr[cb[1]:cb[3], cb[0]:cb[2]])
                    if tb and len(tb) == 4 and tb[2] > tb[0] and tb[3] > tb[1]:
                        final_match.text_patch_base64 = self._resize_and_compress(image_bgr[tb[1]:tb[3], tb[0]:tb[2]])

                final_match = self._recognize_single_legend(final_match)

            target_color = np.array(final_match.avg_color)
            lower = np.clip(target_color - 30, 0, 255).astype(np.uint8)
            upper = np.clip(target_color + 30, 0, 255).astype(np.uint8)

            mask = cv2.inRange(image_rgb, lower, upper)
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

            cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            target_contour = None
            pt = (float(cx), float(cy))

            min_dist = float('inf')
            for c in cnts:
                dist = cv2.pointPolygonTest(c, pt, True)
                if dist >= 0:
                    target_contour = c
                    break
                elif abs(dist) < min_dist:
                    min_dist = abs(dist)
                    target_contour = c

            if target_contour is None:
                target_contour = np.array(
                    [[[cx - 10, cy - 10]], [[cx + 10, cy - 10]], [[cx + 10, cy + 10]], [[cx - 10, cy + 10]]])

            real_cnt = target_contour.squeeze().tolist() if target_contour.ndim == 3 else target_contour.tolist()
            if isinstance(real_cnt, list) and len(real_cnt) > 0 and not isinstance(real_cnt[0], list):
                real_cnt = [real_cnt]

            region_area = float(cv2.contourArea(target_contour)) if target_contour is not None else 0.0

            try:
                symbol_dict = final_match.symbol.model_dump(by_alias=True) if hasattr(final_match.symbol,
                                                                                      'model_dump') else final_match.symbol.dict(
                    by_alias=True)
            except Exception:
                symbol_dict = {}

            match_conf = round(max(0.0, 100.0 * (1.0 - best_score / 441.67)), 2)

            return RegionItem(
                id=0, contour=real_cnt, centroid=[float(cx), float(cy)],
                area=region_area, matched_legend_id=final_match.id, match_score=match_conf,
                geo=RegionGeoInfo(
                    symbol=symbol_dict, unit_name=final_match.legend_text, legend_id=final_match.id,
                    match_score=match_conf, legend_color_name=final_match.color_name,
                    legend_color_rgb=final_match.avg_color
                ),
                region_color_rgb=click_color.tolist(),
                top_matches=[RegionMatch(
                    legend_id=final_match.id, match_degree=float(best_score), raw=float(best_score),
                    dE=float(best_score), hist=0.0, legend_rgb=final_match.avg_color, legend_name=final_match.color_name
                )]
            )

        except Exception as e:
            print(f">>> [ImageAgent] 交互提取异常: {e}")
            return None

    def _simple_analysis(self, image_path, instruction):
        try:
            b64_str = self._file_to_base64(image_path)
            if not b64_str:
                return {"analysis": ImageAnalysisResult(summary="图像解析失败。", legends=[], regions=[])}

            messages = [HumanMessage(content=[{"type": "text", "text": f"分析地质图件：{instruction}"},
                                              {"type": "image_url",
                                               "image_url": {"url": f"data:image/jpeg;base64,{b64_str}"}}])]
            resp = self.llm.invoke(messages)

            print(f">>> [ImageAgent] 处理完成，基础全图直读分析完毕。")
            return {"analysis": ImageAnalysisResult(summary=self._extract_content_str(resp.content), legends=[],
                                                    regions=[])}

        except Exception as e:
            return {"analysis": ImageAnalysisResult(summary=f"模型失败: {str(e)}", legends=[], regions=[])}

    def _build_graph(self):
        workflow = StateGraph(AgentState)
        workflow.add_node("analyze_image", self._analyze_node)
        workflow.add_edge(START, "analyze_image")
        workflow.add_edge("analyze_image", END)
        return workflow.compile()

    def run(self, image_path: str, instruction: str = "请分析这张地质图件"):
        return self.graph.invoke({"image_path": image_path, "instruction": instruction})