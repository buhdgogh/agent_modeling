"""Geological map analysis, legend recognition and region matching."""
import os
import base64
import json
import re
import time
import csv
import logging
import math
import numpy as np
import cv2
from typing import List, Optional, TypedDict, Dict, Any, Union
from pydantic import ValidationError
from langchain_core.messages import HumanMessage
from langchain_community.chat_models import ChatTongyi
from agent_modeling.execution import invoke_model, invoke_tool, check_cancelled, worker_count, submit
from langgraph.graph import StateGraph, START, END
from agent_modeling.errors import FileValidationError, ModelInvocationError, error_code, public_error_message
from agent_modeling.config import DEFAULT_IMAGE_OUTPUT

from agent_modeling.models import GeoSymbol, ImageAnalysisResult, LegendItem, RegionGeoInfo, RegionItem, RegionMatch

logger = logging.getLogger(__name__)

# === 1. 依赖库导入与环境配置 (双轨制与 YOLO 前置探针) ===

try:
    from skimage import io as sio
    from skimage.feature import local_binary_pattern
    from shapely.geometry import Polygon

    HAS_CV_LIBS = True
except Exception:
    logger.warning("Optional image-processing dependencies are unavailable", exc_info=True)
    HAS_CV_LIBS = False

try:
    from PEACE.tool_pool.map_legend_detector import map_legend_detector
    from PEACE.tool_pool.map_component_detector import map_component_detector

    HAS_PEACE = True
except Exception:
    logger.warning("PEACE map detectors are unavailable", exc_info=True)
    HAS_PEACE = False

try:
    from ultralytics import YOLO

    HAS_YOLO = True
except Exception:
    logger.warning("Ultralytics YOLO is unavailable; color-based fallback will be used", exc_info=True)
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


class ImageAgentState(TypedDict):
    image_path: str
    instruction: str
    output_dir: str
    analysis: Optional[ImageAnalysisResult]


# ==========================================
# 4. 辅助函数
# ==========================================

def safe_parse_json(text: str) -> Optional[dict]:
    if not text: return None
    text = re.sub(r"```json|```", "", text, flags=re.IGNORECASE).strip()
    try:
        return json.loads(text)
    except (json.JSONDecodeError, TypeError):
        logger.debug("VLM response is not a complete JSON document")
    match = re.search(r"\{[\s\S]*\}", text)
    if match:
        try:
            return json.loads(match.group())
        except (json.JSONDecodeError, TypeError):
            logger.debug("Unable to recover JSON object from VLM response")
    return None


def _safe_text(value, default=""):
    return default if value is None else str(value)


def _safe_confidence(value):
    if isinstance(value, bool):
        return 0.0
    try:
        score = float(value)
    except (TypeError, ValueError, OverflowError):
        return 0.0
    return min(1.0, max(0.0, score)) if math.isfinite(score) else 0.0


def _ensure_text_fallback(txt):
    if isinstance(txt, dict):
        return {"legend_text": _safe_text(txt.get("legend_text")),
                "confidence": _safe_confidence(txt.get("confidence"))}
    return {"legend_text": "", "confidence": 0.0}


def _ensure_symbol_fallback(sym):
    if isinstance(sym, dict):
        normalized = dict(sym)
        for name in ("final", "terminal", "latex", "symbol_final", "symbol_terminal", "symbol_latex"):
            if name in normalized:
                normalized[name] = _safe_text(normalized[name], "Unknown")
        for name in ("html", "symbol_html"):
            if name in normalized:
                normalized[name] = _safe_text(normalized[name])
        for name in ("confidence", "symbol_conf"):
            if name in normalized:
                normalized[name] = _safe_confidence(normalized[name])
        for name in ("base", "superscript", "subscript"):
            if normalized.get(name) is not None:
                normalized[name] = _safe_text(normalized[name])
        return normalized
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


def _compute_match_score(legend_rgb, region_rgb, legend_hist=None, region_hist=None) -> float:
    """Compute a combined match score from color distance and optional texture similarity.
    Returns score in [0, 1], higher = better match."""
    # Color score: normalize L1 distance (0-765) to [0,1], invert so closer = higher
    l1_dist = float(sum(abs(a - b) for a, b in zip(legend_rgb, region_rgb)))
    color_score = max(0.0, 1.0 - l1_dist / 400.0)  # 400 = ~half max RGB distance

    # Texture score: histogram intersection (Bhattacharyya-like)
    texture_score = 0.5  # neutral default
    if legend_hist is not None and region_hist is not None:
        try:
            lh = np.array(legend_hist, dtype=np.float64)
            rh = np.array(region_hist, dtype=np.float64)
            lh = lh / (lh.sum() + 1e-10)
            rh = rh / (rh.sum() + 1e-10)
            # Histogram intersection
            texture_score = float(np.sum(np.minimum(lh, rh)))
        except Exception:
            logger.debug("Texture similarity calculation failed; using color score only", exc_info=True)

    # Weighted combination: 70% color, 30% texture (if available)
    if legend_hist is not None and region_hist is not None:
        return round(0.7 * color_score + 0.3 * texture_score, 4)
    return round(color_score, 4)


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

class ImageAnalysisAgent:
    def __init__(self, api_key: str, model_name: str = "qwen-vl-max"):
        self.model_name = model_name
        self.llm = ChatTongyi(model=model_name, dashscope_api_key=api_key, temperature=0.01, max_retries=1)
        self._yolo_model = None  # lazy-loaded, prevents repeated GitHub checks

        self.output_dir = str(DEFAULT_IMAGE_OUTPUT)
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
            logger.warning("Unable to encode in-memory image", exc_info=True)
            return ""

    def _file_to_base64(self, path):
        try:
            img_bgr = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
            if img_bgr is not None:
                return self._resize_and_compress(img_bgr)
        except Exception:
            logger.debug("OpenCV could not decode image; trying raw file encoding", exc_info=True)
        try:
            with open(path, "rb") as f:
                return base64.b64encode(f.read()).decode("utf-8")
        except Exception:
            logger.exception("Unable to read image file: %s", path)
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
            resp = invoke_model(self.llm, messages, name="image.recognize", model_name=self.model_name)
            content_str = self._extract_content_str(resp.content)
            return safe_parse_json(content_str) or {}
        except Exception:
            logger.exception("Vision-language model invocation or response parsing failed")
            return {}

    def _analyze_node(self, state: ImageAgentState):
        image_path = state["image_path"]
        instruction = state["instruction"]
        output_root = state.get("output_dir") or self.output_dir
        os.makedirs(output_root, exist_ok=True)

        if not image_path or not os.path.exists(image_path):
            wrapped = FileValidationError("Image file does not exist", details=str(image_path))
            logger.warning("Image file does not exist: %s", image_path)
            return {"analysis": ImageAnalysisResult(
                summary=public_error_message(wrapped), error_code=error_code(wrapped), legends=[]
            )}

        filename = os.path.basename(image_path)
        print(f">>> [ImageAnalysisAgent] 图例理解与区域分割智能体启动，解析文件: {filename}")

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
                        "逐项提取每个图例的完整信息，以纯JSON数组格式输出。\n\n"
                        "输出格式要求（严格JSON，不要Markdown标记，不要任何额外文字）:\n"
                        '[\n'
                        '  {"index": 1, "formation": "地层名称", "code": "地层代号", "color_desc": "色块颜色描述", "confidence": 0.95},\n'
                        '  {"index": 2, "formation": "地层名称", "code": "地层代号", "color_desc": "色块颜色描述", "confidence": 0.90}\n'
                        ']\n\n'
                        "填写规则:\n"
                        "- index: 从1开始递增\n"
                        "- formation: 图例文字中的地层名称或岩性描述\n"
                        "- code: 图例中的地层代号/符号（如 Q, Pt1, J3 等），如果没有则填 null\n"
                        "- color_desc: 色块的颜色描述（如'淡黄色''灰绿色''粉紫色'等）\n"
                        "- confidence: 当前图例文字与代号识别的证据置信度，取值0.0到1.0\n"
                        "- 请勿遗漏任何图例项，也勿编造不存在的项"
                    )

                    messages = [
                        HumanMessage(content=[
                            {"type": "text", "text": fast_prompt},
                            {"type": "image", "image": f"data:image/jpeg;base64,{b64_str}"}
                        ])
                    ]
                    resp = invoke_model(self.llm, messages, name="image.recognize", model_name=self.model_name)
                    raw_content = self._extract_content_str(resp.content)

                    # Parse JSON response to build structured LegendItems
                    fast_legends = []
                    try:
                        raw_clean = re.sub(r'```json|```', '', raw_content, flags=re.IGNORECASE).strip()
                        match = re.search(r'\[[\s\S]*\]', raw_clean)
                        if match:
                            parsed = json.loads(match.group())
                            for item in parsed:
                                confidence = _safe_confidence(item.get("confidence"))
                                leg = LegendItem(
                                    id=item.get("index", len(fast_legends)),
                                    avg_color=[200, 200, 200],  # placeholder, matched later
                                    legend_text=_safe_text(item.get("formation")),
                                    symbol=GeoSymbol(
                                        final=_safe_text(item.get("code"), "Unknown"),
                                        terminal=_safe_text(item.get("code"), "Unknown"),
                                        confidence=confidence
                                    ),
                                    is_recognized=True,
                                    legend_text_confidence=confidence,
                                    color_name=_safe_text(item.get("color_desc"), "unknown")
                                )
                                fast_legends.append(leg)
                    except (json.JSONDecodeError, TypeError, ValueError):
                        logger.warning("Fast legend response could not be parsed", exc_info=True)

                    summary = f"【虚拟钻孔极速图例提取】共识别 {len(fast_legends)} 个图例项（VLM结构化JSON）"
                    return {"analysis": ImageAnalysisResult(
                        summary=summary,
                        legends=fast_legends,
                        regions=[]
                    )}
            except Exception:
                logger.warning("Fast legend pipeline failed; switching to full analysis", exc_info=True)

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
            run_dir = output_root
            folder_name = os.path.basename(output_root)

            if legend_regions:
                timestamp_str = str(int(time.time()))
                base_name = os.path.splitext(os.path.basename(image_path))[0]
                folder_name = f"{base_name}_{timestamp_str}"

                run_dir = os.path.join(output_root, folder_name)
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
                    check_cancelled()
                    cb = info.get("color_bndbox", [])
                    tb = info.get("text_bndbox", [])
                    if not isinstance(cb, (list, tuple)) or len(cb) != 4: cb = [0, 0, 0, 0]
                    if not isinstance(tb, (list, tuple)) or len(tb) != 4: tb = [0, 0, 0, 0]

                    cx0, cy0, cx1, cy1 = [int(v) for v in cb]
                    tx0, ty0, tx1, ty1 = [int(v) for v in tb]

                    valid_c = cx1 > cx0 and cy1 > cy0
                    valid_t = tx1 > tx0 and ty1 > ty0

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

                    rel_color_img = os.path.join(legend_items_dir, f"legend_color_{k}.png") if valid_c else ""
                    rel_text_img = os.path.join(legend_items_dir, f"legend_text_{k}.png") if valid_t else ""

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

                                        # === 🚀 YOLO segmentation (lazy-load model once) ===
                        if HAS_YOLO:
                            try:
                                if not hasattr(self, '_yolo_model'):
                                    yolo_model_path = os.getenv("YOLO_SEG_MODEL", "yolov10n-seg.pt")
                                    self._yolo_model = YOLO(yolo_model_path)
                                yolo_results = self._yolo_model(map_patch, verbose=False)

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
                                        min_l1 = float(dists[best_idx])

                                        leg = extracted_legends[best_idx]
                                        target_bgr = legend_bgr_colors[best_idx]
                                        match_sc = _compute_match_score(leg.avg_color, region_rgb.tolist(),
                                                                        leg.hist if leg.hist else None)

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
                                            "matched_legend_id": leg.id, "match_score": match_sc,
                                            "geo": {
                                                "unit_name": leg.legend_text, "legend_id": leg.id,
                                                "legend_color_name": leg.color_name, "legend_color_rgb": leg.avg_color
                                            },
                                            "region_color_rgb": leg.avg_color
                                        })

                                        # === 🚀 恢复完整字段收集 (YOLO 分支) ===
                                        all_region_matches.append({
                                            "region_id": region_id_counter, "matched_legend_id": leg.id,
                                            "match_score": match_sc,
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
                                logger.warning("YOLO segmentation failed; using color fallback", exc_info=True)
                                yolo_success = False

                        # === 🚀 自适应降维提取优化（曼哈顿色彩回退分支） ===
                        if not yolo_success:
                            print(">>> [ImageAnalysisAgent] 启用自适应曼哈顿矩阵降维分割提速引擎...")

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
                                check_cancelled()
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

                                    mh_match = _compute_match_score(leg.avg_color, leg.avg_color, leg.hist if leg.hist else None)
                                    all_regions_ui.append({
                                        "id": region_id_counter, "contour": contour_list,
                                        "centroid": [cx, cy], "area": float(area_small / (scale_ratio ** 2)),
                                        "matched_legend_id": leg.id, "match_score": mh_match,
                                        "geo": {
                                            "unit_name": leg.legend_text, "legend_id": leg.id,
                                            "legend_color_name": leg.color_name, "legend_color_rgb": leg.avg_color
                                        },
                                        "region_color_rgb": leg.avg_color
                                    })

                                    # === 🚀 恢复完整字段收集 (曼哈顿降维分支) ===
                                    all_region_matches.append({
                                        "region_id": region_id_counter, "matched_legend_id": leg.id,
                                        "match_score": mh_match,
                                        "symbol_final": leg.symbol.final, "symbol_terminal": leg.symbol.terminal,
                                        "symbol_html": leg.symbol.html, "symbol_latex": leg.symbol.latex,
                                        "symbol_conf": leg.symbol.confidence, "legend_text": leg.legend_text,
                                        "legend_color_name": leg.color_name, "legend_color_rgb": leg.avg_color,
                                        "area_px": float(area_small / (scale_ratio ** 2)), "centroid_x": cx,
                                        "centroid_y": cy
                                    })
                                    region_id_counter += 1

                except Exception:
                    logger.exception("Map region segmentation failed")

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
                if 'cropped_legend' in locals():
                    self._save_fast_vis(cropped_legend, os.path.join(run_dir, "cropped_legend.jpg"))

                with open(os.path.join(run_dir, "legend_info.json"), "w", encoding="utf-8") as f:
                    json.dump(convert_for_json(basic_legends_list), f, ensure_ascii=False, indent=4)

                gemini_data = []
                for item in extracted_legends:
                    check_cancelled()
                    try:
                        d = item.model_dump(exclude={'color_patch_base64', 'text_patch_base64'},
                                            by_alias=True) if hasattr(item, "model_dump") else item.dict(
                            exclude={'color_patch_base64', 'text_patch_base64'}, by_alias=True)
                        gemini_data.append(d)
                    except Exception:
                        logger.warning("Unable to serialize a legend result; item skipped", exc_info=True)

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
                f">>> [ImageAnalysisAgent] 处理完成，共识别 {len(extracted_legends)} 个图例，分割 {len(all_regions_ui)} 个空间图斑。")

            return {"analysis": ImageAnalysisResult(
                summary=summary, legends=extracted_legends, regions=[],
                annotated_map_base64=annotated_map_b64, cropped_legend_base64=cropped_legend_b64,
                artifacts_dir=run_dir
            )}

        except Exception:
            logger.warning("Full image pipeline failed; switching to basic VLM analysis", exc_info=True)
            return self._simple_analysis(image_path, instruction)

    def _recognize_single_legend(self, legend: LegendItem):
        if legend.is_recognized: return legend
        GEOLOGY_PROMPT = (
            "你是一个地质专家。请识别图例色块中的地质符号（包含上下标）。\n"
            "请严格输出纯JSON: "
            '{"base": "", "superscript": "", "subscript": "", "symbol_final": "", "symbol_terminal": "", "confidence": 0.99}\n'
            "示例: 符号 Pt₁q → {\"base\":\"Pt\",\"superscript\":\"\",\"subscript\":\"1q\",\"symbol_final\":\"Pt₁q\",\"symbol_terminal\":\"Pt1q\",\"confidence\":0.95}\n"
            "绝不要输出其他废话。"
        )
        LEGEND_TEXT_PROMPT = (
            "你是一个地质专家。请识别图例文字说明，提取地层名称、岩性、年代信息。\n"
            "请严格输出纯JSON: "
            '{"formation": "地层名称", "lithology": "岩性描述", "age_text": "地质年代", "confidence": 0.99}\n'
            "示例: 文字 '堑头岩组 变粒岩、片麻岩 Pt₁' → "
            '{"formation":"堑头岩组","lithology":"变粒岩、片麻岩","age_text":"古元古代","confidence":0.95}\n'
            "如果没有某项信息，对应字段填 null。绝不要输出其他废话。"
        )

        if legend.text_patch_base64:
            txt_res = self._call_vlm(legend.text_patch_base64, LEGEND_TEXT_PROMPT)
            if isinstance(txt_res, dict):
                formation = _safe_text(txt_res.get("formation"))
                legend_text = formation or _safe_text(txt_res.get("legend_text"))
                legend.legend_text = legend_text
                legend.formation = formation or legend_text
                legend.lithology = _safe_text(txt_res.get("lithology"))
                legend.age_text = _safe_text(txt_res.get("age_text"))
                legend.legend_text_confidence = _safe_confidence(txt_res.get("confidence"))
            else:
                txt_dict = _ensure_text_fallback(txt_res)
                legend.legend_text = txt_dict.get("legend_text", "识别失败")
                legend.legend_text_confidence = float(txt_dict.get("confidence", 0.0))

        if legend.color_patch_base64:
            sym_res = self._call_vlm(legend.color_patch_base64, GEOLOGY_PROMPT)
            legend.symbol = GeoSymbol(**_ensure_symbol_fallback(sym_res))

        legend.is_recognized = True
        return legend

    def _parse_legend_input(self, legends_input: List[Union[Dict, LegendItem]]) -> List[LegendItem]:
        parsed_legends = []
        invalid_count = 0
        for item in legends_input:
            if isinstance(item, LegendItem):
                parsed_legends.append(item)
            elif isinstance(item, dict):
                try:
                    parsed_legends.append(LegendItem(**item))
                except (ValidationError, TypeError, ValueError):
                    invalid_count += 1
                    logger.debug("Invalid legend item ignored", exc_info=True)
        if invalid_count:
            logger.warning(
                "Ignored %s invalid legend items; %s valid items remain. "
                "Historical results with null fields must be regenerated.",
                invalid_count, len(parsed_legends),
            )
        return parsed_legends

    def identify_clicked_region(self, image_path: str, click_point: tuple,
                                legends_input: List[Union[Dict, LegendItem]]) -> Optional[RegionItem]:
        if not os.path.exists(image_path) or not legends_input: return None

        try:
            legends = self._parse_legend_input(legends_input)
            if not legends:
                logger.warning("Region matching skipped: no valid legends available")
                return None
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
                check_cancelled()
                if len(leg.avg_color) != 3 or not np.all(np.isfinite(leg.avg_color)):
                    logger.debug("Ignoring legend with invalid RGB color: id=%s", leg.id)
                    continue
                leg_rgb = np.array(leg.avg_color, dtype=np.float32)
                dist = np.linalg.norm(click_arr - leg_rgb)
                candidates.append((dist, leg))

            if not candidates:
                logger.warning("Region matching skipped: no legends with valid RGB colors")
                return None
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
                logger.debug("Unable to serialize geological symbol", exc_info=True)
                symbol_dict = {}

            match_conf = round(max(0.0, 100.0 * (1.0 - best_score / 441.67)), 2)

            return RegionItem(
                id=0, contour=real_cnt, centroid=[float(cx), float(cy)],
                area=region_area, matched_legend_id=final_match.id, match_score=match_conf,
                geo=RegionGeoInfo(
                    symbol=symbol_dict,
                    unit_name=_safe_text(final_match.legend_text) or _safe_text(final_match.formation) or "未识别地层",
                    legend_id=final_match.id,
                    match_score=match_conf, legend_color_name=final_match.color_name,
                    legend_color_rgb=final_match.avg_color
                ),
                region_color_rgb=click_color.tolist(),
                top_matches=[RegionMatch(
                    legend_id=final_match.id, match_degree=float(best_score), raw=float(best_score),
                    dE=float(best_score), hist=0.0, legend_rgb=final_match.avg_color, legend_name=final_match.color_name
                )]
            )

        except Exception:
            logger.exception("Interactive map region extraction failed")
            return None

    def _simple_analysis(self, image_path, instruction):
        try:
            b64_str = self._file_to_base64(image_path)
            if not b64_str:
                wrapped = FileValidationError("Image could not be decoded", details=str(image_path))
                return {"analysis": ImageAnalysisResult(
                    summary=public_error_message(wrapped), error_code=error_code(wrapped), legends=[], regions=[]
                )}

            messages = [HumanMessage(content=[{"type": "text", "text": f"分析地质图件：{instruction}"},
                                              {"type": "image", "image": f"data:image/jpeg;base64,{b64_str}"}])]
            resp = invoke_model(self.llm, messages, name="image.recognize", model_name=self.model_name)

            print(f">>> [ImageAnalysisAgent] 处理完成，基础全图直读分析完毕。")
            return {"analysis": ImageAnalysisResult(summary=self._extract_content_str(resp.content), legends=[],
                                                    regions=[])}

        except Exception as exc:
            wrapped = ModelInvocationError("Basic image model invocation failed", details=str(exc))
            logger.exception("Basic image model invocation failed")
            return {"analysis": ImageAnalysisResult(
                summary=public_error_message(wrapped), error_code=error_code(wrapped), legends=[], regions=[]
            )}

    def _build_graph(self):
        workflow = StateGraph(ImageAgentState)
        workflow.add_node("analyze_image", self._analyze_node)
        workflow.add_edge(START, "analyze_image")
        workflow.add_edge("analyze_image", END)
        return workflow.compile()

    def run(self, image_path: str, instruction: str = "请分析这张地质图件", output_dir: str = ""):
        return self.graph.invoke({
            "image_path": image_path,
            "instruction": instruction,
            "output_dir": output_dir
        })
