import os
import time
import hashlib
import logging
import numpy as np
import streamlit as st
import streamlit.components.v1 as components
import html
import inspect
from PIL import Image
from dotenv import load_dotenv
from workspace_manager import create_run_dir, clear_session_workspace, save_uploaded_file
from state_serialization import serialize_result_state
from runtime_config import configure_tracing
from app_errors import AgentExecutionError, ConfigurationError, public_error_message, error_code

logger = logging.getLogger(__name__)

# === 🚀 核心修复：解除 PIL 库对超大分辨率地质图的"像素炸弹"防爆限制 ===
Image.MAX_IMAGE_PIXELS = None

os.environ["FOR_DISABLE_CONSOLE_CTRL_HANDLER"] = "1"

# === Lazy imports for heavy modules (loaded only when needed) ===
_MasterAgent = None
_DBManager = None
_cv2 = None
_pandas = None

def _get_MasterAgent():
    global _MasterAgent
    if _MasterAgent is None:
        from master_agent import MasterAgent as MA
        _MasterAgent = MA
    return _MasterAgent

def _get_DBManager():
    global _DBManager
    if _DBManager is None:
        try:
            from db_manager import DBManager as DB
            _DBManager = DB
        except ImportError:
            logger.exception("MySQL persistence dependency is unavailable")
            raise
    return _DBManager

def _get_cv2():
    global _cv2
    if _cv2 is None:
        import cv2
        _cv2 = cv2
    return _cv2

def _get_pandas():
    global _pandas
    if _pandas is None:
        import pandas as pd
        _pandas = pd
    return _pandas

# === 启动流程 ===
try:
    from streamlit_image_coordinates import streamlit_image_coordinates

    HAS_INTERACTIVE = True
except ImportError:
    HAS_INTERACTIVE = False

current_dir = os.path.dirname(os.path.abspath(__file__))
env_path = os.path.join(current_dir, "api_key.env")
if os.path.exists(env_path):
    load_dotenv(env_path, override=True)

ENV_API_KEY = os.getenv("DASHSCOPE_API_KEY")
configure_tracing()

st.set_page_config(page_title="基于多智能体的三维地质建模钻孔数据生成", page_icon="🌍", layout="wide", initial_sidebar_state="expanded")

try:
    DB = _get_DBManager()
    db = DB()
    if not db.check_connection(): st.error("⚠️ 数据库连接失败。"); st.stop()
except Exception:
    logger.exception("Database initialization failed")
    st.error("数据库初始化错误")
    st.stop()

# DBManager reloads api_key.env; reapply explicit telemetry policy afterward.
configure_tracing()

# === 全局状态初始化 ===
if "master_agent" not in st.session_state: st.session_state.master_agent = None
if "current_session_id" not in st.session_state: st.session_state.current_session_id = None
if "uploader_key" not in st.session_state: st.session_state.uploader_key = 0

# === 核心状态：控制输入框与终止按钮的自动变换 ===
if "is_processing" not in st.session_state: st.session_state.is_processing = False
if "pending_prompt" not in st.session_state: st.session_state.pending_prompt = None

if "app_initialized" not in st.session_state:
    st.session_state.app_initialized = True
    try:
        initial_sessions = db.get_all_sessions()
        if initial_sessions:
            st.session_state.current_session_id = initial_sessions[0]['id']
    except Exception:
        logger.exception("Failed to restore the initial session")

# === UI theme: professional geological workspace ===
st.markdown("""
<style>
    :root {
        --geo-ink: #172b2d;
        --geo-muted: #637779;
        --geo-line: #dce6e3;
        --geo-soft: #f3f7f5;
        --geo-panel: #ffffff;
        --geo-green: #176b5b;
        --geo-green-dark: #105247;
        --geo-sand: #c98b3c;
        --geo-blue: #3e7086;
    }
    .stApp { background: #f7faf8; color: var(--geo-ink); }
    [data-testid="stAppViewContainer"] > .main {
        background:
          radial-gradient(circle at 92% 2%, rgba(23,107,91,.07), transparent 25rem),
          #f7faf8;
    }
    [data-testid="stMainBlockContainer"] {
        max-width: 1180px;
        padding-top: 1.4rem;
        padding-bottom: 8rem;
    }
    #MainMenu, footer { visibility: hidden; }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background: #eef4f1;
        border-right: 1px solid var(--geo-line);
    }
    [data-testid="stSidebarContent"] { padding-top: .65rem; }
    .sidebar-brand {
        display: flex; align-items: center; gap: .8rem;
        padding: .7rem .35rem 1rem;
    }
    .brand-mark {
        display: grid; place-items: center; width: 42px; height: 42px;
        border-radius: 13px; background: var(--geo-green); color: white;
        font-size: 1.35rem; box-shadow: 0 8px 20px rgba(23,107,91,.18);
    }
    .brand-name { color: var(--geo-ink); font-weight: 750; font-size: 1.05rem; }
    .brand-subtitle { color: var(--geo-muted); font-size: .73rem; margin-top: .1rem; }
    .section-label {
        color: #54706d; font-size: .72rem; font-weight: 750;
        letter-spacing: .08em; text-transform: uppercase; margin: .25rem 0 .45rem;
    }
    [data-testid="stSidebar"] hr { margin: .8rem 0; border-color: var(--geo-line); }
    [data-testid="stSidebar"] button {
        border-radius: 10px !important; min-height: 2.45rem;
        border-color: #d5e1dd !important; box-shadow: none !important;
    }
    [data-testid="stSidebar"] button[kind="primary"] {
        background: var(--geo-green) !important; border-color: var(--geo-green) !important;
    }
    [data-testid="stSidebar"] button[kind="secondary"] { justify-content: flex-start !important; }
    [data-testid="stSidebar"] button[kind="secondary"] p {
        font-size: .88rem; margin: 0 !important; text-align: left !important;
    }

    /* Page header and empty state */
    .workspace-hero {
        background: rgba(255,255,255,.92); border: 1px solid var(--geo-line);
        border-radius: 18px; padding: 1.2rem 1.35rem; margin-bottom: 1.1rem;
        box-shadow: 0 8px 28px rgba(31,65,60,.055);
        display: flex; justify-content: space-between; align-items: center; gap: 1rem;
    }
    .workspace-eyebrow { color: var(--geo-green); font-size: .72rem; font-weight: 800; letter-spacing: .09em; }
    .workspace-title { color: var(--geo-ink); font-size: 1.45rem; font-weight: 760; margin: .2rem 0; }
    .workspace-subtitle { color: var(--geo-muted); font-size: .84rem; }
    .hero-meta { display: flex; flex-wrap: wrap; justify-content: flex-end; gap: .45rem; }
    .meta-pill {
        background: var(--geo-soft); border: 1px solid var(--geo-line); color: #42615f;
        padding: .34rem .62rem; border-radius: 999px; font-size: .72rem; white-space: nowrap;
    }
    .empty-panel {
        padding: 2.4rem 1rem 1.35rem; text-align: center;
        color: var(--geo-muted);
    }
    .empty-icon {
        width: 58px; height: 58px; display: grid; place-items: center; margin: 0 auto .85rem;
        background: #e5f0ec; color: var(--geo-green); border-radius: 18px; font-size: 1.65rem;
    }
    .empty-title { color: var(--geo-ink); font-weight: 750; font-size: 1.15rem; margin-bottom: .35rem; }
    .starter-card {
        min-height: 112px; background: white; border: 1px solid var(--geo-line);
        border-radius: 14px; padding: 1rem; color: var(--geo-muted); font-size: .8rem;
        box-shadow: 0 5px 18px rgba(31,65,60,.04);
    }
    .starter-card strong { display:block; color: var(--geo-ink); font-size: .9rem; margin: .35rem 0; }

    /* Chat and result surfaces */
    [data-testid="stChatMessage"] {
        background: white; border: 1px solid var(--geo-line); border-radius: 15px;
        padding: .85rem 1rem; margin-bottom: .7rem;
        box-shadow: 0 4px 16px rgba(31,65,60,.035);
    }
    [data-testid="stChatMessage"] [data-testid="stChatMessageAvatarUser"] {
        background: #e8eeec; color: var(--geo-ink);
    }
    [data-testid="stChatMessage"] [data-testid="stChatMessageAvatarAssistant"] {
        background: #dceee8; color: var(--geo-green-dark);
    }
    [data-testid="stExpander"] {
        background: rgba(255,255,255,.85); border: 1px solid var(--geo-line);
        border-radius: 12px !important; overflow: hidden;
    }
    [data-testid="stMetric"] {
        background: var(--geo-soft); border: 1px solid var(--geo-line);
        border-radius: 12px; padding: .7rem .85rem;
    }
    [data-testid="stDataFrame"] { border: 1px solid var(--geo-line); border-radius: 12px; overflow: hidden; }
    [data-testid="stFileUploaderDropzone"] {
        background: rgba(255,255,255,.65); border: 1px dashed #9ebbb3; border-radius: 12px;
    }
    [data-testid="stStatusWidget"] { border-color: var(--geo-line); border-radius: 12px; }
    .agent-box { padding: 6px 10px; border-radius: 999px; border: 1px solid var(--geo-line); text-align: center; font-size: .75em; margin:2px; display:inline-block; }
    .agent-active { background: #dff1eb; color: var(--geo-green-dark); border-color: #88bbae; font-weight: 700; }
    .agent-inactive { background: var(--geo-soft); color: #8a9998; }
    .geo-card { background: #eef7f4; padding: 15px; border-radius: 12px; border-left: 4px solid var(--geo-green); margin-top: 10px; font-size: .95em; line-height: 1.6; }
    .thought-process { background: #f3f7f5; border-left: 3px solid #78928e; padding: 10px 15px; margin: 10px 0; border-radius: 0 8px 8px 0; font-family: ui-monospace, monospace; font-size: .84em; color: #48615f; }
    .thought-item { margin-bottom: 5px; }
    .legend-row { display: flex; align-items: center; margin-bottom: 5px; border-bottom: 1px solid var(--geo-line); padding: 5px 0; }
    .legend-img { width: 40px; height: 30px; object-fit: cover; border-radius: 6px; margin-right: 10px; border:1px solid var(--geo-line); }

    /* Composer */
    [data-testid="stBottom"] > div { background: linear-gradient(transparent, #f7faf8 26%) !important; padding-bottom: .75rem !important; }
    [data-testid="stChatInput"] {
        border: 1px solid #bfd0cb !important; border-radius: 15px !important;
        background: white !important; box-shadow: 0 10px 30px rgba(31,65,60,.11) !important;
    }
    [data-testid="stChatInput"]:focus-within { border-color: var(--geo-green) !important; }
    @media (max-width: 760px) {
        [data-testid="stMainBlockContainer"] { padding: .75rem .7rem 7rem; }
        .workspace-hero { align-items: flex-start; flex-direction: column; padding: 1rem; }
        .hero-meta { justify-content: flex-start; }
        .workspace-title { font-size: 1.2rem; }
    }
</style>
""", unsafe_allow_html=True)


def safe_get(obj, key):
    if isinstance(obj, dict): return obj.get(key)
    return getattr(obj, key, None)


def build_langchain_history(history, max_messages=20):
    """Convert persisted MySQL messages into bounded LangChain chat context."""
    from langchain_core.messages import AIMessage, HumanMessage

    converted = []
    for item in (history or [])[-max_messages:]:
        content = str(item.get("content") or "").strip()
        if not content:
            continue
        if item.get("role") == "user":
            converted.append(HumanMessage(content=content))
        elif item.get("role") == "assistant":
            converted.append(AIMessage(content=content))
    return converted


def highlight_contour_on_image(image_path, contour, color=(0, 0, 255), thickness=5):
    if not os.path.exists(image_path): return None
    cv2 = _get_cv2()
    img_bgr = cv2.imdecode(np.fromfile(image_path, dtype=np.uint8), cv2.IMREAD_COLOR)
    if img_bgr is None: return None
    pts = np.array(contour, dtype=np.int32)
    overlay = img_bgr.copy()
    cv2.fillPoly(overlay, [pts], (0, 200, 0))
    img_bgr = cv2.addWeighted(overlay, 0.4, img_bgr, 0.6, 0)
    cv2.polylines(img_bgr, [pts], True, color, thickness)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    return Image.fromarray(img_rgb)


@st.dialog("🌍 地质图交互查看器", width="large")
def show_geo_viewer_modal(image_path, legends, unique_key):
    st.info("💡 提示：点击图片任意区域，AI 将实时分割该区域并识别对应的地质含义。")
    col1, col2 = st.columns([2, 1])

    region_state_key = f"last_region_{unique_key}"
    session_legends_key = f"legends_cache_{unique_key}"

    if region_state_key not in st.session_state: st.session_state[region_state_key] = None

    with col1:
        DISPLAY_WIDTH = 700
        try:
            original_img = Image.open(image_path)
            orig_w, orig_h = original_img.size
            scale_factor = orig_w / DISPLAY_WIDTH
        except Exception:
            logger.exception("Failed to load image in interactive viewer: %s", image_path)
            st.error("无法加载图片")
            return

        coords = streamlit_image_coordinates(image_path, width=DISPLAY_WIDTH, key=f"modal_coords_{unique_key}")

        if coords:
            last_coords_key = f"last_coords_{unique_key}"
            current_coords_str = f"{coords['x']},{coords['y']}"
            if st.session_state.get(last_coords_key) != current_coords_str:
                st.session_state[last_coords_key] = current_coords_str
                click_x = coords['x'] * scale_factor
                click_y = coords['y'] * scale_factor

                if st.session_state.master_agent:
                    with st.spinner("🧠 正在进行机器视觉拓扑分割与多模态识别..."):
                        region = st.session_state.master_agent.image_bot.identify_clicked_region(
                            image_path, (click_x, click_y), st.session_state[session_legends_key]
                        )
                        st.session_state[region_state_key] = region

                        if region and hasattr(region, 'matched_legend_id'):
                            for leg in st.session_state[session_legends_key]:
                                leg_id = leg.get('id') if isinstance(leg, dict) else getattr(leg, 'id', None)
                                if leg_id == region.matched_legend_id:
                                    geo_data = region.geo if hasattr(region, 'geo') else region.get('geo', {})
                                    unit_name = getattr(geo_data, 'unit_name', '') if not isinstance(geo_data,
                                                                                                     dict) else geo_data.get(
                                        'unit_name', '')
                                    sym_data = getattr(geo_data, 'symbol', {}) if not isinstance(geo_data,
                                                                                                 dict) else geo_data.get(
                                        'symbol', {})

                                    if isinstance(leg, dict):
                                        leg['is_recognized'] = True
                                        leg['legend_text'] = unit_name
                                        if isinstance(sym_data, dict):
                                            leg['symbol'] = sym_data
                                        else:
                                            leg['symbol'] = sym_data.model_dump() if hasattr(sym_data,
                                                                                             'model_dump') else sym_data.dict()
                                    else:
                                        leg.is_recognized = True
                                        leg.legend_text = unit_name
                                        leg.symbol = sym_data
                                    break

        selected_region = st.session_state.get(region_state_key)
        if selected_region:
            cnt = selected_region.contour if hasattr(selected_region, 'contour') else selected_region.get('contour')
            hl_img = highlight_contour_on_image(image_path, cnt)
            if hl_img: st.image(hl_img, caption="✅ 拓扑分割结果 (高亮区域)", width=DISPLAY_WIDTH)

    with col2:
        selected_region = st.session_state.get(region_state_key)
        if selected_region:
            if hasattr(selected_region, 'geo'):
                geo = selected_region.geo.model_dump() if hasattr(selected_region.geo,
                                                                  'model_dump') else selected_region.geo.dict()
            elif isinstance(selected_region, dict):
                geo = selected_region.get('geo', {})
            else:
                geo = {}

            sym_obj = geo.get('symbol', {})
            if isinstance(sym_obj, dict):
                sym_str = sym_obj.get('symbol_terminal') or sym_obj.get('terminal') or sym_obj.get(
                    'symbol_final') or sym_obj.get('final') or sym_obj.get('base') or 'N/A'
            else:
                sym_str = getattr(sym_obj, 'terminal', None) or getattr(sym_obj, 'final', None) or getattr(sym_obj,
                                                                                                           'base',
                                                                                                           None) or 'N/A'

            st.markdown(f"""
            <div class="geo-card">
                <h3 style="margin-top:0">📍 区域详情</h3>
                <p><b>地质单元:</b><br>{geo.get('unit_name', '解析失败/待解析')}</p>
                <p><b>地质符号:</b> <span style="color:red; font-size:1.2em; font-weight:bold">{sym_str}</span></p>
                <hr>
                <p style="font-size:0.9em; color:#666;">
                   <b>匹配图例ID:</b> {geo.get('legend_id')}<br>
                   <b>置信度:</b> {geo.get('match_score', 'N/A')}%<br>
                   <b>主色调:</b> {geo.get('legend_color_rgb')}
                </p>
            </div>
            """, unsafe_allow_html=True)

            with st.expander("📄 查看底层提取 JSON 结构", expanded=False):
                try:
                    if isinstance(selected_region, dict):
                        display_json = {k: v for k, v in selected_region.items() if k != 'contour'}
                    else:
                        display_json = selected_region.model_dump() if hasattr(selected_region, 'model_dump') else (
                            selected_region.dict() if hasattr(selected_region, 'dict') else vars(selected_region))
                        if 'contour' in display_json:
                            del display_json['contour']

                    st.json(display_json)
                except Exception:
                    logger.exception("Unable to render selected region as JSON")
                    st.warning("JSON 数据暂时无法显示，请查看服务日志。（错误码：SERIALIZATION_ERROR）")

        else:
            st.info("👈 请在左侧点击想要解析的地质区块...")

        current_legends = st.session_state.get(session_legends_key, [])
        with st.expander(f"🧩 已识别特征库 ({len(current_legends)})", expanded=True):
            if current_legends:
                html_str = ""
                sorted_legends = sorted(current_legends, key=lambda x: (
                    not (x.get('is_recognized') if isinstance(x, dict) else getattr(x, 'is_recognized', False)),
                    (x.get('id') if isinstance(x, dict) else getattr(x, 'id', 0))))
                for leg in sorted_legends[:20]:
                    if isinstance(leg, dict):
                        is_rec = leg.get('is_recognized', False)
                        b64 = leg.get('color_patch_base64')
                        txt = leg.get('legend_text', '待解析')

                        sym_dict = leg.get('symbol', {})
                        if isinstance(sym_dict, dict):
                            sym = sym_dict.get('symbol_final', '') or sym_dict.get('final', '')
                        else:
                            sym = getattr(sym_dict, 'final', '') or getattr(sym_dict, 'symbol_final', '')
                    else:
                        is_rec = getattr(leg, 'is_recognized', False)
                        b64 = getattr(leg, 'color_patch_base64', None)
                        txt = getattr(leg, 'legend_text', '待解析')
                        sym_obj = getattr(leg, 'symbol', None)
                        sym = getattr(sym_obj, 'final', '') if sym_obj else ''

                    status_icon = "✅" if is_rec else "⏳"
                    img_src = f"data:image/jpeg;base64,{b64}" if b64 else ""
                    txt_str = str(txt) if txt else "待解析"
                    html_str += f"""
                    <div class="legend-row">
                        <img src="{img_src}" class="legend-img"/>
                        <div style="flex:1">
                            <div style="font-size:0.8em; font-weight:bold">{status_icon} {txt_str[:12]}...</div>
                            <div style="font-size:0.7em; color:red">{sym}</div>
                        </div>
                    </div>
                    """
                st.markdown(html_str, unsafe_allow_html=True)


def generate_kg_html(text, nodes, edges):
    safe_text = html.escape(text)
    vis_nodes = []
    node_ids = set()
    for n in nodes:
        n_dict = n if isinstance(n, dict) else (n.model_dump() if hasattr(n, 'model_dump') else n.dict())
        node_id = n_dict.get('id', '')
        if not node_id: continue
        node_ids.add(node_id)
        label = f"{node_id}\\n({n_dict.get('type', 'Unknown')})"
        vis_nodes.append({"id": node_id, "label": label, "title": n_dict.get('properties', '')})

    vis_edges = []
    for e in edges:
        e_dict = e if isinstance(e, dict) else (e.model_dump() if hasattr(e, 'model_dump') else e.dict())
        vis_edges.append({
            "from": e_dict.get('source'),
            "to": e_dict.get('target'),
            "label": e_dict.get('relation'),
            "title": e_dict.get('description', '')
        })

    sorted_nodes = sorted(list(node_ids), key=len, reverse=True)
    for node_id in sorted_nodes:
        safe_node_id = html.escape(node_id)
        if safe_node_id in safe_text:
            span_html = f'<span id="entity-{node_id}" data-id="{node_id}" class="entity-span">{safe_node_id}</span>'
            safe_text = safe_text.replace(safe_node_id, span_html)

    safe_text = safe_text.replace("\\n", "<br>").replace("\\r", "").replace("\n", "<br>")

    template = """
    <!DOCTYPE html>
    <html>
    <head>
      <meta charset="utf-8">
      <script type="text/javascript" src="https://unpkg.com/vis-network/standalone/umd/vis-network.min.js"></script>
      <style>
        body, html { margin: 0; padding: 0; height: 100%; font-family: 'Microsoft YaHei', sans-serif; overflow: hidden; background: #fff;}
        #container { display: flex; width: 100%; height: 100%; position: relative; }
        #text-pane { width: 45%; height: 100%; overflow-y: auto; padding: 25px; box-sizing: border-box; border-right: 2px solid #e0e0e0; font-size: 14px; line-height: 1.9; color: #444; background: #fafafa; }
        #graph-pane { width: 55%; height: 100%; box-sizing: border-box; background: #ffffff; }
        .entity-span { background-color: #dcedc8; border-radius: 4px; padding: 2px 4px; cursor: pointer; border: 1px solid #c5e1a5; font-weight: 600; color: #33691e; transition: all 0.3s; }
        .entity-span:hover { background-color: #aed581; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
        .entity-active { background-color: #ffcc80 !important; border-color: #ff9800 !important; color: #e65100 !important; box-shadow: 0 2px 8px rgba(255,152,0,0.4); }
        #svg-overlay { position: absolute; top: 0; left: 0; width: 100%; height: 100%; pointer-events: none; z-index: 100; }
        path.connection-line { fill: none; stroke: #ff9800; stroke-width: 3; stroke-dasharray: 6,6; animation: dash 1s linear infinite; filter: drop-shadow(0 2px 2px rgba(255,152,0,0.3)); }
        @keyframes dash { to { stroke-dashoffset: -12; } }
        #text-pane::-webkit-scrollbar { width: 6px; }
        #text-pane::-webkit-scrollbar-track { background: transparent; }
        #text-pane::-webkit-scrollbar-thumb { background: #ccc; border-radius: 3px; }
      </style>
    </head>
    <body>
      <div id="container">
        <div id="text-pane">{{TEXT_CONTENT}}</div>
        <div id="graph-pane"></div>
        <svg id="svg-overlay">
          <path id="active-line" class="connection-line" d=""></path>
        </svg>
      </div>
      <script>
        const nodesData = {{NODES_JSON}};
        const edgesData = {{EDGES_JSON}};
        const container = document.getElementById('graph-pane');
        const data = { nodes: new vis.DataSet(nodesData), edges: new vis.DataSet(edgesData) };
        const options = {
            physics: { enabled: true, solver: 'forceAtlas2Based', stabilization: { iterations: 150 } },
            nodes: { shape: 'box', margin: 12, color: { background: '#e3f2fd', border: '#2196f3', highlight: { background: '#ffe0b2', border: '#ff9800' } }, font: { size: 14, multi: true } },
            edges: { arrows: 'to', color: '#bbdefb', smooth: { type: 'cubicBezier' } },
            interaction: { hover: true, zoomView: true }
        };
        const network = new vis.Network(container, data, options);
        const activeLine = document.getElementById('active-line');
        const textPane = document.getElementById('text-pane');
        let selectedNodeId = null;

        function updateLine() {
            if (!selectedNodeId) { activeLine.setAttribute('d', ''); return; }
            const span = document.getElementById('entity-' + selectedNodeId);
            if (!span) { activeLine.setAttribute('d', ''); return; }

            const spanRect = span.getBoundingClientRect();
            const textPaneRect = textPane.getBoundingClientRect();
            const startX = spanRect.right;
            const startY = spanRect.top + spanRect.height / 2;

            if(startY < textPaneRect.top || startY > textPaneRect.bottom) {
                 activeLine.setAttribute('d', ''); return;
            }

            const nodePos = network.getPosition(selectedNodeId);
            const domPos = network.canvasToDOM(nodePos);
            const graphRect = container.getBoundingClientRect();
            const endX = graphRect.left + domPos.x;
            const endY = graphRect.top + domPos.y;

            const cp1X = startX + (endX - startX) * 0.4;
            const cp1Y = startY;
            const cp2X = endX - (endX - startX) * 0.4;
            const cp2Y = endY;

            const d = `M ${startX} ${startY} C ${cp1X} ${cp1Y}, ${cp2X} ${cp2Y}, ${endX} ${endY}`;
            activeLine.setAttribute('d', d);
        }

        document.querySelectorAll('.entity-span').forEach(span => {
            span.addEventListener('click', function() {
                const nodeId = this.getAttribute('data-id');
                selectedNodeId = nodeId;
                network.selectNodes([nodeId]);
                network.focus(nodeId, { scale: 1.2, animation: true });
                document.querySelectorAll('.entity-span').forEach(s => s.classList.remove('entity-active'));
                this.classList.add('entity-active');
                updateLine();
            });
        });

        network.on('click', function(properties) {
            if (properties.nodes.length > 0) {
                const nodeId = properties.nodes[0];
                selectedNodeId = nodeId;
                const span = document.getElementById('entity-' + nodeId);
                document.querySelectorAll('.entity-span').forEach(s => s.classList.remove('entity-active'));
                if (span) {
                    span.scrollIntoView({ behavior: 'smooth', block: 'center' });
                    span.classList.add('entity-active');
                }
                updateLine();
            } else {
                selectedNodeId = null;
                updateLine();
                document.querySelectorAll('.entity-span').forEach(s => s.classList.remove('entity-active'));
            }
        });

        network.on('dragging', updateLine);
        network.on('zoom', updateLine);
        network.on('animationIteration', updateLine);
        network.on('afterDrawing', updateLine);
        textPane.addEventListener('scroll', updateLine);
        window.addEventListener('resize', updateLine);
      </script>
    </body>
    </html>
    """
    import json
    return template.replace("{{TEXT_CONTENT}}", safe_text).replace("{{NODES_JSON}}", json.dumps(vis_nodes)).replace(
        "{{EDGES_JSON}}", json.dumps(vis_edges))


def render_analysis_result(decision, result_state, image_path=None, msg_id=None):
    thought_log = (safe_get(result_state, 'turn_thought_log') or
                   safe_get(result_state, 'thought_log'))
    if thought_log:
        with st.expander("🛠️ 智能体思考与调用路径", expanded=False):
            st.markdown('<div class="thought-process">', unsafe_allow_html=True)
            for log in thought_log:
                st.markdown(f'<div class="thought-item">{log}</div>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)

    st.caption("🔍 智能体调度路径")
    c1, c2, c3 = st.columns(3)

    def style(t):
        return "agent-active" if decision == t else "agent-inactive"

    with c1:
        st.markdown(f'<span class="agent-box {style("extract_text_info")}">📄 文本信息</span>', unsafe_allow_html=True)
    with c2:
        st.markdown(f'<span class="agent-box {style("analyze_image")}">🖼️ 图件</span>', unsafe_allow_html=True)
    with c3:
        st.markdown(f'<span class="agent-box {style("generate_boreholes")}">🕳️ 虚拟钻孔</span>', unsafe_allow_html=True)

    final_text_info = safe_get(result_state, 'final_text_info')
    if decision == "extract_text_info" and final_text_info:
        strata = final_text_info.get('strata', []) if isinstance(final_text_info, dict) else getattr(final_text_info,
                                                                                                     'strata', [])
        profiles = final_text_info.get('profiles', []) if isinstance(final_text_info, dict) else getattr(
            final_text_info, 'profiles', [])
        source_text = final_text_info.get('source_text', '') if isinstance(final_text_info, dict) else getattr(
            final_text_info, 'source_text', '')

        if source_text:
            nodes_dict = {}
            edges_list = []

            for s in strata:
                s_dict = s if isinstance(s, dict) else (s.model_dump() if hasattr(s, 'model_dump') else s.dict())
                f_name = s_dict.get('formation')
                age_name = s_dict.get('formation_age_1')
                if f_name:
                    nodes_dict[f_name] = {"id": f_name, "type": "地层", "properties": s_dict.get('rock_features') or ''}
                if age_name:
                    nodes_dict[age_name] = {"id": age_name, "type": "年代",
                                            "properties": s_dict.get('formation_age_code_1') or ''}
                if f_name and age_name:
                    edges_list.append(
                        {"source": f_name, "target": age_name, "relation": "属于地质年代", "description": ""})

            for p in profiles:
                p_dict = p if isinstance(p, dict) else (p.model_dump() if hasattr(p, 'model_dump') else p.dict())
                p_name = p_dict.get('name')
                p_form = p_dict.get('formation')
                if p_name:
                    nodes_dict[p_name] = {"id": p_name, "type": "地层剖面",
                                          "properties": p_dict.get('rock_combination') or ''}
                if p_name and p_form:
                    if p_form not in nodes_dict:
                        nodes_dict[p_form] = {"id": p_form, "type": "地层", "properties": ""}
                    edges_list.append({"source": p_name, "target": p_form, "relation": "包含地层", "description": ""})

            nodes_list = list(nodes_dict.values())

            if nodes_list:
                st.markdown("##### 🕸️ 文本实体抽取与原文溯源交互面板")
                st.caption("💡 提示：点击左侧文本的高亮地质实体，或右侧的图谱节点，可动态追踪知识抽取的来源对应关系！")
                html_content = generate_kg_html(source_text, nodes_list, edges_list)
                components.html(html_content, height=600, scrolling=False)

        if strata:
            st.markdown("##### 🪨 岩石地层信息表")
            st.dataframe(
                [s if isinstance(s, dict) else (s.model_dump() if hasattr(s, 'model_dump') else s.dict()) for s in
                 strata], hide_index=True)
        if profiles:
            st.markdown("##### ⛰️ 剖面信息表")
            st.dataframe(
                [p if isinstance(p, dict) else (p.model_dump() if hasattr(p, 'model_dump') else p.dict()) for p in
                 profiles], hide_index=True)

    final_image_analysis = safe_get(result_state, 'final_image_analysis')
    if decision == "analyze_image" and final_image_analysis:
        summary = final_image_analysis.get('summary', '') if isinstance(final_image_analysis, dict) else getattr(
            final_image_analysis, 'summary', '')
        st.info(summary)

        annotated_map_base64 = final_image_analysis.get('annotated_map_base64') if isinstance(final_image_analysis,
                                                                                              dict) else getattr(
            final_image_analysis, 'annotated_map_base64', None)
        cropped_legend_base64 = final_image_analysis.get('cropped_legend_base64') if isinstance(final_image_analysis,
                                                                                                dict) else getattr(
            final_image_analysis, 'cropped_legend_base64', None)
        artifacts_dir = final_image_analysis.get('artifacts_dir') if isinstance(final_image_analysis, dict) else getattr(
            final_image_analysis, 'artifacts_dir', None)
        annotated_map_path = os.path.join(artifacts_dir, "legend_items_detected.jpg") if artifacts_dir else ""
        cropped_legend_path = os.path.join(artifacts_dir, "cropped_legend.jpg") if artifacts_dir else ""

        if (annotated_map_base64 or cropped_legend_base64 or
                os.path.exists(annotated_map_path) or os.path.exists(cropped_legend_path)):
            st.markdown("##### 🎯 智能图例区域定位")
            cols = st.columns(2)
            if annotated_map_base64 or os.path.exists(annotated_map_path):
                with cols[0]:
                    annotated_source = (f"data:image/jpeg;base64,{annotated_map_base64}"
                                        if annotated_map_base64 else annotated_map_path)
                    st.image(annotated_source, caption="🗺️ 自动圈定的图例分布区",
                             width="stretch")
            if cropped_legend_base64 or os.path.exists(cropped_legend_path):
                with cols[1]:
                    cropped_source = (f"data:image/jpeg;base64,{cropped_legend_base64}"
                                      if cropped_legend_base64 else cropped_legend_path)
                    st.image(cropped_source, caption="✂️ 精确截取的图例内容",
                             width="stretch")

        legends = final_image_analysis.get('legends', []) if isinstance(final_image_analysis, dict) else getattr(
            final_image_analysis, 'legends', [])
        if legends:
            with st.expander(f"🧩 提取到的图例基准特征库 ({len(legends)} 个)"):
                display_legends = []
                for leg in legends:
                    leg_dict = leg if isinstance(leg, dict) else (
                        leg.model_dump() if hasattr(leg, 'model_dump') else leg.dict())
                    display_legends.append({
                        "ID": leg_dict.get("id"),
                        "主色调(RGB)": str(leg_dict.get("avg_color")),
                        "地层属性": leg_dict.get("legend_text", "待交互时智能解析...")
                    })
                st.dataframe(display_legends, hide_index=True)

            if image_path and os.path.exists(image_path) and HAS_INTERACTIVE:
                btn_key = f"btn_view_{msg_id}" if msg_id else f"btn_view_temp_{hashlib.md5(image_path.encode()).hexdigest()}"
                if st.button("🔍 打开交互式图解窗口", key=btn_key, width="stretch", type="primary"):
                    session_legends_key = f"legends_cache_{btn_key}"
                    if session_legends_key not in st.session_state:
                        st.session_state[session_legends_key] = legends
                    show_geo_viewer_modal(image_path, st.session_state[session_legends_key], unique_key=btn_key)
            elif not HAS_INTERACTIVE:
                st.warning("缺少交互组件。请在终端执行 `pip install streamlit-image-coordinates` 后重启。")

    final_boreholes = safe_get(result_state, 'final_boreholes')
    if decision == "generate_boreholes" and final_boreholes:
        status = final_boreholes.get("status", "")
        bh_data = final_boreholes.get("virtual_boreholes_data", [])
        csv_path = final_boreholes.get("csv_file_path", "")
        total_rows = final_boreholes.get("total_rows", len(bh_data))

        if status == "Success" and bh_data:
            pd = _get_pandas()
            df_show = pd.DataFrame(bh_data)
            st.success(
                f"✅ 钻孔采样完毕，总计生成 {total_rows} 行标准格式数据！*(下方表格仅展示前 {len(df_show)} 行预览)*")

            try:
                import plotly.express as px
                st.markdown("##### 🌍 三维虚拟钻孔空间分布阵列")
                df_plot = df_show.copy()
                for col in ['x', 'y', 'z']:
                    df_plot[col] = pd.to_numeric(df_plot[col], errors='coerce')
                df_plot = df_plot.dropna(subset=['x', 'y', 'z'])

                if len(df_plot) > 5000:
                    st.warning(
                        f"⚠️ 钻孔数据量过大 ({len(df_plot)} 行)，为保证三维渲染流畅，已自动降采样至 5000 个预览点进行可视化。完整数据不受影响，请下载 CSV 查看。")
                    df_plot = df_plot.sample(n=5000, random_state=42)

                if 'formation_code' in df_plot.columns:
                    df_plot['formation_code_str'] = df_plot['formation_code'].astype(str)
                    color_col = 'formation_code_str'
                else:
                    color_col = None

                fig = px.scatter_3d(
                    df_plot, x='x', y='y', z='z',
                    color=color_col,
                    hover_name='borehole_id' if 'borehole_id' in df_plot.columns else None,
                    hover_data={'x': False, 'y': False, 'z': True, 'formation_code': True, 'formation_code_str': False},
                    labels={'formation_code_str': '地层编号'}
                )
                fig.update_traces(marker=dict(size=4, opacity=0.85, line=dict(width=0)))
                fig.update_layout(
                    margin=dict(l=0, r=0, b=0, t=10),
                    scene=dict(
                        xaxis_title='经度/X (m)',
                        yaxis_title='纬度/Y (m)',
                        zaxis_title='深度/Z (m)',
                        aspectmode='auto'
                    ),
                    legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01)
                )
                st.plotly_chart(fig, width="stretch", key=f"plotly_3d_{msg_id}")
            except ImportError:
                st.info("💡 提示：在终端执行 `pip install plotly` 后重启，即可在此处解锁酷炫的 3D 交互式钻孔可视化预览！")

            st.dataframe(df_show, hide_index=True)
            dl_key = f"dl_btn_{msg_id}" if msg_id else f"dl_btn_temp_{hashlib.md5(str(bh_data)[:100].encode()).hexdigest()}"

            dl_c1, dl_c2 = st.columns([1, 2])
            with dl_c1:
                if csv_path and os.path.exists(csv_path):
                    with open(csv_path, "rb") as f:
                        st.download_button("📥 一键下载完整标准格式 CSV 钻孔表", data=f,
                                           file_name="virtual_boreholes_points.csv", mime="text/csv", type="primary",
                                           key=f"{dl_key}_full", width="stretch")
                else:
                    csv_str = df_show.to_csv(index=False, encoding='utf-8-sig')
                    st.download_button("📥 一键下载预览版 CSV 钻孔表", data=csv_str,
                                       file_name="virtual_boreholes_preview.csv", mime="text/csv", type="primary",
                                       key=f"{dl_key}_prev", width="stretch")
        else:
            st.error(f"❌ 数据生成失败，原因:\n\n{status}")


# ==========================================
# 扁平化、现代化的侧边栏配置与上传区
# ==========================================
try:
    db.delete_empty_sessions(exclude_session_id=st.session_state.current_session_id)
except Exception:
    logger.exception("Failed to clean empty sessions")

_all_sessions = db.get_all_sessions()
_valid_ids = [s['id'] for s in _all_sessions]
if st.session_state.current_session_id is not None and st.session_state.current_session_id not in _valid_ids:
    st.session_state.current_session_id = _valid_ids[0] if _valid_ids else None

if st.session_state.current_session_id is None:
    new_session_id = db.create_session(title="新对话")
    if new_session_id is None or new_session_id < 0:
        st.error("无法创建会话工作区，请检查数据库连接。")
        st.stop()
    st.session_state.current_session_id = new_session_id
    _all_sessions = db.get_all_sessions()

cur_sid = st.session_state.current_session_id
msgs = db.get_history(cur_sid) if cur_sid is not None else []
display_sessions = db.get_all_sessions()

with st.sidebar:
    st.markdown("""
    <div class="sidebar-brand">
      <div class="brand-mark">◫</div>
      <div><div class="brand-name">GeoAgent Studio</div>
      <div class="brand-subtitle">多智能体地质建模工作台</div></div>
    </div>
    """, unsafe_allow_html=True)

    if st.button("＋ 新建分析会话", type="primary", width="stretch"):
        new_session_id = db.create_session(title="新对话")
        if new_session_id is not None and new_session_id >= 0:
            st.session_state.current_session_id = new_session_id
            st.session_state.uploader_key += 1
            st.rerun()
        st.error("新建会话失败，请检查数据库连接。")

    if st.button(
        "↻ 清空并重新开始",
        type="secondary",
        use_container_width=True,
        help="删除当前会话全部历史、运行状态和上传缓存，此操作不可撤销。"
    ):
        if cur_sid is not None:
            checkpoint_cleared = True
            if st.session_state.master_agent:
                try:
                    st.session_state.master_agent.reset_thread(cur_sid)
                except Exception:
                    checkpoint_cleared = False
                    logger.exception("Failed to reset checkpoint for session %s", cur_sid)
            cleared = (
                db.clear_session_history(cur_sid, reset_title=True)
                if checkpoint_cleared else False
            )
            if cleared and checkpoint_cleared:
                try:
                    clear_session_workspace(cur_sid)
                except Exception:
                    logger.exception("Failed to clear workspace for session %s", cur_sid)
                for state_key in list(st.session_state.keys()):
                    if state_key.startswith(("last_region_", "legends_cache_", "last_coords_")):
                        del st.session_state[state_key]
                st.session_state.uploader_key += 1
                st.session_state.is_processing = False
                st.session_state.pending_prompt = None
                st.toast("当前会话已清空，可以重新开始。", icon="✅")
                st.rerun()
            else:
                st.error("会话未能完全重置，请检查数据库连接或运行日志。")

    st.divider()
    st.markdown('<div class="section-label">AI ENGINE</div>', unsafe_allow_html=True)
    model_options = ["qwen-max", "qwen-plus", "deepseek-r1"]
    selected_model = st.selectbox(
        "任务执行模型",
        model_options,
        index=0,
        help="注意：主控路由节点将固定使用qwen-max保证调度稳定性，此处选择的模型将用于承担最繁重的文本提取与逻辑推理任务。"
    )

    if "current_model" not in st.session_state:
        st.session_state.current_model = selected_model

    if ENV_API_KEY:
        if st.session_state.master_agent is None or st.session_state.current_model != selected_model:
            st.session_state.current_model = selected_model
            try:
                MA = _get_MasterAgent()
                st.session_state.master_agent = MA(api_key=ENV_API_KEY, model_name=selected_model)
                st.toast(f"计算引擎已成功切换至: {selected_model}", icon="🔄")
            except Exception as exc:
                wrapped = ConfigurationError("Agent engine initialization failed", details=str(exc))
                logger.exception("Agent engine initialization failed: model=%s", selected_model)
                st.error(public_error_message(wrapped))
    st.divider()

    st.markdown('<div class="section-label">RECENT SESSIONS</div>', unsafe_allow_html=True)
    if not display_sessions:
        st.caption("暂无历史会话")
    for s in display_sessions:
        col1, col2 = st.columns([8.5, 1.5])
        with col1:
            is_active = (cur_sid == s['id'])
            prefix = "● " if is_active else "○ "
            if st.button(f"{prefix}{s['title']}", key=f"s_{s['id']}", width="stretch"):
                st.session_state.current_session_id = s['id']
                st.rerun()
        with col2:
            if st.button("🗑️", key=f"del_{s['id']}", help="删除此会话", width="stretch"):
                if st.session_state.master_agent:
                    st.session_state.master_agent.reset_thread(s['id'])
                db.delete_session(s['id'])
                try:
                    clear_session_workspace(s['id'])
                except Exception:
                    logger.exception("Failed to delete workspace for session %s", s['id'])
                if cur_sid == s['id']: st.session_state.current_session_id = None
                st.rerun()
    st.divider()

    st.markdown('<div class="section-label">DATA WORKSPACE</div>', unsafe_allow_html=True)
    uploaded_files = st.file_uploader(
        "拖入地质报告、图件或空间数据", type=["txt", "jpg", "png", "csv", "tif", "tiff", "shp", "shx", "dbf", "prj", "cpg"],
        accept_multiple_files=True, label_visibility="collapsed", key=f"file_uploader_{st.session_state.uploader_key}"
    )
    st.caption("TXT · JPG/PNG · CSV · SHP · GeoTIFF")
    file_paths = []
    image_paths = []
    file_context = ""

    if uploaded_files:
        for uploaded in uploaded_files:
            fname = uploaded.name
            ext = os.path.splitext(fname)[1].lower()
            save_path = save_uploaded_file(
                uploaded,
                session_id=cur_sid,
                batch_id=st.session_state.uploader_key
            )

            if fname.lower().endswith(('.jpg', '.png')):
                image_paths.append(save_path)
                st.image(save_path, width="stretch")
            elif fname.lower().endswith(('.shp', '.tif', '.tiff')):
                file_paths.append(save_path)
                st.success(f"🗺️ 空间数据就绪: {fname}")
            elif fname.lower().endswith(('.shx', '.dbf', '.prj', '.cpg')):
                st.caption(f"🔧 SHP辅助文件: {fname} (已归档)")
            else:
                file_paths.append(save_path)
                st.success(f"📄 数据就绪: {fname}")
                if fname.lower().endswith('.txt'):
                    try:
                        content = uploaded.getvalue().decode("utf-8")
                    except UnicodeDecodeError:
                        try:
                            content = uploaded.getvalue().decode("gbk")
                        except UnicodeDecodeError:
                            logger.warning("Text file is neither UTF-8 nor GBK: %s", fname)
                            content = uploaded.getvalue().decode("utf-8", errors="ignore")
                    file_context += f"\n--- {fname} ---\n{content}\n"

    image_path = image_paths[0] if image_paths else ""
    file_path = file_paths[0] if file_paths else ""

# ==========================================
# 主界面聊天与逻辑
# ==========================================
current_title = "智能分析终端"
if cur_sid is None:
    current_title = "新建地质分析"
else:
    for s in display_sessions:
        if s['id'] == cur_sid:
            current_title = s['title']
            break

safe_title = html.escape(str(current_title))
safe_model = html.escape(str(selected_model))
file_count = len(image_paths) + len(file_paths)
st.markdown(f"""
<section class="workspace-hero">
  <div>
    <div class="workspace-eyebrow">GEOLOGICAL INTELLIGENCE WORKSPACE</div>
    <div class="workspace-title">{safe_title}</div>
    <div class="workspace-subtitle">报告解析、图件识别、知识融合与虚拟钻孔生成</div>
  </div>
  <div class="hero-meta">
    <span class="meta-pill">模型 · {safe_model}</span>
    <span class="meta-pill">文件 · {file_count}</span>
    <span class="meta-pill">有状态会话 · 已启用</span>
  </div>
</section>
""", unsafe_allow_html=True)

if not msgs:
    st.markdown("""
    <div class="empty-panel">
      <div class="empty-icon">⌁</div>
      <div class="empty-title">开始一次地质智能分析</div>
      <div>在左侧添加数据，然后在底部输入任务；也可以直接进行地质知识问答。</div>
    </div>
    """, unsafe_allow_html=True)
    starter_cols = st.columns(3)
    starter_cards = [
        ("▤", "解析地质报告", "抽取地层、年代、岩性、厚度与叠覆关系"),
        ("◈", "识别地质图件", "检测图例、分割区域并匹配地层符号"),
        ("⌖", "生成虚拟钻孔", "融合报告、图件、SHP 与 DEM 生成三维点云"),
    ]
    for col, (icon_text, title_text, desc_text) in zip(starter_cols, starter_cards):
        with col:
            st.markdown(
                f'<div class="starter-card"><span>{icon_text}</span><strong>{title_text}</strong>{desc_text}</div>',
                unsafe_allow_html=True
            )

for msg in msgs:
    avatar_icon = "👤" if msg["role"] == "user" else "✨"

    with st.chat_message(msg["role"], avatar=avatar_icon):
        f_path = msg.get("file_path")
        if f_path:
            for p in f_path.split("|"):
                if os.path.exists(p):
                    if p.lower().endswith(('.jpg', '.png')):
                        st.image(p, width=300)
                    else:
                        st.caption(f"📄 {os.path.basename(p)}")
        st.markdown(msg["content"])
        if msg.get("result_state") and msg.get("decision"):
            with st.expander("📊 结果数据面板",
                             expanded=(msg["decision"] == "generate_boreholes" or msg["decision"] == "analyze_image")):
                img_p = None
                if f_path and msg["decision"] == "analyze_image":
                    for p in f_path.split("|"):
                        if p.lower().endswith(('.jpg', '.png')) and os.path.exists(p):
                            img_p = p;
                            break
                render_analysis_result(msg["decision"], msg["result_state"], image_path=img_p, msg_id=msg.get("id"))
            # Monitoring panel
            if msg.get("result_state"):
                rs = msg["result_state"]
                has_monitoring = any(k in rs for k in ["node_statuses", "overall_confidence", "conflicts_detected", "backtrack_history"])
                if has_monitoring:
                    with st.expander("📈 执行监控面板", expanded=False):
                        # Node timing
                        node_statuses = rs.get("node_statuses", {})
                        if node_statuses:
                            st.caption("**节点耗时**")
                            for name, ns in node_statuses.items():
                                dur = ns.get("duration_ms", 0)
                                status = ns.get("status", "?")
                                icon = "✅" if status == "success" else "❌" if status == "failed" else "⏳"
                                model = ns.get("model") or "rule/runtime"
                                retries = ns.get("retry_count", 0)
                                error_suffix = f" · {ns.get('error_code')}" if ns.get("error_code") else ""
                                st.text(
                                    f"{icon} {name}: {dur:.0f}ms · {model} · retries={retries}{error_suffix}"
                                )

                        # Confidence scores
                        conf_cols = st.columns(4)
                        conf_fields = [
                            ("文本置信度", "text_extraction_confidence"),
                            ("图例置信度", "image_extraction_confidence"),
                            ("融合置信度", "fusion_confidence"),
                            ("综合置信度", "overall_confidence"),
                        ]
                        for i, (label, key) in enumerate(conf_fields):
                            val = rs.get(key, None)
                            if val is not None:
                                color = "green" if val >= 0.7 else "orange" if val >= 0.4 else "red"
                                conf_cols[i].metric(label, f"{val:.2f}")

                        # Task routing log
                        routings = rs.get("task_routing_log", [])
                        if routings:
                            st.caption(f"**🧭 模型调度决策 ({len(routings)} 项)**")
                            for r in routings:
                                st.text(f"  {r.get('task', '?')} → {r.get('selected_model', '?')} ({r.get('reason', '?')})")

                        # Conflicts
                        conflicts = rs.get("conflicts_detected", [])
                        if conflicts:
                            st.caption(f"**⚠️ 冲突检测 ({len(conflicts)} 项)**")
                            for c in conflicts:
                                sev = c.get("severity", "?")
                                sev_icon = "🔴" if sev == "high" else "🟡" if sev == "medium" else "🟢"
                                st.text(f"{sev_icon} [{c.get('conflict_type', '?')}] {c.get('description', '?')}")

                        # Backtrack history
                        bh = rs.get("backtrack_history", [])
                        if bh:
                            st.caption(f"**🔄 回溯历史 ({len(bh)} 次)**")
                            for h in bh:
                                st.text(h)

                        execution_errors = rs.get("errors", [])
                        if execution_errors:
                            st.caption(f"**❌ 结构化错误 ({len(execution_errors)} 项)**")
                            for item in execution_errors:
                                st.text(
                                    f"{item.get('node', '?')} · {item.get('code', 'UNKNOWN')} · "
                                    f"{item.get('message', '')}"
                                )

# === 🚀 新增：隐形锚点与自动滚动 JS 逻辑 ===
# 在渲染完所有历史消息后，立即注入一个隐形 DOM，并通知前端瞬间滚动到此位置
st.markdown("<div id='chat-end-anchor'></div>", unsafe_allow_html=True)
components.html("""
<script>
    // 延时 150ms 确保 Streamlit 大组件和图表完全挂载渲染完毕
    setTimeout(function() {
        const parentDoc = window.parent.document;
        const anchor = parentDoc.getElementById('chat-end-anchor');
        if (anchor) {
            // 原生平滑滚动到这个锚点
            anchor.scrollIntoView({ behavior: 'smooth', block: 'end' });
        } else {
            // 兜底方案：尝试找到主容器直接推到底部
            const mainContainer = parentDoc.querySelector('.main') || parentDoc.querySelector('.stMainBlockContainer');
            if (mainContainer) {
                mainContainer.scrollTop = mainContainer.scrollHeight;
            }
        }
    }, 150);
</script>
""", height=0, width=0)

# ==========================================
# 🌟 100%纯净原生的 ChatGPT 交互式底部聊天框
# ==========================================

# 利用原生机制捕获底层输入控制区
if "bottom" in inspect.signature(st.container).parameters:
    bottom_ctrl = st.container(bottom=True)
else:
    bottom_ctrl = st.container()

with bottom_ctrl:
    if st.session_state.is_processing:
        # 1. 任务执行状态：在底层聊天框正上方悬浮一个优雅的"停止生成"按钮
        col1, col2, col3 = st.columns([3, 2, 3])
        with col2:
            if st.button("🛑 停止生成", type="secondary", use_container_width=True):
                # 强行打断底层的阻塞线程，恢复界面的闲置状态
                st.session_state.is_processing = False
                st.session_state.pending_prompt = None
                st.rerun()

        # 显示一个被禁用(disabled)的聊天框，防止用户在执行期间重复发送指令
        st.chat_input("⏳ 模型正在飞速计算中，请稍候...", disabled=True)

    else:
        # 2. 正常闲置状态：使用 Streamlit 最原生的 st.chat_input
        # 它自带回车发送、自适应多行文本输入、完美的移动端适配等全部高级特性！
        if prompt := st.chat_input("✍️ 请在此输入地质勘探指令 (按回车发送)..."):
            st.session_state.pending_prompt = prompt
            st.session_state.is_processing = True
            st.rerun()

# ==========================================
# 核心任务调度与大模型逻辑执行区
# ==========================================
if st.session_state.is_processing and st.session_state.pending_prompt:
    prompt = st.session_state.pending_prompt

    if not st.session_state.master_agent:
        st.error("引擎未连接，请检查环境配置中的 API_KEY。")
        st.session_state.is_processing = False
        st.session_state.pending_prompt = None
        st.stop()

    is_first_message = False
    if cur_sid is None:
        cur_sid = db.create_session(title="新对话")
        st.session_state.current_session_id = cur_sid
        is_first_message = True
    else:
        is_first_message = (len(msgs) == 0)

    all_curr_files = image_paths + file_paths
    curr_file_db = "|".join(all_curr_files) if all_curr_files else None

    if image_paths and file_paths:
        curr_type_db = "mixed"
    elif image_paths:
        curr_type_db = "image"
    elif any(f.endswith(('.csv', '.shp', '.tif')) for f in file_paths):
        curr_type_db = "data"
    elif file_paths:
        curr_type_db = "text"
    else:
        curr_type_db = None

    db.add_message(cur_sid, "user", prompt, file_path=curr_file_db, file_type=curr_type_db)

    if is_first_message:
        try:
            from langchain_community.chat_models import ChatTongyi
            from langchain_core.messages import HumanMessage

            title_llm = ChatTongyi(model="qwen-max", dashscope_api_key=st.session_state.master_agent.api_key,
                                   temperature=0.1)
            title_prompt = f"请将下面的提问精炼成一个极简的对话小标题（最多8个字）。要求：直接输出纯文本，绝不包含任何标点符号、引号或多余的解释说明。\n\n提问：{prompt}"
            title_res = title_llm.invoke([HumanMessage(content=title_prompt)])
            new_title = (title_res.content or "").strip().replace('"', '').replace("'", "").replace("。", "")
            if new_title: db.update_session_title(cur_sid, new_title)
        except Exception:
            logger.exception("Automatic session title generation failed")
            db.update_session_title(cur_sid, prompt[:8] + "...")

    # 在执行期间也带上高雅的头像
    with st.chat_message("user", avatar="👤"):
        if all_curr_files:
            for p in all_curr_files:
                if p.lower().endswith(('.jpg', '.png')):
                    st.image(p, width=300)
                else:
                    st.caption(f"📄 {os.path.basename(p)}")
        st.markdown(prompt)

    lc_hist = build_langchain_history(msgs)
    with st.chat_message("assistant", avatar="✨"):
        status = st.status("🧠 分析及生成中...", expanded=True)
        try:
            start_time = time.time()
            run_workspace = create_run_dir(cur_sid)
            result = st.session_state.master_agent.run(
                text=file_context, instruction=prompt, file_path=curr_file_db, image_path=image_path,
                chat_history=lc_hist, session_id=cur_sid, workspace_dir=run_workspace
            )
            elapsed_time = time.time() - start_time
            print(f">>> [System] 本次任务后台执行总耗时: {elapsed_time:.2f} 秒")

            current_logs = result.get("turn_thought_log", result.get("thought_log", []))
            for l in current_logs: status.write(l)
            status.update(label="✅ 完成", state="complete", expanded=False)

            decision = result.get("next_step")
            final_resp = result.get("final_response", "")

            render_analysis_result(decision, result, msg_id="current")
            st.markdown(final_resp)

            # Store only safe subset of state (MySQL JSON column has size/type limits)
            safe_result = serialize_result_state(result) or {}
            safe_result["thought_log"] = current_logs
            db.add_message(cur_sid, "assistant", final_resp, decision, safe_result, file_path=curr_file_db,
                           file_type=curr_type_db)
            st.session_state.uploader_key += 1

        except Exception as e:
            wrapped = AgentExecutionError("Top-level request execution failed", details=str(e))
            logger.exception("Top-level request execution failed: session_id=%s", cur_sid)
            status.update(label="❌ 出错", state="error")
            err_msg = public_error_message(wrapped)
            st.error(err_msg)
            db.add_message(cur_sid, "assistant", err_msg, decision="error", result_state={
                               "errors": [{"node": "application", "code": error_code(wrapped),
                                           "message": err_msg}]
                           },
                           file_path=curr_file_db, file_type=curr_type_db)

        st.session_state.is_processing = False
        st.session_state.pending_prompt = None
        st.rerun()
