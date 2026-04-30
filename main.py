import os
import time
import hashlib
import cv2
import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
import html
import inspect
from PIL import Image
from dotenv import load_dotenv

# === 🚀 核心修复：解除 PIL 库对超大分辨率地质图的“像素炸弹”防爆限制 ===
Image.MAX_IMAGE_PIXELS = None

os.environ["FOR_DISABLE_CONSOLE_CTRL_HANDLER"] = "1"

try:
    from master_agent import MasterAgent
except ImportError:
    MasterAgent = None

try:
    from db_manager import DBManager
except ImportError:
    class DBManager:
        def check_connection(self): return True

        def create_session(self, title="新对话"): return 1

        def get_all_sessions(self): return [{'id': 1, 'title': '默认会话'}]

        def get_history(self, sid): return []

        def add_message(self, sid, role, content, decision=None, result_state=None, file_path=None,
                        file_type=None): pass

        def delete_session(self, sid): pass

        def update_session_title(self, sid, title): pass

        def delete_empty_sessions(self, exclude_session_id=None): pass

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

st.set_page_config(page_title="基于多智能体的三维地质建模钻孔数据生成", page_icon="🌍", layout="wide", initial_sidebar_state="expanded")

try:
    db = DBManager()
    if not db.check_connection(): st.error("⚠️ 数据库连接失败。"); st.stop()
except Exception:
    st.error("数据库初始化错误");
    st.stop()

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
    except:
        pass

# === 🚀 样式代码：清理所有输入框黑魔法，仅保留内部组件美化 ===
st.markdown("""
<style>
    /* 全局小组件美化 */
    .agent-box { padding: 5px 8px; border-radius: 12px; border: 1px solid #eee; text-align: center; font-size: 0.75em; margin:2px; display:inline-block; }
    .agent-active { background-color: #e8f5e9; color: #2e7d32; border-color: #4CAF50; font-weight: bold; }
    .agent-inactive { background-color: #f5f5f5; color: #aaa; }
    .geo-card { background: #f0f8ff; padding: 15px; border-radius: 8px; border-left: 5px solid #2196F3; margin-top: 10px; font-size: 0.95em; line-height: 1.6; }
    .thought-process { background-color: #f8f9fa; border-left: 4px solid #6c757d; padding: 10px 15px; margin: 10px 0; border-radius: 0 4px 4px 0; font-family: monospace; font-size: 0.9em; color: #495057; }
    .thought-item { margin-bottom: 5px; }

    /* 侧边栏按钮美化 */
    [data-testid="stSidebar"] button[kind="secondary"] { justify-content: flex-start !important; padding-left: 15px !important; }
    [data-testid="stSidebar"] button[kind="secondary"] > div[data-testid="stMarkdownContainer"],
    [data-testid="stSidebar"] button[kind="secondary"] > div { width: 100% !important; display: flex !important; justify-content: flex-start !important; }
    [data-testid="stSidebar"] button[kind="secondary"] p { font-size: 0.95rem; margin: 0 !important; text-align: left !important; width: 100% !important; display: flex !important; justify-content: flex-start !important; }
    [data-testid="stSidebar"] button[kind="primary"] { justify-content: center !important; }
    [data-testid="stSidebar"] hr { margin-top: 1rem; margin-bottom: 1rem; opacity: 0.5; }
    .legend-row { display: flex; align-items: center; margin-bottom: 5px; border-bottom: 1px solid #eee; padding: 5px 0; }
    .legend-img { width: 40px; height: 30px; object-fit: cover; border-radius: 4px; margin-right: 10px; border:1px solid #ddd; }

    /* 悬浮终止按钮的底层容器透明化，打造漂浮感 */
    [data-testid="stBottom"] > div {
        background-color: transparent !important;
        padding-bottom: 0px !important;
    }
</style>
""", unsafe_allow_html=True)


def safe_get(obj, key):
    if isinstance(obj, dict): return obj.get(key)
    return getattr(obj, key, None)


def highlight_contour_on_image(image_path, contour, color=(0, 0, 255), thickness=5):
    if not os.path.exists(image_path): return None
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
        except:
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
                except Exception as e:
                    st.warning(f"JSON 序列化显示失败: {e}")

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
    thought_log = safe_get(result_state, 'thought_log')
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

        if annotated_map_base64 or cropped_legend_base64:
            st.markdown("##### 🎯 智能图例区域定位")
            cols = st.columns(2)
            if annotated_map_base64:
                with cols[0]:
                    st.image(f"data:image/jpeg;base64,{annotated_map_base64}", caption="🗺️ 自动圈定的图例分布区",
                             width="stretch")
            if cropped_legend_base64:
                with cols[1]:
                    st.image(f"data:image/jpeg;base64,{cropped_legend_base64}", caption="✂️ 精确截取的图例内容",
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
    pass

_all_sessions = db.get_all_sessions()
_valid_ids = [s['id'] for s in _all_sessions]
if st.session_state.current_session_id is not None and st.session_state.current_session_id not in _valid_ids:
    st.session_state.current_session_id = _valid_ids[0] if _valid_ids else None

cur_sid = st.session_state.current_session_id
msgs = db.get_history(cur_sid) if cur_sid is not None else []
display_sessions = db.get_all_sessions()

with st.sidebar:
    st.header("⚙️ 控制面板")

    if st.button("🔄 重置当前对话状态", type="secondary", use_container_width=True):
        st.session_state.is_processing = False
        st.session_state.pending_prompt = None
        st.rerun()

    st.divider()

    st.title("🌍 地质智能助手")
    if st.button("➕ 新建对话", type="primary", width="stretch"):
        st.session_state.current_session_id = None
        st.session_state.uploader_key += 1
        st.rerun()
    st.divider()

    st.caption("🤖 大模型引擎配置")
    model_options = ["qwen-max", "qwen-plus", "deepseek-r1"]
    selected_model = st.selectbox(
        "选择任务执行核心模型",
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
                st.session_state.master_agent = MasterAgent(api_key=ENV_API_KEY, model_name=selected_model)
                st.toast(f"计算引擎已成功切换至: {selected_model}", icon="🔄")
            except Exception as e:
                st.error(f"引擎初始化失败: {e}")
    st.divider()

    st.caption("💬 历史会话")
    for s in display_sessions:
        col1, col2 = st.columns([8.5, 1.5])
        with col1:
            is_active = (cur_sid == s['id'])
            prefix = "🔹 " if is_active else "🔸 "
            if st.button(f"{prefix}{s['title']}", key=f"s_{s['id']}", width="stretch"):
                st.session_state.current_session_id = s['id']
                st.rerun()
        with col2:
            if st.button("🗑️", key=f"del_{s['id']}", help="删除此会话", width="stretch"):
                db.delete_session(s['id'])
                if cur_sid == s['id']: st.session_state.current_session_id = None
                st.rerun()
    st.divider()

    st.caption("📂 当前会话工作区 (文件投喂)")
    uploaded_files = st.file_uploader(
        "文件", type=["txt", "jpg", "png", "csv", "tif", "tiff", "shp", "shx", "dbf", "prj", "cpg"],
        accept_multiple_files=True, label_visibility="collapsed", key=f"file_uploader_{st.session_state.uploader_key}"
    )
    file_paths = []
    image_paths = []
    file_context = ""

    if uploaded_files:
        for uploaded in uploaded_files:
            fname = uploaded.name
            temp_dir = os.path.join(current_dir, "temp")
            if not os.path.exists(temp_dir): os.makedirs(temp_dir)
            ext = os.path.splitext(fname)[1].lower()
            if ext in ['.shp', '.shx', '.dbf', '.prj', '.cpg']:
                base_name = os.path.splitext(fname)[0]
                shp_folder = os.path.join(temp_dir, base_name)
                if not os.path.exists(shp_folder): os.makedirs(shp_folder)
                save_path = os.path.join(shp_folder, fname)
            else:
                save_path = os.path.join(temp_dir, fname)

            with open(save_path, "wb") as f:
                f.write(uploaded.getbuffer())

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
                        except:
                            content = uploaded.getvalue().decode("utf-8", errors="ignore")
                    file_context += f"\n--- {fname} ---\n{content}\n"

    image_path = image_paths[0] if image_paths else ""
    file_path = file_paths[0] if file_paths else ""

# ==========================================
# 主界面聊天与逻辑
# ==========================================
current_title = "✨ 智能分析终端"
if cur_sid is None:
    current_title = "✨ 新对话"
else:
    for s in display_sessions:
        if s['id'] == cur_sid:
            current_title = f"💬 {s['title']}"
            break

st.header(current_title)
if cur_sid is None:
    st.info("👋 欢迎！这是一个全新的对话。请在左侧侧边栏上传分析所需的文件，或者在下方直接向我发送指令。")

for msg in msgs:
    # 🌟 为聊天气泡引入拟人化头像，大幅提升沉浸感
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
        # 1. 任务执行状态：在底层聊天框正上方悬浮一个优雅的“停止生成”按钮
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
            new_title = title_res.content.strip().replace('"', '').replace("'", "").replace("。", "")
            if new_title: db.update_session_title(cur_sid, new_title)
        except Exception:
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

    lc_hist = []
    with st.chat_message("assistant", avatar="✨"):
        status = st.status("🧠 分析及生成中...", expanded=True)
        try:
            start_time = time.time()
            result = st.session_state.master_agent.run(
                text=file_context, instruction=prompt, file_path=curr_file_db, image_path=image_path,
                chat_history=lc_hist
            )
            elapsed_time = time.time() - start_time
            print(f">>> [System] 本次任务后台执行总耗时: {elapsed_time:.2f} 秒")

            for l in result.get("thought_log", []): status.write(l)
            status.update(label="✅ 完成", state="complete", expanded=False)

            decision = result.get("next_step")
            final_resp = result.get("final_response", "")

            render_analysis_result(decision, result, msg_id="current")
            st.markdown(final_resp)

            db.add_message(cur_sid, "assistant", final_resp, decision, result, file_path=curr_file_db,
                           file_type=curr_type_db)
            st.session_state.uploader_key += 1

        except Exception as e:
            status.update(label="❌ 出错", state="error")
            err_msg = f"执行过程中发生系统异常: {str(e)}"
            st.error(err_msg)
            db.add_message(cur_sid, "assistant", err_msg, decision="error", result_state={"error": str(e)},
                           file_path=curr_file_db, file_type=curr_type_db)

        st.session_state.is_processing = False
        st.session_state.pending_prompt = None
        st.rerun()