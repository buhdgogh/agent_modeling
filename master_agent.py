#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Master orchestration agent — clean, stable version."""

import os, re, time, json
from typing import TypedDict, Optional, Any, List
from langchain_community.chat_models import ChatTongyi
from langgraph.graph import StateGraph, START, END
from langchain_core.tools import tool
from langchain_core.messages import AIMessage, SystemMessage, HumanMessage, BaseMessage
import concurrent.futures

try:
    from langchain_text_splitters import RecursiveCharacterTextSplitter; HAS_SPLITTER = True
except ImportError:
    try:
        from langchain.text_splitter import RecursiveCharacterTextSplitter; HAS_SPLITTER = True
    except ImportError:
        HAS_SPLITTER = False

try:
    from text_info_agent import TextInfoAgent
    from kg_builder import KGBuilderAgent
    from image_agent import ImageAgent
    from gempy_agent import GempyAgent
except ImportError:
    pass

def S(s):
    """Safe strip."""
    if s is None or not isinstance(s, str): return ""
    return s.strip()

class MasterState(TypedDict):
    input_text: str; file_path: str; image_path: str
    user_instruction: str; chat_history: List[BaseMessage]
    next_step: str
    final_text_info: Optional[Any]; final_kg: Optional[Any]
    final_image_analysis: Optional[Any]; final_boreholes: Optional[Any]
    final_response: str; thought_log: List[str]; messages: List[Any]

class MasterAgent:
    def __init__(self, api_key: str, model_name: str = "qwen-max"):
        self.model_name = model_name; self.api_key = api_key
        if not api_key: raise ValueError("API Key required.")

        self.text_info_bot = TextInfoAgent(api_key=api_key, model_name=model_name)
        self.kg_bot = KGBuilderAgent(api_key=api_key, model_name=model_name)
        self.image_bot = ImageAgent(api_key=api_key, model_name="qwen-vl-max")
        self.borehole_bot = GempyAgent(api_key=api_key, model_name=model_name)

        self.tools_map = {}  # filled below

        # === Tools ===
        @tool("extract_text_info_tool")
        def extract_text_info_tool(text: str) -> dict:
            """Extract stratigraphic formation and profile info from geological text."""
            res = self.text_info_bot.run(text) or {}
            return {"type": "text_info", "data": res.get("extraction")}

        @tool("build_kg_tool")
        def build_kg_tool(text: str) -> dict:
            """Build knowledge graph from geological entities, store to Neo4j."""
            res = self.kg_bot.run(text) or {}
            return {"type": "kg", "data": res.get("kg_data")}

        @tool("analyze_image_tool")
        def analyze_image_tool(instruction: str, image_path: str = "") -> dict:
            """Analyze geological map: detect legends, segment regions, match colors."""
            if not image_path or not os.path.exists(image_path):
                return {"type": "error", "data": f"Image not found: {image_path}"}
            res = self.image_bot.run(image_path, instruction) or {}
            return {"type": "image", "data": res.get("analysis")}

        @tool("auto_borehole_pipeline")
        def auto_borehole_pipeline(instruction: str, text_content: str = "", image_path: str = "",
                                   shp_path: str = "", tif_path: str = "", csv_path: str = "") -> dict:
            """Full auto borehole workflow: text+image fusion -> CSV -> spatial compute."""
            try:
                final_csv_path = csv_path
                if not final_csv_path or not os.path.exists(final_csv_path):
                    # --- Parallel: text + image extraction (with caching) ---
                    _img_cache = [None, None]  # [legends_list, summary_str]
                    def _get_img():
                        if image_path and os.path.exists(image_path):
                            r = self.image_bot.run(image_path, "提取图例顺序与地层代号") or {}
                            a = r.get("analysis")
                            legends = []
                            if a:
                                if hasattr(a, 'legends') and a.legends:
                                    legends = [l.model_dump() if hasattr(l,'model_dump') else l for l in a.legends]
                                summary = str(a.summary) if hasattr(a,'summary') else str(a)
                                _img_cache[0] = legends
                                _img_cache[1] = summary
                                return legends, summary
                            _img_cache[0] = []
                            _img_cache[1] = "无图件数据"
                            return [], "无图件数据"
                        _img_cache[0] = []
                        _img_cache[1] = "无图件数据"
                        return [], "无图件数据"

                    def _get_txt():
                        if text_content and text_content.strip():
                            r = self.text_info_bot.run(text_content) or {}
                            e = r.get("extraction")
                            strata, profiles = [], []
                            if e:
                                if hasattr(e, 'strata'): strata = list(e.strata)
                                if hasattr(e, 'profiles'): profiles = list(e.profiles)
                                return strata, profiles, str(e)
                            return [], [], str(e) if e else "无文本数据"
                        return [], [], "无文本数据"

                    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as ex:
                        f_img = ex.submit(_get_img)
                        f_txt = ex.submit(_get_txt)
                        legends_list, legends_summary = f_img.result()
                        strata_list, profiles_list, entities_summary = f_txt.result()

                    # Serialize for fusion prompt
                    legends_data = legends_summary if legends_summary else json.dumps(legends_list, ensure_ascii=False)
                    entities_data = entities_summary

                    ls = (legends_data or "").replace("{","[").replace("}","]")
                    es = (entities_data or "").replace("{","[").replace("}","]")

                    fusion_prompt = (
                        "你是一个地质数据对齐专家。请结合图例和文本信息，生成标准虚拟钻孔地层CSV。\n"
                        "必须严格8列: part_code,formation,formation_code,formation_age_1,"
                        "formation_age_code_1,formation_age_2,formation_age_code_2,厚度\n"
                        "part_code固定part_1; formation_code从上到下(新到老)填1,2,3...整数; "
                        "厚度默认30(单位米)。\n"
                        f"图例序列:\n{ls[:1500]}\n"
                        f"文本提取:\n{es[:3000]}\n"
                        "直接输出纯文本CSV(含表头)，不要Markdown标记。"
                    )

                    llm = ChatTongyi(model=self.model_name, dashscope_api_key=self.api_key,
                                     temperature=0.01, max_tokens=3000, max_retries=2)
                    try:
                        resp = llm.invoke([HumanMessage(content=fusion_prompt)])
                        raw = (resp.content or "") if resp else ""
                    except Exception as e:
                        print(f">>> [Pipeline] Fusion LLM timeout/error: {str(e)[:100]}")
                        raw = ""

                    if not raw or not raw.strip():
                        raise Exception("Fusion LLM returned empty response (API timeout or error)")

                    if "deepseek" in self.model_name.lower():
                        raw = re.sub(r"<think>.*?</think>", "", raw or "", flags=re.DOTALL).strip()

                    hp = r"(part_code\s*,\s*formation\s*,\s*formation_code\s*,\s*formation_age_1\s*,\s*formation_age_code_1\s*,\s*formation_age_2\s*,\s*formation_age_code_2\s*,\s*厚度.*)"
                    m = re.search(hp, raw or "", re.IGNORECASE|re.DOTALL)
                    if m:
                        csv_content = (m.group(1) or "").strip()
                        csv_content = re.sub(r"\n\s*```.*$","",csv_content,flags=re.DOTALL|re.IGNORECASE).strip()
                    else:
                        csv_content = (raw or "").replace("```csv","").replace("```","").strip()

                    td = os.path.join(os.path.dirname(os.path.abspath(__file__)), "temp")
                    os.makedirs(td, exist_ok=True)
                    final_csv_path = os.path.join(td, "auto_generated_formation.csv")
                    with open(final_csv_path, "w", encoding="utf-8-sig") as f:
                        f.write(csv_content)

                # --- Spatial compute ---
                if not shp_path:
                    return {"type": "boreholes",
                            "data": {"status": "PartialSuccess", "note": "No SHP, CSV only",
                                     "csv_file_path": final_csv_path or ""}}

                # Find regions_ui.json for spatial constraints
                rj = ""
                od = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
                if os.path.exists(od):
                    for sd in sorted([d for d in os.listdir(od) if os.path.isdir(os.path.join(od,d))], reverse=True):
                        rp = os.path.join(od, sd, "regions_ui.json")
                        if os.path.exists(rp): rj = rp; break

                fps = f"{shp_path}|{tif_path}|{final_csv_path}"
                if rj: fps += f"|{rj}"
                gres = self.borehole_bot.run(text="", file_paths=fps) or {}
                bh_data = gres.get("borehole_result", {"status":"Unknown"})

                if not tif_path:
                    bh_data["dem_note"] = "No DEM, default Z=0"

                return {"type": "boreholes", "data": bh_data}
            except Exception as e:
                return {"type": "boreholes", "data": {"status": f"Pipeline Error: {str(e)}"}}

        self.tools = [extract_text_info_tool, build_kg_tool, analyze_image_tool, auto_borehole_pipeline]
        self.tools_map = {t.name: t for t in self.tools}
        self.router_llm = ChatTongyi(model="qwen-max", dashscope_api_key=api_key,
                                     temperature=0, max_tokens=1000, max_retries=2,
                                     request_timeout=120).bind_tools(self.tools)
        if HAS_SPLITTER:
            self.splitter = RecursiveCharacterTextSplitter(chunk_size=15000, chunk_overlap=200)
        self.graph = self._build_graph()

    # === GRAPH NODES ===
    def _decider_node(self, state: MasterState):
        inst = S(state.get("user_instruction",""))
        text = state.get("input_text",""); fp = state.get("file_path","")
        ip = state.get("image_path","")
        logs = [f"Router: {inst[:50]}"]

        if not inst:
            return {"messages":[], "final_response":"请输入指令。","next_step":"general_chat","thought_log":logs}

        has_text = bool(text and text.strip())
        has_image = bool(ip and os.path.exists(ip))

        sp = ("你是地质智能中枢。\n"
              "1.虚拟钻孔/建模: 同时有图件+文本+文件时 -> auto_borehole_pipeline\n"
              "2.仅图件分析: -> analyze_image_tool\n"
              "3.仅文本提取: -> extract_text_info_tool\n"
              "4.知识图谱: -> build_kg_tool\n"
              "5.其他: 直接回答")

        msgs = [SystemMessage(content=sp)] + (state.get("chat_history",[]) or [])
        parts = []
        if fp: parts.append(f"[Files] {fp}")
        if ip: parts.append(f"[Image] {ip}")
        if has_text: parts.append(f"[Text] {len(text)} chars")
        parts.append(f"[Task] {inst}")
        msgs.append(HumanMessage(content="\n".join(parts)))

        for attempt in range(3):
            try:
                resp = self.router_llm.invoke(msgs)
                if resp is None or not resp.tool_calls:
                    logs.append("Decision: chat")
                    rc = (resp.content or "") if resp else ""
                    return {"messages":[resp] if resp else [],
                            "final_response":rc or "(empty)","next_step":"general_chat","thought_log":logs}
                names = [tc["name"] for tc in resp.tool_calls]
                logs.append(f"Decision: {', '.join(names)}")
                return {"messages":[resp],"thought_log":logs}
            except Exception as e:
                if attempt < 2: logs.append(f"Retry {attempt+1}..."); time.sleep(2)
                else: return {"messages":[],"final_response":f"Router error: {str(e)[:100]}",
                              "next_step":"error","thought_log":logs}

    def _tool_execution_node(self, state: MasterState):
        msgs = state.get("messages",[]) or []
        last = msgs[-1] if msgs else None
        logs = list(state.get("thought_log",[]) or [])
        updates = {"final_response":"Failed.","next_step":"finish","thought_log":logs}

        if not (isinstance(last, AIMessage) and last.tool_calls):
            return updates

        tc = last.tool_calls[0]
        tname, targs = tc["name"], dict(tc.get("args",{}) or {})

        # Inject state parameters
        if tname in ["extract_text_info_tool","build_kg_tool"]:
            ft = state.get("input_text","")
            if ft.strip(): targs["text"] = ft
        if tname == "analyze_image_tool":
            rip = state.get("image_path","")
            if rip and os.path.exists(rip): targs["image_path"] = rip
        if tname == "auto_borehole_pipeline":
            fps = state.get("file_path","") or ""
            shp = tif = csv = ""
            for p in fps.split("|"):
                pl = p.lower()
                if pl.endswith('.shp'): shp = p
                elif pl.endswith(('.tif','.tiff')): tif = p
                elif pl.endswith('.csv'): csv = p
            targs.update({"text_content":state.get("input_text",""),"shp_path":shp,
                          "tif_path":tif,"csv_path":csv,"instruction":state.get("user_instruction","")})
            rip = state.get("image_path","")
            if rip and os.path.exists(rip): targs["image_path"] = rip

        tool = self.tools_map.get(tname)
        if not tool: return updates

        try:
            logs.append(f"Exec: {tname}")
            out = tool.invoke(targs) or {}
            dtp = out.get("type"); dc = out.get("data")

            if dtp == "text_info":
                updates.update({"final_text_info":dc,"next_step":"extract_text_info","final_response":"Text done."})
            elif dtp == "kg":
                updates.update({"final_kg":dc,"next_step":"build_kg","final_response":"KG done."})
            elif dtp == "image":
                updates.update({"final_image_analysis":dc,"next_step":"analyze_image","final_response":"Image done."})
            elif dtp == "boreholes":
                st = (dc or {}).get("status","Unknown") if isinstance(dc,dict) else "Unknown"
                updates.update({"final_boreholes":dc,"next_step":"generate_boreholes",
                                "final_response":f"Boreholes: {st}"})
            elif dtp == "error":
                updates["final_response"] = f"Error: {dc}"
            logs.append("Done.")
        except Exception as e:
            updates["final_response"] = f"Tool error: {str(e)[:100]}"
            logs.append(f"Error: {str(e)[:100]}")

        updates["thought_log"] = logs
        return updates

    # === GRAPH ===
    def _should_execute(self, state):
        msgs = state.get("messages",[]) or []
        last = msgs[-1] if msgs else None
        if isinstance(last, AIMessage) and last.tool_calls: return "execute"
        return "end"

    def _build_graph(self):
        wf = StateGraph(MasterState)
        wf.add_node("decider", self._decider_node)
        wf.add_node("tool_executor", self._tool_execution_node)
        wf.add_edge(START, "decider")
        wf.add_conditional_edges("decider", self._should_execute, {"execute":"tool_executor","end":END})
        wf.add_edge("tool_executor", END)
        return wf.compile()

    # === ENTRY ===
    def run(self, text: str, instruction: str = "", file_path: str = "", image_path: str = "",
            chat_history: Optional[List[BaseMessage]] = None):
        return self.graph.invoke({
            "input_text":text,"file_path":file_path,"image_path":image_path,
            "user_instruction":instruction,"chat_history":chat_history or [],
            "messages":[],"thought_log":[],
        })
