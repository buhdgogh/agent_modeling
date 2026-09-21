#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Master orchestration agent — clean, stable version."""

import os, re, time, json, sqlite3, logging
from operator import add
from typing import Annotated, TypedDict, Optional, Any, List
from langchain_community.chat_models import ChatTongyi
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.types import Overwrite
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
try:
    from langgraph.checkpoint.sqlite import SqliteSaver
except ImportError:
    SqliteSaver = None
from langchain_core.tools import tool
from langchain_core.messages import AIMessage, SystemMessage, HumanMessage, ToolMessage, BaseMessage
import concurrent.futures
from workspace_manager import checkpoint_db_path, create_run_dir
from runtime_config import configure_tracing
from fusion_validation import repair_fusion, validate_fusion_csv
from fusion_generation import build_fusion_prompt, generate_fusion_csv
from fusion_matching import match_entities
from app_errors import (
    AgentModelingError,
    AgentExecutionError,
    ConfigurationError,
    FileValidationError,
    ModelInvocationError,
    error_code,
    public_error_message,
)

MAX_CHECKPOINT_MESSAGES = 60
MAX_CHECKPOINT_LOGS = 200
logger = logging.getLogger(__name__)


def merge_monitoring(left: Optional[dict], right: Optional[dict]) -> dict:
    return {**(left or {}), **(right or {})}


def _node_status(started_at: float, status: str, *, model: str = "",
                 retry_count: int = 0, error: Optional[Exception] = None) -> dict:
    record = {
        "status": status,
        "duration_ms": round((time.perf_counter() - started_at) * 1000, 2),
        "model": model,
        "retry_count": retry_count,
        "error_code": error_code(error) if error else None,
    }
    return record


def _estimate_confidence(value: Any) -> Optional[float]:
    values = []

    def visit(item: Any, depth: int = 0):
        if depth > 4 or len(values) >= 100:
            return
        if hasattr(item, "model_dump"):
            item = item.model_dump()
        elif hasattr(item, "dict"):
            item = item.dict()
        if isinstance(item, dict):
            for key, child in item.items():
                if "confidence" in str(key).lower() and isinstance(child, (int, float)):
                    score = float(child)
                    if 0 <= score <= 1:
                        values.append(score)
                else:
                    visit(child, depth + 1)
        elif isinstance(item, (list, tuple)):
            for child in item:
                visit(child, depth + 1)

    visit(value)
    return round(sum(values) / len(values), 4) if values else None

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
    logger.exception("Failed to import one or more specialist agents")
    raise

def S(s):
    """Safe strip."""
    if s is None or not isinstance(s, str): return ""
    return s.strip()

class MasterState(TypedDict):
    fixed_task_type: str
    input_text: str; file_path: str; image_path: str
    workspace_dir: str
    user_instruction: str; chat_history: List[BaseMessage]
    next_step: str
    final_text_info: Optional[Any]; final_kg: Optional[Any]
    final_image_analysis: Optional[Any]; final_boreholes: Optional[Any]
    final_response: str
    thought_log: Annotated[List[str], add]
    messages: Annotated[List[BaseMessage], add_messages]
    node_statuses: Annotated[dict, merge_monitoring]
    task_routing_log: Annotated[List[dict], add]
    errors: Annotated[List[dict], add]
    overall_confidence: Optional[float]
    text_extraction_confidence: Optional[float]
    image_extraction_confidence: Optional[float]
    fusion_confidence: Optional[float]

class MasterAgent:
    def __init__(self, api_key: str, model_name: str = "qwen-max", *,
                 routing_mode: str = "dynamic", enable_backtracking: bool = True):
        if routing_mode not in {"dynamic", "fixed"}:
            raise ValueError("routing_mode must be dynamic or fixed")
        self.routing_mode = routing_mode
        self.enable_backtracking = enable_backtracking
        configure_tracing()
        self.model_name = model_name; self.api_key = api_key
        if not api_key:
            raise ConfigurationError("DASHSCOPE_API_KEY is missing")

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
        def analyze_image_tool(instruction: str, image_path: str = "", workspace_dir: str = "") -> dict:
            """Analyze geological map: detect legends, segment regions, match colors."""
            if not image_path or not os.path.exists(image_path):
                wrapped = FileValidationError("Image file does not exist", details=image_path)
                logger.warning("Image analysis requested with a missing file: %s", image_path)
                return {"type": "error", "data": {
                    "status": "Error", "error_code": error_code(wrapped),
                    "message": public_error_message(wrapped),
                }}
            image_output_dir = os.path.join(workspace_dir, "image_analysis") if workspace_dir else ""
            res = self.image_bot.run(image_path, instruction, output_dir=image_output_dir) or {}
            return {"type": "image", "data": res.get("analysis")}

        @tool("auto_borehole_pipeline")
        def auto_borehole_pipeline(instruction: str, text_content: str = "", image_path: str = "",
                                   shp_path: str = "", tif_path: str = "", csv_path: str = "",
                                   workspace_dir: str = "", extraction_bundle: Optional[dict] = None) -> dict:
            """Full auto borehole workflow: text+image fusion -> CSV -> spatial compute."""
            try:
                image_confidence = None
                text_confidence = None
                final_csv_path = csv_path
                fusion_trace = []
                td = workspace_dir or os.path.join(os.path.dirname(os.path.abspath(__file__)), "temp")
                os.makedirs(td, exist_ok=True)
                trace_path = os.path.join(td, "fusion_validation_trace.json")
                def save_trace(trace):
                    with open(trace_path, "w", encoding="utf-8") as stream:
                        json.dump({"model": self.model_name,
                                   "backtracking_enabled": self.enable_backtracking,
                                   "validation_scope": "configuration_only", "checks": trace},
                                  stream, ensure_ascii=False, indent=2)
                if not final_csv_path or not os.path.exists(final_csv_path):
                    # --- Parallel: text + image extraction (with caching) ---
                    _img_cache = [None, None]  # [legends_list, summary_str]
                    def _get_img():
                        if extraction_bundle is not None:
                            return extraction_bundle["image"]
                        if image_path and os.path.exists(image_path):
                            image_output_dir = os.path.join(workspace_dir, "image_analysis") if workspace_dir else ""
                            r = self.image_bot.run(
                                image_path,
                                "完整解析地质图件，识别图例并分割地图区域",
                                output_dir=image_output_dir
                            ) or {}
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
                        if extraction_bundle is not None:
                            return extraction_bundle["text"]
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
                    image_confidence = _estimate_confidence(legends_list)
                    text_confidence = _estimate_confidence([strata_list, profiles_list])

                    alignment_path = os.path.join(td, "fusion_alignment_audit.json")
                    with open(alignment_path, "w", encoding="utf-8") as stream:
                        json.dump(match_entities(strata_list, legends_list), stream,
                                  ensure_ascii=False, indent=2)

                    fusion_prompt = build_fusion_prompt(legends_list, strata_list, profiles_list)

                    llm = ChatTongyi(model=self.model_name, dashscope_api_key=self.api_key,
                                     temperature=0.01, max_tokens=8000, max_retries=2)
                    def generate_checked_csv(previous, issues):
                        return generate_fusion_csv(llm, fusion_prompt, previous, issues)
                    try:
                        csv_content, fusion_trace = repair_fusion(
                            generate_checked_csv,
                            max_repairs=2 if self.enable_backtracking else 0,
                            on_check=save_trace,
                        )
                    except Exception as exc:
                        wrapped = ModelInvocationError("Fusion model invocation failed", details=str(exc))
                        logger.exception("Fusion model invocation failed")
                        raise wrapped from exc

                    if csv_content is None:
                        return {"type": "boreholes", "data": {
                            "status": "NeedsReview", "message": "融合配置校验失败，已停止空间推演，请人工复核。",
                            "validation_trace_path": trace_path,
                            "fusion_validation": fusion_trace}}
                    final_csv_path = os.path.join(td, "auto_generated_formation.csv")
                    with open(final_csv_path, "w", encoding="utf-8-sig") as f:
                        f.write(csv_content)

                else:
                    # User-supplied configurations are checked but never rewritten by an LLM.
                    with open(final_csv_path, encoding="utf-8-sig") as stream:
                        supplied_csv = stream.read()
                    issues = validate_fusion_csv(supplied_csv)
                    fusion_trace = [{"round": 0, "stage": "supplied_csv", "passed": not issues,
                                     "issues": issues, "csv": supplied_csv}]
                    save_trace(fusion_trace)
                    if issues:
                        return {"type": "boreholes", "data": {
                            "status": "NeedsReview", "message": "输入CSV未通过配置校验，请人工修改。",
                            "validation_trace_path": trace_path, "fusion_validation": fusion_trace}}

                # --- Spatial compute ---
                if not shp_path:
                    return {"type": "boreholes",
                            "data": {"status": "PartialSuccess", "note": "No SHP, CSV only",
                                     "csv_file_path": final_csv_path or "",
                                     "validation_trace_path": trace_path, "fusion_validation": fusion_trace}}

                # Find spatial constraints only inside this request workspace.
                rj = ""
                if workspace_dir and os.path.isdir(workspace_dir):
                    candidates = []
                    for root, _, files in os.walk(workspace_dir):
                        if "regions_ui.json" in files:
                            candidates.append(os.path.join(root, "regions_ui.json"))
                    if candidates:
                        rj = max(candidates, key=os.path.getmtime)

                fps = f"{shp_path}|{tif_path}|{final_csv_path}"
                if rj: fps += f"|{rj}"
                gres = self.borehole_bot.run(text="", file_paths=fps, output_dir=workspace_dir) or {}
                bh_data = gres.get("borehole_result", {"status":"Unknown"})
                bh_data["validation_trace_path"] = trace_path
                bh_data["fusion_validation"] = fusion_trace

                if not tif_path:
                    bh_data["dem_note"] = "No DEM, default Z=0"

                available_confidences = [
                    value for value in (text_confidence, image_confidence) if value is not None
                ]
                if available_confidences:
                    bh_data["confidence"] = round(
                        sum(available_confidences) / len(available_confidences), 4
                    )
                    bh_data["confidence_components"] = {
                        "text": text_confidence,
                        "image": image_confidence,
                    }

                return {"type": "boreholes", "data": bh_data}
            except Exception as exc:
                wrapped = exc if isinstance(exc, AgentModelingError) else AgentExecutionError(
                    "Automatic borehole pipeline failed", details=str(exc)
                )
                logger.exception("Automatic borehole pipeline failed")
                return {"type": "boreholes", "data": {
                    "status": "Error",
                    "error_code": error_code(wrapped),
                    "message": public_error_message(wrapped),
                }}

        self.tools = [extract_text_info_tool, build_kg_tool, analyze_image_tool, auto_borehole_pipeline]
        self.tools_map = {t.name: t for t in self.tools}
        self.router_llm = ChatTongyi(model=self.model_name, dashscope_api_key=api_key,
                                     temperature=0, max_tokens=1000, max_retries=2,
                                     request_timeout=120).bind_tools(self.tools)
        if HAS_SPLITTER:
            self.splitter = RecursiveCharacterTextSplitter(chunk_size=15000, chunk_overlap=200)
        if SqliteSaver is None:
            raise RuntimeError(
                "缺少持久化 Checkpoint 依赖，请执行: "
                "pip install -r requirements-checkpoint.txt"
            )
        self._checkpoint_conn = sqlite3.connect(
            checkpoint_db_path(), timeout=30, check_same_thread=False
        )
        trusted_types = [
            ("image_agent", name) for name in (
                "ImageAnalysisResult", "LegendItem", "GeoSymbol", "RegionItem",
                "RegionGeoInfo", "RegionMatch",
            )
        ] + [
            ("text_info_agent", name) for name in (
                "TextInfoResult", "LLMExtraction", "StratumItem", "ProfileItem",
            )
        ] + [("kg_builder", name) for name in ("KnowledgeGraphData", "Node", "Edge")]
        self.checkpointer = SqliteSaver(
            self._checkpoint_conn,
            serde=JsonPlusSerializer(allowed_msgpack_modules=trusted_types),
        )
        self.checkpointer.setup()
        self.graph = self._build_graph()

    # === GRAPH NODES ===
    def _decider_node(self, state: MasterState):
        started_at = time.perf_counter()
        inst = S(state.get("user_instruction",""))
        text = state.get("input_text",""); fp = state.get("file_path","")
        ip = state.get("image_path","")
        logs = [f"Router: {inst[:50]}"]

        def finish(payload: dict, *, status: str = "success", retry_count: int = 0,
                   reason: str = "", selected: str = "qwen-max",
                   error: Optional[Exception] = None) -> dict:
            payload["node_statuses"] = {
                "decider": _node_status(
                    started_at, status, model="rule" if selected == "rule" else self.model_name,
                    retry_count=retry_count, error=error
                )
            }
            if reason:
                payload["task_routing_log"] = [{
                    "task": inst[:120] or "empty_instruction",
                    "selected_model": selected,
                    "reason": reason,
                }]
            if error:
                payload["errors"] = [{
                    "node": "decider",
                    "code": error_code(error),
                    "message": public_error_message(error),
                }]
            return payload

        if not inst:
            return finish(
                {"messages":[], "final_response":"请输入指令。","next_step":"general_chat","thought_log":logs},
                reason="用户指令为空，跳过工具调用", selected="rule"
            )

        has_text = bool(text and text.strip())
        has_image = bool(ip and os.path.exists(ip))

        if self.routing_mode == "fixed":
            task_type = state.get("fixed_task_type", "")
            fixed_tools = {"text": "extract_text_info_tool", "image": "analyze_image_tool",
                           "fusion_borehole": "auto_borehole_pipeline", "kg": "build_kg_tool"}
            if task_type not in fixed_tools:
                return finish({"messages": [], "next_step": "error", "thought_log": logs,
                               "final_response": "固定流程需要预先指定任务类型。"},
                              status="failed", selected="rule", reason="固定任务类型缺失或不支持")
            name = fixed_tools[task_type]
            args = {"text": text} if task_type in {"text", "kg"} else {"instruction": inst}
            if task_type == "image":
                args["image_path"] = ip
            response = AIMessage(content="", tool_calls=[{
                "name": name, "args": args, "id": "fixed_task_call", "type": "tool_call"}])
            return finish({"messages": [response], "thought_log": logs + [f"Fixed: {name}"]},
                          selected="rule", reason="执行预先指定的任务专用固定流程")

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
                    return finish(
                        {"messages":[resp] if resp else [],
                         "final_response":rc or "(empty)","next_step":"general_chat","thought_log":logs},
                        retry_count=attempt, reason="无需专业工具，直接生成回答"
                    )
                names = [tc["name"] for tc in resp.tool_calls]
                logs.append(f"Decision: {', '.join(names)}")
                return finish(
                    {"messages":[resp],"thought_log":logs},
                    retry_count=attempt,
                    reason=f"路由到专业工具：{', '.join(names)}",
                    selected=self.model_name
                )
            except Exception as e:
                if attempt < 2: logs.append(f"Retry {attempt+1}..."); time.sleep(2)
                else:
                    wrapped = ModelInvocationError("Router model invocation failed", details=str(e))
                    logger.exception("Router model invocation failed after retries")
                    return finish(
                        {"messages":[], "final_response":public_error_message(wrapped),
                         "next_step":"error", "thought_log":logs},
                        status="failed", retry_count=attempt, reason="路由模型调用失败",
                        error=wrapped
                    )

    def _tool_execution_node(self, state: MasterState):
        started_at = time.perf_counter()
        msgs = state.get("messages",[]) or []
        last = msgs[-1] if msgs else None
        logs = []
        updates = {"final_response":"Failed.","next_step":"finish","thought_log":logs}

        def finish(*, status: str, model: str = "", error: Optional[Exception] = None) -> dict:
            updates["node_statuses"] = {
                "tool_executor": _node_status(started_at, status, model=model, error=error)
            }
            if error:
                updates["errors"] = [{
                    "node": "tool_executor",
                    "code": error_code(error),
                    "message": public_error_message(error),
                }]
            return updates

        if not (isinstance(last, AIMessage) and last.tool_calls):
            error = AgentExecutionError("No executable tool call was found")
            updates["final_response"] = public_error_message(error)
            return finish(status="failed", error=error)

        tc = last.tool_calls[0]
        tname, targs = tc["name"], dict(tc.get("args",{}) or {})

        # Inject state parameters
        if tname in ["extract_text_info_tool","build_kg_tool"]:
            ft = state.get("input_text","")
            if ft.strip(): targs["text"] = ft
        if tname == "analyze_image_tool":
            rip = state.get("image_path","")
            if rip and os.path.exists(rip): targs["image_path"] = rip
            targs["workspace_dir"] = state.get("workspace_dir", "")
        if tname == "auto_borehole_pipeline":
            fps = state.get("file_path","") or ""
            shp = tif = csv = ""
            for p in fps.split("|"):
                pl = p.lower()
                if pl.endswith('.shp'): shp = p
                elif pl.endswith(('.tif','.tiff')): tif = p
                elif pl.endswith('.csv'): csv = p
            targs.update({"text_content":state.get("input_text",""),"shp_path":shp,
                          "tif_path":tif,"csv_path":csv,"instruction":state.get("user_instruction",""),
                          "workspace_dir":state.get("workspace_dir", ""), "extraction_bundle": None})
            rip = state.get("image_path","")
            if rip and os.path.exists(rip): targs["image_path"] = rip

        tool = self.tools_map.get(tname)
        if not tool:
            error = AgentExecutionError(f"Unknown tool: {tname}")
            updates["final_response"] = public_error_message(error)
            return finish(status="failed", error=error)

        try:
            logs.append(f"Exec: {tname}")
            out = tool.invoke(targs) or {}
            dtp = out.get("type"); dc = out.get("data")
            reported_error = None

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
                if st not in {"Success", "PartialSuccess"}:
                    reported_error = AgentExecutionError(f"Borehole pipeline status: {st}")
            elif dtp == "error":
                reported_error = AgentExecutionError("Tool reported an error", details=dc)
                updates["final_response"] = public_error_message(reported_error)
            nested_error_code = (
                getattr(dc, "error_code", None)
                if dc is not None and not isinstance(dc, dict)
                else (dc or {}).get("error_code") if isinstance(dc, dict) else None
            )
            if nested_error_code:
                if reported_error is None:
                    reported_error = AgentExecutionError("Specialist agent reported an error", details=dc)
                reported_error.code = str(nested_error_code)
                updates["final_response"] = public_error_message(reported_error)
            confidence = _estimate_confidence(dc)
            if confidence is not None:
                updates["overall_confidence"] = confidence
                confidence_key = {
                    "text_info": "text_extraction_confidence",
                    "image": "image_extraction_confidence",
                    "boreholes": "fusion_confidence",
                }.get(dtp)
                if confidence_key:
                    updates[confidence_key] = confidence
            if dtp == "boreholes" and isinstance(dc, dict):
                components = dc.get("confidence_components") or {}
                if components.get("text") is not None:
                    updates["text_extraction_confidence"] = components["text"]
                if components.get("image") is not None:
                    updates["image_extraction_confidence"] = components["image"]
            updates["messages"] = [ToolMessage(
                content=updates["final_response"],
                tool_call_id=tc.get("id") or "tool_call"
            )]
            if reported_error:
                logs.append(f"Error: {error_code(reported_error)}")
                updates["thought_log"] = logs
                return finish(
                    status="failed", model=self.model_name, error=reported_error
                )
            logs.append("Done.")
        except Exception as e:
            wrapped = AgentExecutionError(f"Tool {tname} failed", details=str(e))
            logger.exception("Tool execution failed: %s", tname)
            updates["final_response"] = public_error_message(wrapped)
            logs.append(f"Error: {error_code(wrapped)}")
            updates["messages"] = [ToolMessage(
                content=updates["final_response"],
                tool_call_id=tc.get("id") or "tool_call"
            )]
            updates["thought_log"] = logs
            return finish(status="failed", model=self.model_name, error=wrapped)

        updates["thought_log"] = logs
        tool_model = "qwen-vl-max" if tname == "analyze_image_tool" else self.model_name
        return finish(status="success", model=tool_model)

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
        return wf.compile(checkpointer=self.checkpointer)

    # === ENTRY ===
    def run(self, text: str, instruction: str = "", file_path: str = "", image_path: str = "",
            chat_history: Optional[List[BaseMessage]] = None,
            session_id: Optional[Any] = None,
            workspace_dir: str = "", fixed_task_type: str = ""):
        configure_tracing()
        thread_id = str(session_id) if session_id is not None else "default"
        config = {"configurable": {"thread_id": thread_id}}
        workspace_dir = workspace_dir or create_run_dir(f"standalone-{thread_id}")

        previous_log_count = 0
        try:
            snapshot = self.graph.get_state(config)
            previous_log_count = len(snapshot.values.get("thought_log", []))
        except Exception:
            logger.debug("No previous checkpoint state found for thread %s", thread_id, exc_info=True)

        result = self.graph.invoke({
            "input_text":text,"file_path":file_path,"image_path":image_path,
            "fixed_task_type": fixed_task_type,
            "workspace_dir":workspace_dir,
            "user_instruction":instruction,"chat_history":chat_history or [],
            "next_step":"","final_text_info":None,"final_kg":None,
            "final_image_analysis":None,"final_boreholes":None,"final_response":"",
            "messages":[HumanMessage(content=instruction)] if instruction else [],
            "thought_log":[],
            "node_statuses":Overwrite({}), "task_routing_log":Overwrite([]),
            "errors":Overwrite([]), "overall_confidence":None,
            "text_extraction_confidence":None, "image_extraction_confidence":None,
            "fusion_confidence":None,
        }, config=config)

        result["turn_thought_log"] = result.get("thought_log", [])[previous_log_count:]

        # Keep durable checkpoints small: completed-run artifacts stay on disk/MySQL,
        # while the graph retains only bounded conversational state and logs.
        compacted_state = {
            "input_text": "",
            "file_path": "",
            "image_path": "",
            "workspace_dir": "",
            "chat_history": [],
            "final_text_info": None,
            "final_kg": None,
            "final_image_analysis": None,
            "final_boreholes": None,
        }
        if len(result.get("messages", [])) > MAX_CHECKPOINT_MESSAGES:
            compacted_state["messages"] = Overwrite(
                result["messages"][-MAX_CHECKPOINT_MESSAGES:]
            )
        if len(result.get("thought_log", [])) > MAX_CHECKPOINT_LOGS:
            compacted_state["thought_log"] = Overwrite(
                result["thought_log"][-MAX_CHECKPOINT_LOGS:]
            )
        self.graph.update_state(config, compacted_state)
        return result

    def reset_thread(self, session_id: Any):
        """Remove all persistent LangGraph checkpoints for one MySQL session."""
        if session_id is not None:
            self.checkpointer.delete_thread(str(session_id))
