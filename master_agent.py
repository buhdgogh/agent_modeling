import os
import re
import traceback
import time
from typing import TypedDict, Optional, Any, List
from langchain_community.chat_models import ChatTongyi
from langgraph.graph import StateGraph, START, END
from langchain_core.tools import tool
from langchain_core.messages import AIMessage, SystemMessage, HumanMessage, BaseMessage

try:
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    HAS_SPLITTER = True
except ImportError:
    try:
        from langchain.text_splitter import RecursiveCharacterTextSplitter

        HAS_SPLITTER = True
    except ImportError:
        HAS_SPLITTER = False

try:
    from text_info_agent import TextInfoAgent, TextInfoResult
    from kg_builder import KGBuilderAgent, KnowledgeGraphData
    from image_agent import ImageAgent, ImageAnalysisResult
    from gempy_agent import GempyAgent
except ImportError:
    pass

import concurrent.futures


class MasterState(TypedDict):
    input_text: str
    file_path: str
    image_path: str
    user_instruction: str
    chat_history: List[BaseMessage]
    next_step: str
    final_text_info: Optional[Any]
    final_kg: Optional[Any]
    final_image_analysis: Optional[Any]
    final_boreholes: Optional[Any]
    final_response: str
    thought_log: List[str]
    messages: List[Any]


class MasterAgent:
    def __init__(self, api_key: str, model_name: str = "qwen-max"):
        self.model_name = model_name
        self.api_key = api_key
        if not api_key: raise ValueError("API Key is required.")

        # 将用户选择的大模型传递给执行重活的下游 Agent
        self.text_info_bot = TextInfoAgent(api_key=api_key, model_name=model_name)
        self.kg_bot = KGBuilderAgent(api_key=api_key, model_name=model_name)
        self.image_bot = ImageAgent(api_key=api_key, model_name="qwen-vl-max")
        self.borehole_bot = GempyAgent(api_key=api_key, model_name=model_name)

        @tool("extract_text_info_tool")
        def extract_text_info_tool(text: str) -> dict:
            """从文本中提取岩石地层信息、剖面等关键文本信息。"""
            return {"type": "text_info", "data": self.text_info_bot.run(text).get("extraction")}

        @tool("build_kg_tool")
        def build_kg_tool(text: str) -> dict:
            """根据文本提取结果构建知识图谱，并将其自动存入后台的 Neo4j 数据库。"""
            return {"type": "kg", "data": self.kg_bot.run(text).get("kg_data")}

        @tool("analyze_image_tool")
        def analyze_image_tool(instruction: str, image_path: str = "") -> dict:
            """分析地质图件、图片内容，提取视觉特征和地质信息。"""
            if not image_path or not os.path.exists(image_path):
                return {"type": "error", "data": f"未找到图片文件: {image_path}"}
            return {"type": "image", "data": self.image_bot.run(image_path, instruction).get("analysis")}

        @tool("auto_borehole_pipeline")
        def auto_borehole_pipeline(instruction: str, text_content: str = "", image_path: str = "", shp_path: str = "",
                                   tif_path: str = "", csv_path: str = "") -> dict:
            """全自动虚拟钻孔工作流：根据图件与文本数据，全自动融合生成严格的地质特征CSV，最后进行空间解算。"""
            try:
                final_csv_path = csv_path

                if not final_csv_path or not os.path.exists(final_csv_path):

                    def _get_image_legends():
                        if image_path and os.path.exists(image_path):
                            img_res = self.image_bot.run(image_path, "提取图例顺序与地层代号")
                            return str(img_res.get("analysis", ""))
                        return "无图件数据"

                    def _get_text_entities():
                        if text_content and text_content.strip():
                            ent_res = self.text_info_bot.run(text_content)
                            return str(ent_res.get("extraction", ""))
                        return "无文本数据"

                    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
                        future_img = executor.submit(_get_image_legends)
                        future_txt = executor.submit(_get_text_entities)
                        legends_data = future_img.result()
                        entities_data = future_txt.result()

                    legends_safe = legends_data.replace("{", "[").replace("}", "]")
                    entities_safe = entities_data.replace("{", "[").replace("}", "]")

                    fusion_prompt = (
                        "你是一个地质数据对齐专家。请结合以下图例信息和文本剖面信息，生成一份标准的虚拟钻孔地层特征CSV文件。\n"
                        "必须严格包含且仅包含以下8列，表头顺序绝不能错：\n"
                        "part_code,formation,formation_code,formation_age_1,formation_age_code_1,formation_age_2,formation_age_code_2,厚度\n\n"
                        "填写强规则:\n"
                        "- part_code: 默认固定填写 part_1\n"
                        "- formation: 地层名称（如 第四系, Q, 玄武岩 等）\n"
                        "- formation_code: 地层数字序号（从上到下/新到老填 1, 2, 3... 必须为整数）\n"
                        "- formation_age_1, formation_age_code_1, formation_age_2, formation_age_code_2: 直接使用文本提取结果中的对应字段，如果没有则填 null。\n"
                        "- 厚度: 提取到的地层厚度纯数字（单位: 米）。如果没有找到确切厚度，请默认填 10。\n\n"
                        f"图例序列信息: {legends_safe[:1500]}\n"
                        f"文本厚度及年代提取结果: {entities_safe[:3000]}\n\n"
                        "最终指令：请直接输出纯文本的 CSV 数据（包含表头），绝不要包含 ```csv 等任何 Markdown 代码块标识，也不要有任何其它多余解释文字！"
                    )

                    clean_llm = ChatTongyi(model=self.model_name, dashscope_api_key=self.api_key, temperature=0.01,
                                           max_tokens=3000)
                    raw_content = clean_llm.invoke([HumanMessage(content=fusion_prompt)]).content

                    if "deepseek" in self.model_name.lower():
                        raw_content = re.sub(r"<think>.*?</think>", "", raw_content, flags=re.DOTALL).strip()

                    header_pattern = r"(part_code\s*,\s*formation\s*,\s*formation_code\s*,\s*formation_age_1\s*,\s*formation_age_code_1\s*,\s*formation_age_2\s*,\s*formation_age_code_2\s*,\s*厚度.*)"
                    match = re.search(header_pattern, raw_content, re.IGNORECASE | re.DOTALL)
                    if match:
                        csv_content = match.group(1).strip()
                        csv_content = re.sub(r"\n\s*```.*$", "", csv_content, flags=re.DOTALL | re.IGNORECASE).strip()
                    else:
                        csv_content = raw_content.replace("```csv", "").replace("```", "").strip()

                    temp_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "temp")
                    if not os.path.exists(temp_dir): os.makedirs(temp_dir)
                    final_csv_path = os.path.join(temp_dir, "auto_generated_formation.csv")
                    with open(final_csv_path, "w", encoding="utf-8-sig") as f:
                        f.write(csv_content)

                file_paths = f"{shp_path}|{tif_path}|{final_csv_path}"
                res = self.borehole_bot.run(text="", file_paths=file_paths)
                return {"type": "boreholes", "data": res.get("borehole_result")}
            except Exception as e:
                return {"type": "boreholes", "data": {"status": f"Pipeline Error: {str(e)}"}}

        self.tools = [extract_text_info_tool, build_kg_tool, analyze_image_tool, auto_borehole_pipeline]
        self.tools_map = {t.name: t for t in self.tools}

        # === 🚀为 Router 模型加入内置 max_retries 并强行限制其上下文复杂度 ===
        self.router_llm = ChatTongyi(model="qwen-max", dashscope_api_key=api_key, temperature=0, max_tokens=1000,
                                     max_retries=3).bind_tools(self.tools)

        if HAS_SPLITTER: self.splitter = RecursiveCharacterTextSplitter(chunk_size=15000, chunk_overlap=200)
        self.graph = self._build_graph()

    def _decider_node(self, state: MasterState):
        instruction, text, file_path, image_path = state.get("user_instruction", ""), state.get("input_text",
                                                                                                ""), state.get(
            "file_path", ""), state.get("image_path", "")
        logs = ["🤖 **主控系统启动**：使用 qwen-max 解析动作意图...", f"📝 **指令**: {instruction[:50]}..."]
        if not instruction.strip(): return {"messages": [], "final_response": "请输入指令。",
                                            "next_step": "general_chat", "thought_log": logs}

        system_prompt = (
            "你是一个地质智能技术中枢。\n"
            "判断逻辑：\n"
            "1. **全自动虚拟钻孔**：如果指令涉及“虚拟钻孔”、“提取钻孔”，或者同时上传了图件、文档、SHP数据要求建模，必须调用 `auto_borehole_pipeline`。\n"
            "2. **图件理解**：如果有 image_path 且指令只要求分析图件，调用 `analyze_image_tool`。\n"
            "3. **文本信息处理**：涉及文本实体与地层信息的抽取，或者要求构建知识图谱存入数据库时，调用对应工具。\n"
            "4. **通用回答**：其他情况直接回答。"
        )

        messages = [SystemMessage(content=system_prompt)] + state.get("chat_history", [])
        content_parts = []
        if file_path: logs.append("📂 检测到文件上传"); content_parts.append(f"【上传文件路径】: {file_path}")
        if image_path: logs.append("🖼️ 检测到图片"); content_parts.append(f"【图片路径】: {image_path}")

        # === 🚀 大幅削减喂给 Router 的背景文本长度 ===
        # 原理：Router 只需要知道有文本上传，以及用文本的前几句判断领域意图。
        # 给带 Tool Calling 的模型发送 2000 字长文极易引发注意力机制死锁和 API 300秒超时。
        if text.strip():
            content_parts.append(
                f"【背景文本提示】:\n(系统已加载总长为 {len(text)} 字的文档。为免大模型超时死锁，主控节点仅截取前200字预览判定意图。后续具体抽取由子智能体全量负责):\n\n{text[:200]}...")
            logs.append(f"📄 检测到背景文本 (长度: {len(text)} 字)")

        content_parts.append(f"【用户指令】: {instruction}")
        messages.append(HumanMessage(content="\n\n".join(content_parts)))

        max_retries = 3
        for attempt in range(max_retries):
            try:
                response = self.router_llm.invoke(messages)
                if not response.tool_calls:
                    logs.append("🧠 **决策**: 通用问答模式。")
                    return {"messages": [response], "final_response": response.content or "（无文本回复）",
                            "next_step": "general_chat", "thought_log": logs}

                tool_name = response.tool_calls[0]["name"]
                logs.append(f"🧠 **决策**: 路由下发至工具 `{tool_name}`。")
                return {"messages": [response], "thought_log": logs}
            except Exception as e:
                err_str = str(e)
                if attempt < max_retries - 1:
                    logs.append(f"⚠️ 路由网关响应超时或波动，正在重发请求 ({attempt + 1}/{max_retries})...")
                    time.sleep(2)
                else:
                    logs.append(f"❌ 决策节点发生致命超时: {err_str}")
                    return {"messages": [],
                            "final_response": f"API 网关响应超时 (Read Timeout 300s)，请检查网络状态或稍后再试: {err_str}",
                            "next_step": "error", "thought_log": logs}

    def _tool_execution_node(self, state: MasterState):
        messages = state.get("messages", [])
        last_message = messages[-1] if messages else None
        logs = list(state.get("thought_log", []))

        updates = {
            "final_text_info": None, "final_kg": None, "final_image_analysis": None, "final_boreholes": None,
            "final_response": "任务处理失败。", "next_step": "finish", "thought_log": logs
        }

        if isinstance(last_message, AIMessage) and last_message.tool_calls:
            tool_call = last_message.tool_calls[0]
            tool_name, tool_args = tool_call["name"], tool_call["args"]

            if tool_name in ["extract_text_info_tool", "build_kg_tool"]:
                full_text = state.get("input_text", "")
                if full_text.strip():
                    tool_args["text"] = full_text
                elif "text" not in tool_args or not tool_args["text"].strip():
                    tool_args["text"] = state.get("user_instruction", "")

            if tool_name == "analyze_image_tool":
                real_image_path = state.get("image_path", "")
                if real_image_path and os.path.exists(real_image_path):
                    tool_args["image_path"] = real_image_path
                elif "image_path" not in tool_args:
                    tool_args["image_path"] = ""

            if tool_name == "auto_borehole_pipeline":
                file_path_str = state.get("file_path", "")
                shp_p, tif_p, csv_p = "", "", ""
                if file_path_str:
                    for p in file_path_str.split("|"):
                        p_lower = p.lower()
                        if p_lower.endswith('.shp'):
                            shp_p = p
                        elif p_lower.endswith('.tif') or p_lower.endswith('.tiff'):
                            tif_p = p
                        elif p_lower.endswith('.csv'):
                            csv_p = p

                tool_args["text_content"] = state.get("input_text", "")
                real_image_path = state.get("image_path", "")
                if real_image_path and os.path.exists(real_image_path): tool_args["image_path"] = real_image_path
                tool_args["shp_path"] = shp_p
                tool_args["tif_path"] = tif_p
                tool_args["csv_path"] = csv_p
                tool_args["instruction"] = state.get("user_instruction", "")

            selected_tool = self.tools_map.get(tool_name)
            if selected_tool:
                try:
                    logs.append(f"⚙️ **执行工具**: `{tool_name}` (底层模型: {self.model_name})...")
                    if tool_name == "auto_borehole_pipeline":
                        logs.append("🔄 启动多模态协同管线：调度视觉与文本 Agent 并行工作...")
                        logs.append("🧠 LLM 裁判正在进行多源知识融合...")

                    tool_output = selected_tool.invoke(tool_args)
                    data_type, data_content = tool_output.get("type"), tool_output.get("data")

                    if data_type == "error":
                        updates.update({"final_response": f"❌ 任务出错中止: {data_content}", "next_step": "error"})
                    elif data_type == "text_info":
                        updates.update({"final_text_info": data_content, "next_step": "extract_text_info",
                                        "final_response": "文本信息抽取完成。"})
                    elif data_type == "kg":
                        updates.update(
                            {"final_kg": data_content, "next_step": "build_kg", "final_response": "图谱构建完成。"})
                    elif data_type == "image":
                        updates.update({"final_image_analysis": data_content, "next_step": "analyze_image",
                                        "final_response": "图件分析完成。"})
                    elif data_type == "boreholes":
                        status = data_content.get("status", "Unknown") if isinstance(data_content, dict) else "Unknown"
                        if status == "Success":
                            updates.update({"final_boreholes": data_content, "next_step": "generate_boreholes",
                                            "final_response": f"🎉 虚拟钻孔执行完毕！已生成标准CSV。"})
                        else:
                            updates.update({"final_boreholes": data_content, "next_step": "generate_boreholes",
                                            "final_response": f"⚠️ 计算出现问题: {status}"})
                    logs.append(f"✅ 执行完毕。")
                except Exception as e:
                    updates["final_response"] = f"工具错误: {str(e)}"
                    logs.append(f"❌ 错误: {str(e)}")

        updates["thought_log"] = logs
        return updates

    def _build_graph(self):
        workflow = StateGraph(MasterState)
        workflow.add_node("decider", self._decider_node)
        workflow.add_node("tool_executor", self._tool_execution_node)
        workflow.add_edge(START, "decider")

        def should_continue(state: MasterState):
            messages = state.get("messages", [])
            return "execute" if state.get("next_step") != "error" and messages and isinstance(messages[-1],
                                                                                              AIMessage) and messages[
                                    -1].tool_calls else "end"

        workflow.add_conditional_edges("decider", should_continue, {"execute": "tool_executor", "end": END})
        workflow.add_edge("tool_executor", END)
        return workflow.compile()

    def run(self, text: str, instruction: str = "", file_path: str = "", image_path: str = "",
            chat_history: List[BaseMessage] = []):
        return self.graph.invoke({
            "input_text": text, "file_path": file_path, "image_path": image_path,
            "user_instruction": instruction, "chat_history": chat_history,
            "messages": [], "thought_log": []
        })