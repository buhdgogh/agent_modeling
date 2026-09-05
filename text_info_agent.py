import os
import re
import json
import time
import concurrent.futures
from typing import List, Optional, TypedDict, Any
from pydantic import BaseModel, Field

from langchain_community.chat_models import ChatTongyi
from langchain_core.messages import SystemMessage, HumanMessage
from langgraph.graph import StateGraph, START, END

try:
    from langchain_text_splitters import RecursiveCharacterTextSplitter
except ImportError:
    from langchain.text_splitter import RecursiveCharacterTextSplitter


# ==========================================
# 数据结构定义 (Pydantic Schema)
# ==========================================
class StratumItem(BaseModel):
    formation: Optional[str] = Field(None, description="地层名称（如：大岩山岩组）")
    formation_code: Optional[str] = Field(None, description="地层代号（如：Pt1d）")
    formation_age_1: Optional[str] = Field(None, description="主年代")
    formation_age_code_1: Optional[str] = Field(None, description="主年代代号")
    formation_age_2: Optional[str] = Field(None, description="次级年代")
    formation_age_code_2: Optional[str] = Field(None, description="次级年代代号")
    province: Optional[str] = Field(None, description="省份")
    distribution: Optional[str] = Field(None, description="分布")
    # ❌ 移除了 creator 字段，专注核心物理特征约束
    rock_features: Optional[str] = Field(None, description="岩石特征概括（限30字内）")
    confidence: float = Field(0.0, ge=0.0, le=1.0, description="该地层条目的置信度")


class ProfileItem(BaseModel):
    name: Optional[str] = Field(None, description="剖面名称")
    formation: Optional[str] = Field(None, description="对应地层")
    thickness: Optional[str] = Field(None, description="厚度")
    location: Optional[str] = Field(None, description="位置")
    underlying_stratum: Optional[str] = Field(None, description="下伏层")
    overlying_stratum: Optional[str] = Field(None, description="上覆层")
    rock_combination: Optional[str] = Field(None, description="岩石组合概括（限30字内）")
    confidence: float = Field(0.0, ge=0.0, le=1.0, description="该剖面条目的置信度")


class LLMExtraction(BaseModel):
    strata: Optional[List[StratumItem]] = Field(default_factory=list)
    profiles: Optional[List[ProfileItem]] = Field(default_factory=list)


class TextInfoResult(BaseModel):
    strata: List[Any] = Field(default_factory=list)
    profiles: List[Any] = Field(default_factory=list)
    source_text: str = ""


class AgentState(TypedDict):
    text: str
    instruction: str
    extraction: Optional[TextInfoResult]


# ==========================================
# 智能体定义
# ==========================================
class TextInfoAgent:
    def __init__(self, api_key: str, model_name: str = "qwen-max"):
        self.api_key = api_key
        self.model_name = model_name
        self.llm = ChatTongyi(model=model_name, dashscope_api_key=api_key, temperature=0.01, max_retries=3)

        # 探针：判断是否为 DeepSeek 模型，决定是否使用双轨制兜底
        self.is_deepseek = "deepseek" in model_name.lower()

        if not self.is_deepseek:
            # Qwen 等原生支持结构化的模型
            self.structural_llm = self.llm.with_structured_output(LLMExtraction)
        else:
            # DeepSeek 等偏推理模型，回退至纯文本 Prompt 解析
            self.structural_llm = self.llm

        # 设置动态重叠分块
        self.splitter = RecursiveCharacterTextSplitter(chunk_size=6000, chunk_overlap=300)
        self.graph = self._build_graph()

    def _extract_node(self, state: AgentState):
        full_text = state["text"]
        instruction = state.get("instruction", "")

        if not full_text or not full_text.strip():
            return {"extraction": TextInfoResult(strata=[], profiles=[], source_text="")}

        chunks = self.splitter.split_text(full_text)
        print(f">>> [TextInfoAgent] 文本长度 {len(full_text)}，优化切分为 {len(chunks)} 个大片段进行并发处理...")

        all_strata = []
        all_profiles = []

        system_msg = (
            "你是一个专业的文本信息抽取专家。\n"
            "指令：请逐字逐句阅读文本，准确提取【岩石地层】和【剖面】实体。\n"
            "【极端警告】：如果没有提及某个字段，必须返回 null。\n"
            "【防截断警告】：对于岩石特征等长文本描述，请进行高度浓缩概括（限制在30个字以内），严禁抄写原文大段描述！\n"
            "【置信度打分标准】：对每个提取条目，根据证据充分度给出 confidence (0.0-1.0)：\n"
            "  - 1.0: 文本明确提及，所有字段有直接依据\n"
            "  - 0.7-0.9: 大部分字段有明确依据，少量字段需推断\n"
            "  - 0.4-0.6: 部分字段需通过上下文间接推断\n"
            "  - 0.1-0.3: 仅模糊提及，高度不确定\n"
            "  - 0.0: 完全无证据\n"
        )
        if instruction:
            system_msg += f"\n特别要求: {instruction}"

        # 🚀 核心：明确定义 DeepSeek 需要遵守的 JSON 键名映射字典模板（无 creator）
        json_template = """
        请严格按照以下JSON结构和给定的【英文键名】输出，绝不能自己发明中文字段名！如果没有对应值请填 null：
        {
          "strata": [
            {
              "formation": "地层名称",
              "formation_code": "地层代号",
              "formation_age_1": "主年代",
              "formation_age_code_1": "主年代代号",
              "formation_age_2": "次级年代",
              "formation_age_code_2": "次级年代代号",
              "province": "省份",
              "distribution": "分布",
              "rock_features": "岩石特征概括",
              "confidence": 0.95
            }
          ],
          "profiles": [
            {
              "name": "剖面名称",
              "formation": "对应地层",
              "thickness": "厚度",
              "location": "位置",
              "underlying_stratum": "下伏层",
              "overlying_stratum": "上覆层",
              "rock_combination": "岩石组合概括",
              "confidence": 0.95
            }
          ]
        }
        """

        # 🚀 核心提速：使用线程池并发处理所有文本分块，打破串行阻塞
        def _process_single_chunk(chunk_data):
            i, chunk = chunk_data
            chunk_strata = []
            chunk_profiles = []
            max_retries = 2

            for attempt in range(max_retries):
                try:
                    if not self.is_deepseek:
                        # 正常的 Qwen 模型结构化解析逻辑
                        messages = [SystemMessage(content=system_msg), HumanMessage(content=f"文本片段：\n{chunk}")]
                        result = self.structural_llm.invoke(messages)
                        if result:
                            if result.strata: chunk_strata.extend(result.strata)
                            if result.profiles: chunk_profiles.extend(result.profiles)
                        break
                    else:
                        # 核心兜底：赋予 DeepSeek 严格的键名模板约束
                        fallback_sys = system_msg + "\n【格式严格要求】请直接输出纯JSON格式数据，绝不要输出Markdown标记或任何额外解释文字。\n" + json_template
                        raw_res = self.llm.invoke(
                            [SystemMessage(content=fallback_sys), HumanMessage(content=f"文本片段：\n{chunk}")])
                        content = raw_res.content

                        # 洗脱 DeepSeek 的思维链 (<think>...</think>)
                        content = re.sub(r'<think>.*?</think>', '', content, flags=re.DOTALL).strip()
                        content = re.sub(r'^```json|^```|```$', '', content, flags=re.IGNORECASE | re.MULTILINE).strip()

                        match = re.search(r"\{[\s\S]*\}", content)
                        if match:
                            data = json.loads(match.group())
                            result = LLMExtraction(**data)
                            if result:
                                if result.strata: chunk_strata.extend(result.strata)
                                if result.profiles: chunk_profiles.extend(result.profiles)
                            break
                        else:
                            raise ValueError("JSON not found in response.")

                except Exception as e:
                    err_msg = str(e)
                    if len(err_msg) > 100: err_msg = err_msg[:100] + "...(解析失败)"
                    print(f"    - 片段 {i + 1} 第 {attempt + 1} 次解析异常: {err_msg}")
                    if attempt < max_retries - 1:
                        time.sleep(2)
                    else:
                        print(f"    - 片段 {i + 1} 连续失败，跳过。")

            return chunk_strata, chunk_profiles

        # 启动并发线程加速抽取 (max_workers=5 提升大规模文本处理速度)
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            futures = [executor.submit(_process_single_chunk, (i, chunk)) for i, chunk in enumerate(chunks)]
            for future in concurrent.futures.as_completed(futures):
                s_res, p_res = future.result()
                all_strata.extend(s_res)
                all_profiles.extend(p_res)

        print(f">>> [TextInfoAgent] 处理完成，共并发提取 {len(all_strata)} 个地层信息，{len(all_profiles)} 个剖面信息。")

        # 将 Pydantic 对象统一转换为字典，以支持 JSON 序列化存储与前端展示
        final_strata = [s.model_dump() if hasattr(s, 'model_dump') else s.dict() for s in all_strata]
        final_profiles = [p.model_dump() if hasattr(p, 'model_dump') else p.dict() for p in all_profiles]

        return {"extraction": TextInfoResult(strata=final_strata, profiles=final_profiles, source_text=full_text)}

    def _build_graph(self):
        workflow = StateGraph(AgentState)
        workflow.add_node("extract_text", self._extract_node)
        workflow.add_edge(START, "extract_text")
        workflow.add_edge("extract_text", END)
        return workflow.compile()

    def run(self, text: str, instruction: str = ""):
        return self.graph.invoke({"text": text, "instruction": instruction})
