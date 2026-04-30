import os
import time
import io
import base64
import re
import json
import networkx as nx
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'Microsoft YaHei', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False

from typing import List, Optional, TypedDict
from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.chat_models import ChatTongyi
from langchain_core.messages import SystemMessage, HumanMessage
from langgraph.graph import StateGraph, START, END
import concurrent.futures  # === 🚀 新增：导入多线程支持 ===

try:
    from neo4j_manager import Neo4jManager

    HAS_NEO4J = True
except ImportError:
    HAS_NEO4J = False

try:
    from langchain_text_splitters import RecursiveCharacterTextSplitter
except ImportError:
    from langchain.text_splitter import RecursiveCharacterTextSplitter


class Node(BaseModel):
    id: str = Field(description="实体名称")
    type: str = Field(description="实体类型")
    properties: Optional[str] = Field(description="属性描述")


class Edge(BaseModel):
    source: str = Field(description="源节点名称")
    target: str = Field(description="目标节点名称")
    relation: str = Field(description="关系类型")
    description: Optional[str] = Field(description="详细描述")


class KnowledgeGraphData(BaseModel):
    nodes: List[Node]
    edges: List[Edge]
    kg_image_base64: Optional[str] = Field(None, description="Neo4j不可用时的静态预览图")
    source_text: Optional[str] = Field(None, description="来源原始文本，用于溯源显示")


class AgentState(TypedDict):
    text: str
    kg_data: Optional[KnowledgeGraphData]


class KGBuilderAgent:
    def __init__(self, api_key: str, model_name: str = "qwen-max", user_prompt: str = ""):
        self.user_prompt = user_prompt
        self.is_deepseek = "deepseek" in model_name.lower()

        self.llm = ChatTongyi(
            model=model_name,
            dashscope_api_key=api_key,
            temperature=0,
        )

        if not self.is_deepseek:
            self.structural_llm = self.llm.with_structured_output(KnowledgeGraphData)
        else:
            self.structural_llm = None

        self.splitter = RecursiveCharacterTextSplitter(chunk_size=1200, chunk_overlap=150)
        self.graph = self._build_graph()

    def _generate_static_graph(self, nodes: List[Node], edges: List[Edge]) -> Optional[str]:
        try:
            G = nx.DiGraph()
            for n in nodes:
                label = n.id if len(n.id) < 10 else n.id[:8] + ".."
                G.add_node(n.id, label=f"{label}\n({n.type})")
            for e in edges:
                G.add_edge(e.source, e.target, label=e.relation)
            if G.number_of_nodes() == 0: return None

            plt.figure(figsize=(12, 9))
            pos = nx.spring_layout(G, k=1.5, seed=42)
            nx.draw_networkx_nodes(G, pos, node_color='lightblue', node_size=2500, alpha=0.9)
            node_labels = nx.get_node_attributes(G, 'label')
            nx.draw_networkx_labels(G, pos, labels=node_labels, font_size=10, font_family='sans-serif')
            nx.draw_networkx_edges(G, pos, edge_color='gray', arrows=True, arrowstyle='->', arrowsize=20)
            edge_labels = nx.get_edge_attributes(G, 'label')
            nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=9, font_color='red')
            plt.axis('off')

            buf = io.BytesIO()
            plt.savefig(buf, format='png', bbox_inches='tight', dpi=150)
            plt.close()
            return base64.b64encode(buf.getvalue()).decode('utf-8')
        except Exception as e:
            print(f">>> [KGBuilderAgent] 生成静态图失败: {e}")
            return None

    def _extraction_node(self, state: AgentState):
        full_text = state["text"]
        if not full_text or not full_text.strip():
            return {"kg_data": KnowledgeGraphData(nodes=[], edges=[])}

        chunks = self.splitter.split_text(full_text)
        print(f">>> [KGBuilderAgent] 文本切分为 {len(chunks)} 个片段...")

        all_nodes = []
        all_edges = []

        system_msg = (
            "你是一个严谨的知识图谱构建专家。\n"
            "任务：仔细阅读文本，将其转化为高质量的图谱节点和边。请尽最大努力提取所有相关信息，绝不要遗漏。\n\n"
            "【强约束：实体分类标准】（请将提取的实体尽可能归入以下类型）：\n"
            "- 岩石与地层（如：组、段、石灰岩、玄武岩）\n"
            "- 地质年代与时间（如：二叠纪、中生代）\n"
            "- 地质构造与现象（如：断层、褶皱、不整合面）\n"
            "- 地理位置与区域（如：四川盆地、剖面位置）\n"
            "- 其他（如不属上述类别可自由定义）\n\n"
            "【强约束：关系分类标准】（请参考以下关系进行提取）：\n"
            "- 组成/岩性（如：某地层-组成-某岩石）\n"
            "- 属于/时代（如：某地层-属于-某时代）\n"
            "- 空间/分布（如：某实体-分布于-某地点）\n"
            "- 上覆/下伏（如：地层A-上覆于-地层B）\n"
            "- 导致/成因（如：某运动-导致-某构造）\n"
        )
        if self.user_prompt: system_msg += f"\n特别关注: {self.user_prompt}"

        # === 🚀 新增：明确定义 DeepSeek 需要遵守的 JSON 键名映射字典模板 ===
        json_template = """
        请严格按照以下JSON格式和【英文键名】输出，绝不能改变字段名或发明中文字段名！
        {
          "nodes": [
            {"id": "实体名称", "type": "实体分类", "properties": "属性描述(若无填null)"}
          ],
          "edges": [
            {"source": "源节点名称", "target": "目标节点名称", "relation": "关系类型", "description": "详细描述(若无填null)"}
          ]
        }
        """

        # === 🚀 核心提速 3：使用线程池并发处理图谱构建，大幅缩短耗时 ===
        def _process_single_chunk(chunk_data):
            i, chunk = chunk_data
            chunk_nodes = []
            chunk_edges = []
            for attempt in range(3):
                try:
                    if not self.is_deepseek:
                        prompt = ChatPromptTemplate.from_messages(
                            [("system", system_msg), ("human", "文本片段：\n{text}")])
                        chain = prompt | self.structural_llm
                        result = chain.invoke({"text": chunk})
                        if result:
                            if result.nodes: chunk_nodes.extend(result.nodes)
                            if result.edges: chunk_edges.extend(result.edges)
                        break
                    else:
                        # 核心兜底：赋予 DeepSeek 严格的键名模板约束
                        fallback_sys = system_msg + "\n【格式要求】请严格输出纯JSON格式数据，包含 nodes 和 edges 列表，绝不能输出Markdown格式或其他解释性废话。\n" + json_template
                        raw_res = self.llm.invoke(
                            [SystemMessage(content=fallback_sys), HumanMessage(content=f"文本片段：\n{chunk}")])
                        content = raw_res.content

                        # 洗脱 DeepSeek-R1 的思维链
                        content = re.sub(r'<think>.*?</think>', '', content, flags=re.DOTALL).strip()
                        content = re.sub(r'^```json|^```|```$', '', content, flags=re.IGNORECASE | re.MULTILINE).strip()

                        match = re.search(r"\{[\s\S]*\}", content)
                        if match:
                            data = json.loads(match.group())
                            result = KnowledgeGraphData(**data)
                            if result:
                                if result.nodes: chunk_nodes.extend(result.nodes)
                                if result.edges: chunk_edges.extend(result.edges)
                            break
                        else:
                            raise ValueError("JSON not found.")

                except Exception as e:
                    print(f"    片段 {i + 1} 失败 (尝试 {attempt + 1}): {str(e)[:50]}")
                    time.sleep(2)

            return chunk_nodes, chunk_edges

        # 启动并发线程加速图谱节点提取
        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
            futures = [executor.submit(_process_single_chunk, (i, chunk)) for i, chunk in enumerate(chunks)]
            for future in concurrent.futures.as_completed(futures):
                n_res, e_res = future.result()
                all_nodes.extend(n_res)
                all_edges.extend(e_res)

        unique_nodes_dict = {n.id: n for n in all_nodes}
        unique_nodes = list(unique_nodes_dict.values())

        print(f">>> [KGBuilderAgent] 抽取完成: {len(unique_nodes)} Nodes, {len(all_edges)} Edges.")

        if HAS_NEO4J:
            try:
                neo4j_mgr = Neo4jManager()
                if neo4j_mgr.check_connection():
                    neo4j_mgr.save_graph_data(unique_nodes, all_edges)
                    print(">>> [KGBuilderAgent] Neo4j 同步成功。")
                    neo4j_mgr.close()
            except Exception as e:
                print(f">>> [KGBuilderAgent] Neo4j 操作异常: {e}")

        print(">>> [KGBuilderAgent] 生成静态截面图用于前端展示...")
        kg_image = self._generate_static_graph(unique_nodes, all_edges)

        return {"kg_data": KnowledgeGraphData(nodes=unique_nodes, edges=all_edges, kg_image_base64=kg_image,
                                              source_text=full_text)}

    def _build_graph(self):
        workflow = StateGraph(AgentState)
        workflow.add_node("extract_kg", self._extraction_node)
        workflow.add_edge(START, "extract_kg")
        workflow.add_edge("extract_kg", END)
        return workflow.compile()

    def run(self, text: str):
        return self.graph.invoke({"text": text})