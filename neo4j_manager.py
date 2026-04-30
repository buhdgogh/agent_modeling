import os
from neo4j import GraphDatabase
from dotenv import load_dotenv
from typing import List, Dict, Any

# 加载配置
current_dir = os.path.dirname(os.path.abspath(__file__))
env_path = os.path.join(current_dir, "api_key.env")
load_dotenv(env_path, override=True)


class Neo4jManager:
    def __init__(self):
        self.uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")
        self.user = os.getenv("NEO4J_USER", "neo4j")
        self.password = os.getenv("NEO4J_PASSWORD", "")

        self.driver = None
        try:
            self.driver = GraphDatabase.driver(self.uri, auth=(self.user, self.password))
            # 验证连接
            self.driver.verify_connectivity()
            print(">>> [Neo4j] 连接成功！")
        except Exception as e:
            print(f">>> [Neo4j] 连接失败: {e}")
            self.driver = None

    def close(self):
        if self.driver:
            self.driver.close()

    def check_connection(self) -> bool:
        return self.driver is not None

    def save_graph_data(self, nodes: List[Any], edges: List[Any]):
        """
        将知识图谱数据存入 Neo4j
        :param nodes: List[Node] Pydantic对象
        :param edges: List[Edge] Pydantic对象
        """
        if not self.driver:
            print(">>> [Neo4j] 未连接，跳过保存。")
            return

        with self.driver.session() as session:
            # 1. 批量创建节点
            for node in nodes:
                # 使用 MERGE 避免重复创建 (基于 id)
                # 假设 node.type 是标签，node.id 是唯一标识
                # 注意：Neo4j 标签不能参数化，需要简单的字符串清洗
                label = self._clean_label(node.type)
                props = node.properties if node.properties else ""

                cypher = f"""
                MERGE (n:`{label}` {{name: $name}})
                SET n.description = $desc
                """
                try:
                    session.run(cypher, name=node.id, desc=props)
                except Exception as e:
                    print(f"Node Error: {e}")

            # 2. 批量创建关系
            for edge in edges:
                # 关系也使用 MERGE
                rel_type = self._clean_label(edge.relation)
                desc = edge.description if edge.description else ""

                # 查找两个节点并建立关系
                # 这里假设所有节点都有 name 属性
                cypher = f"""
                MATCH (a), (b)
                WHERE a.name = $source AND b.name = $target
                MERGE (a)-[r:`{rel_type}`]->(b)
                SET r.description = $desc
                """
                try:
                    session.run(cypher, source=edge.source, target=edge.target, desc=desc)
                except Exception as e:
                    print(f"Edge Error: {e}")

            print(f">>> [Neo4j] 数据写入完成: {len(nodes)} Nodes, {len(edges)} Edges.")

    def _clean_label(self, text: str) -> str:
        """简单的标签清洗，防止注入，保留字母数字下划线"""
        if not text: return "Entity"
        # 替换掉空格和特殊字符，Neo4j Label 最好是纯单词
        return "".join(c for c in text if c.isalnum() or c == "_")


# 单例模式测试
if __name__ == "__main__":
    nm = Neo4jManager()
    nm.close()