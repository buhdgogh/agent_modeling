import mysql.connector
import os
import json
from dotenv import load_dotenv
from typing import List, Dict, Any

# 强制使用绝对路径加载 .env
current_dir = os.path.dirname(os.path.abspath(__file__))
env_path = os.path.join(current_dir, "api_key.env")
load_dotenv(env_path, override=True)


class DBManager:
    def __init__(self):
        # 再次加载以防万一
        load_dotenv(env_path, override=True)
        self.config = {
            'user': os.getenv("DB_USER", "root"),
            'password': os.getenv("DB_PASSWORD", ""),
            'host': os.getenv("DB_HOST", "127.0.0.1"),
            'port': os.getenv("DB_PORT", "3306"),
            'database': os.getenv("DB_NAME", "agent_chat_db"),
            'raise_on_warnings': True,
            'auth_plugin': 'mysql_native_password'  # 🚀 修复：强制指定认证插件，解决 2059 already loaded 报错
        }

    def get_connection(self):
        try:
            return mysql.connector.connect(**self.config)
        except mysql.connector.Error as err:
            print(f"[DB Error] {err}")
            return None

    def check_connection(self) -> bool:
        try:
            conn = self.get_connection()
            if conn:
                conn.close()
                return True
            return False
        except:
            return False

    # === 会话管理 ===
    def create_session(self, title: str = "新对话") -> int:
        conn = self.get_connection()
        if not conn: return -1
        cursor = conn.cursor()
        try:
            cursor.execute("INSERT INTO sessions (title) VALUES (%s)", (title,))
            conn.commit()
            return cursor.lastrowid
        finally:
            cursor.close()
            conn.close()

    def get_all_sessions(self) -> List[Dict]:
        conn = self.get_connection()
        if not conn: return []
        cursor = conn.cursor(dictionary=True)
        try:
            # === 按最后交互时间倒序排序 ===
            # 关联 chat_history 表，找出每条会话最近的一条消息时间
            # 如果是空会话(没有消息)，则 COALESCE 会退回使用会话的创建时间
            query = """
                SELECT s.id, s.title, s.created_at, 
                       COALESCE(MAX(c.created_at), s.created_at) as last_active
            FROM sessions s
            LEFT JOIN chat_history c ON s.id = c.session_id
            GROUP BY s.id, s.title, s.created_at
            ORDER BY last_active DESC
            """
            cursor.execute(query)
            return cursor.fetchall()
        finally:
            cursor.close()
            conn.close()

    def delete_session(self, session_id: int):
        conn = self.get_connection()
        if not conn: return
        cursor = conn.cursor()
        try:
            cursor.execute("DELETE FROM sessions WHERE id = %s", (session_id,))
            conn.commit()
        finally:
            cursor.close()
            conn.close()

    # === 一键清理闲置的空对话 ===
    def delete_empty_sessions(self, exclude_session_id: int = None):
        """静默删除所有没有任何聊天记录的会话，跳过用户当前正在选中的会话"""
        conn = self.get_connection()
        if not conn: return
        cursor = conn.cursor()
        try:
            if exclude_session_id is not None:
                query = """
                    DELETE FROM sessions 
                    WHERE id != %s 
                      AND id NOT IN (SELECT session_id FROM chat_history)
                """
                cursor.execute(query, (exclude_session_id,))
            else:
                query = """
                    DELETE FROM sessions 
                    WHERE id NOT IN (SELECT session_id FROM chat_history)
                """
                cursor.execute(query)
            conn.commit()
        except Exception as e:
            print(f"[Delete Empty Sessions Error] {e}")
        finally:
            cursor.close()
            conn.close()

    def update_session_title(self, session_id: int, title: str):
        conn = self.get_connection()
        if not conn: return
        cursor = conn.cursor()
        try:
            cursor.execute("UPDATE sessions SET title = %s WHERE id = %s", (title, session_id))
            conn.commit()
        finally:
            cursor.close()
            conn.close()

    # === 消息管理 ===
    def add_message(self, session_id: int, role: str, content: str,
                    decision: str = None, result_state: Dict = None,
                    file_path: str = None, file_type: str = None):
        conn = self.get_connection()
        if not conn: return
        cursor = conn.cursor()

        json_state = None
        if result_state:
            def pydantic_encoder(obj):
                if hasattr(obj, "model_dump"): return obj.model_dump()
                if hasattr(obj, "dict"): return obj.dict()
                return str(obj)

            try:
                # 重新加入 final_kg 字段用于后台数据持久化记录，但不在前端渲染
                keys_to_keep = ['final_text_info', 'final_kg', 'final_image_analysis', 'final_boreholes', 'thought_log']
                filtered = {k: v for k, v in result_state.items() if k in keys_to_keep and v is not None}
                if filtered:
                    json_state = json.dumps(filtered, default=pydantic_encoder, ensure_ascii=False)
            except Exception as e:
                print(f"[Serialization Error] {e}")

        try:
            query = """
                INSERT INTO chat_history 
                (session_id, role, content, decision, result_state, file_path, file_type) 
                VALUES (%s, %s, %s, %s, %s, %s, %s)
            """
            cursor.execute(query, (session_id, role, content, decision, json_state, file_path, file_type))
            conn.commit()
        except Exception as e:
            print(f"[Add Message Error] {e}")
        finally:
            cursor.close()
            conn.close()

    def get_history(self, session_id: int) -> List[Dict]:
        conn = self.get_connection()
        if not conn: return []
        cursor = conn.cursor(dictionary=True)
        try:
            query = "SELECT * FROM chat_history WHERE session_id = %s ORDER BY id ASC"
            cursor.execute(query, (session_id,))
            history = []
            for row in cursor.fetchall():
                res_state = None
                if row['result_state']:
                    try:
                        res_state = json.loads(row['result_state'])
                    except:
                        pass

                history.append({
                    "id": row['id'],
                    "role": row['role'],
                    "content": row['content'],
                    "decision": row['decision'],
                    "result_state": res_state,
                    "file_path": row.get('file_path'),
                    "file_type": row.get('file_type')
                })
            return history
        finally:
            cursor.close()
            conn.close()