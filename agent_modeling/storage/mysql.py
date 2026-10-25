import mysql.connector
import os
import json
import logging
from datetime import datetime
from agent_modeling.config import load_settings
from typing import List, Dict, Any

logger = logging.getLogger(__name__)


def _session_timestamp(value):
    """Return unknown legacy dates as None, preserving valid MySQL datetime values."""
    if value is None or isinstance(value, datetime):
        return value
    try:
        return datetime.fromisoformat(value)
    except (TypeError, ValueError):
        return None

# 强制使用绝对路径加载 .env
load_settings()


class MySQLSessionStore:
    def __init__(self):
        # 再次加载以防万一
        load_settings()
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
            logger.exception("MySQL connection failed")
            return None

    def check_connection(self) -> bool:
        try:
            conn = self.get_connection()
            if conn:
                conn.close()
                return True
            return False
        except Exception:
            logger.exception("MySQL connection health check failed")
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
            # Aggregate strings, not temporal values: legacy zero dates (and some
            # MySQL aggregate/coalesce conversions) raise warning 1292 in strict mode.
            # Test zero components as strings to avoid converting a zero-date literal.
            query = """
                SELECT s.id, s.title, CAST(s.created_at AS CHAR) AS created_at,
                       MAX(CASE
                           WHEN SUBSTRING(CAST(c.created_at AS CHAR), 1, 4) <> '0000'
                            AND SUBSTRING(CAST(c.created_at AS CHAR), 6, 2) <> '00'
                            AND SUBSTRING(CAST(c.created_at AS CHAR), 9, 2) <> '00'
                           THEN CAST(c.created_at AS CHAR)
                           ELSE NULL
                       END) AS last_message_at
                FROM sessions s
                LEFT JOIN chat_history c ON s.id = c.session_id
                GROUP BY s.id, s.title, s.created_at
            """
            cursor.execute(query)
            sessions = cursor.fetchall()
            for session in sessions:
                session['created_at'] = _session_timestamp(session['created_at'])
                session['last_active'] = (_session_timestamp(session.pop('last_message_at'))
                                          or session['created_at'])
            # Unknown timestamps sort last, without inventing a creation time.
            sessions.sort(key=lambda item: (item['last_active'] or datetime.min, item['id']), reverse=True)
            return sessions
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

    def clear_session_history(self, session_id: int, reset_title: bool = True) -> bool:
        """Delete all persisted messages for a session while keeping the session itself."""
        conn = self.get_connection()
        if not conn:
            return False
        cursor = conn.cursor()
        try:
            cursor.execute("DELETE FROM chat_history WHERE session_id = %s", (session_id,))
            if reset_title:
                cursor.execute(
                    "UPDATE sessions SET title = %s WHERE id = %s",
                    ("新对话", session_id)
                )
            conn.commit()
            return True
        except Exception as e:
            conn.rollback()
            logger.exception("Failed to clear session history: session_id=%s", session_id)
            return False
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
            logger.exception("Failed to delete empty sessions")
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
            import math
            def pydantic_encoder(obj):
                if hasattr(obj, "model_dump"): return obj.model_dump()
                if hasattr(obj, "dict"): return obj.dict()
                if isinstance(obj, float) and (math.isnan(obj) or math.isinf(obj)): return None
                return str(obj)

            try:
                keys_to_keep = [
                    'final_text_info', 'final_kg', 'final_image_analysis', 'final_boreholes',
                    'thought_log', 'node_statuses', 'task_routing_log', 'errors',
                    'overall_confidence', 'text_extraction_confidence',
                    'image_extraction_confidence', 'fusion_confidence',
                    'task_record', 'issues', 'evidence', 'artifacts', 'call_events', 'execution_metrics',
                    'image_path', 'source_files', 'pending_confirmations', 'confirmation_history'
                ]
                filtered = {k: v for k, v in result_state.items() if k in keys_to_keep and v is not None}
                if filtered:
                    json_state = json.dumps(filtered, default=pydantic_encoder, ensure_ascii=False,
                                           allow_nan=False)
            except Exception as e:
                logger.exception("Failed to serialize message result state")

        try:
            query = """
                INSERT INTO chat_history 
                (session_id, role, content, decision, result_state, file_path, file_type) 
                VALUES (%s, %s, %s, %s, %s, %s, %s)
            """
            cursor.execute(query, (session_id, role, content, decision, json_state, file_path, file_type))
            conn.commit()
        except Exception as e:
            logger.exception("Failed to persist chat message: session_id=%s role=%s", session_id, role)
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
                    except (TypeError, ValueError, json.JSONDecodeError):
                        logger.warning("Ignored invalid result_state JSON for message_id=%s", row['id'])

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
