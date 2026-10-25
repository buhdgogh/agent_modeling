"""Application exception taxonomy and safe user-facing error conversion."""

from __future__ import annotations

from typing import Any


class AgentModelingError(Exception):
    code = "SYSTEM_ERROR"
    public_message = "系统处理失败，请稍后重试。"

    def __init__(self, message: str = "", *, details: Any = None):
        super().__init__(message or self.public_message)
        self.details = details


class ConfigurationError(AgentModelingError):
    code = "CONFIGURATION_ERROR"
    public_message = "系统配置不完整，请联系管理员检查服务配置。"


class DatabaseError(AgentModelingError):
    code = "DATABASE_ERROR"
    public_message = "数据库操作失败，请稍后重试。"


class ModelInvocationError(AgentModelingError):
    code = "MODEL_INVOCATION_ERROR"
    public_message = "模型服务暂时不可用，请稍后重试。"


class FileValidationError(AgentModelingError):
    code = "FILE_VALIDATION_ERROR"
    public_message = "输入文件无效或文件组合不完整，请检查后重试。"


class DocumentParsingError(AgentModelingError):
    code = "DOCUMENT_PARSING_ERROR"
    public_message = "文档解析失败，请检查文档、解析依赖和服务配置。"


class GeospatialError(AgentModelingError):
    code = "GEOSPATIAL_ERROR"
    public_message = "地理空间计算失败，请检查坐标系和输入数据。"


class AgentExecutionError(AgentModelingError):
    code = "AGENT_EXECUTION_ERROR"
    public_message = "智能体执行失败，请稍后重试。"


def error_code(error: Exception) -> str:
    return getattr(error, "code", "UNEXPECTED_ERROR")


def public_error_message(error: Exception) -> str:
    if isinstance(error, AgentModelingError):
        return f"{error.public_message}（错误码：{error.code}）"
    return "系统发生未预期异常，请稍后重试。（错误码：UNEXPECTED_ERROR）"
