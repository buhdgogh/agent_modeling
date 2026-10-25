"""Structured business records."""


from typing import Any, Dict, List, Optional, Literal
from pydantic import BaseModel, Field


class EvidenceRef(BaseModel):
    evidence_id: str
    source_kind: Literal["text", "image", "user", "configuration", "default"]
    document_id: str = ""
    block_id: str = ""
    source: Dict[str, Any] = Field(default_factory=dict)
    quote: str = ""
    entity_id: str = ""
    field: str = ""
    value: Any = None
    verified: bool = False


class IssueRecord(BaseModel):
    issue_id: str
    code: str
    message: str
    stage: str
    status: Literal["open", "resolved"] = "open"
    row_id: str = ""
    field: str = ""
    evidence_ids: List[str] = Field(default_factory=list)
    resolution: str = ""


class ArtifactRecord(BaseModel):
    artifact_id: str
    task_id: str
    version: int = Field(ge=1)
    kind: str
    path: str
    sha256: str
    evidence_ids: List[str] = Field(default_factory=list)


class TaskRecord(BaseModel):
    task_id: str
    session_id: str
    status: Literal["running", "awaiting_review", "completed", "failed", "cancelled"]
    version: int = Field(default=1, ge=1)
    resume_stage: str = ""
    issues: List[IssueRecord] = Field(default_factory=list)
    artifacts: List[ArtifactRecord] = Field(default_factory=list)
    revisions: List[dict] = Field(default_factory=list)


class SemanticFinding(BaseModel):
    model_config = {"extra": "forbid"}
    row_id: str
    field: str
    message: str = Field(min_length=1, max_length=2000)
    evidence_ids: List[str]


class ConfirmationDecision(BaseModel):
    model_config = {"extra": "forbid"}
    question_id: str
    action: Literal["accept_default", "set_thickness", "select_formation", "defer"]
    thickness: Optional[float] = Field(default=None, gt=0, allow_inf_nan=False)
    entity_id: str = ""


class SemanticReviewResult(BaseModel):
    model_config = {"extra": "forbid"}
    status: Literal["passed", "repair", "needs_review"]
    summary: str = Field(min_length=1, max_length=4000)
    findings: List[SemanticFinding] = Field(max_length=100)


class CallEvent(BaseModel):
    event_id: str
    call_id: str
    task_id: str
    version: int
    parent_call_id: str = ""
    operation_id: str = ""
    kind: Literal["node", "model", "tool", "run"]
    name: str
    status: Literal["started", "success", "failed", "cancelled", "interrupted"]
    timestamp: str
    duration_ms: float = 0
    model: str = ""
    attempt: int = Field(default=1, ge=1)
    input_tokens: Optional[int] = None
    output_tokens: Optional[int] = None
    cost: Optional[float] = None
    input_price_per_million: Optional[float] = None
    output_price_per_million: Optional[float] = None
    currency: str = "CNY"
    error_code: str = ""


class ExtractedStratum(BaseModel):
    """Model-facing business fields only; provenance is assigned by the server."""
    model_config = {"extra": "ignore"}
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


class StratumItem(ExtractedStratum):
    entity_id: str = ""
    evidence: Dict[str, List[dict]] = Field(default_factory=dict)


class ExtractedProfile(BaseModel):
    """Preserve reported thickness qualifiers rather than inferring an exact value."""
    model_config = {"extra": "ignore"}
    name: Optional[str] = Field(None, description="剖面名称")
    formation: Optional[str] = Field(None, description="对应地层")
    thickness: Optional[str] = Field(None, description="厚度")
    location: Optional[str] = Field(None, description="位置")
    underlying_stratum: Optional[str] = Field(None, description="下伏层")
    overlying_stratum: Optional[str] = Field(None, description="上覆层")
    rock_combination: Optional[str] = Field(None, description="岩石组合概括（限30字内）")
    confidence: float = Field(0.0, ge=0.0, le=1.0, description="该剖面条目的置信度")


class ProfileItem(ExtractedProfile):
    entity_id: str = ""
    evidence: Dict[str, List[dict]] = Field(default_factory=dict)


class LLMExtraction(BaseModel):
    strata: Optional[List[ExtractedStratum]] = Field(default_factory=list)
    profiles: Optional[List[ExtractedProfile]] = Field(default_factory=list)


class TextInfoResult(BaseModel):
    strata: List[Any] = Field(default_factory=list)
    profiles: List[Any] = Field(default_factory=list)
    source_text: str = ""
    tool_trace: List[dict] = Field(default_factory=list)


from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class GeoSymbol(BaseModel):
    base: Optional[str] = Field(None)
    superscript: Optional[str] = Field(None)
    subscript: Optional[str] = Field(None)
    final: Optional[str] = Field("Unknown", alias="symbol_final")
    terminal: Optional[str] = Field("Unknown", alias="symbol_terminal")
    html: Optional[str] = Field("", alias="symbol_html")
    latex: Optional[str] = Field("Unknown", alias="symbol_latex")
    confidence: float = Field(0.0, alias="symbol_conf")

    class Config:
        populate_by_name = True


class LegendItem(BaseModel):
    id: int
    color_bbox: List[int] = Field(default_factory=list)
    text_bbox: List[int] = Field(default_factory=list)
    avg_color: List[int] = Field(...)
    color_name: str = Field("unknown")
    area: float = Field(0.0)
    color_img: str = Field("")
    text_img: str = Field("")
    hist: Optional[List[float]] = Field(None)

    legend_text: str = Field("")
    formation: str = Field("")         # VLM识别的地层名称
    lithology: str = Field("")         # VLM识别的岩性描述
    age_text: str = Field("")          # VLM识别的地质年代文字
    symbol: GeoSymbol = Field(default_factory=GeoSymbol)
    is_recognized: bool = Field(False)
    legend_text_confidence: float = Field(0.0, ge=0.0, le=1.0, description="VLM识别图例文字的置信度")

    color_patch_base64: Optional[str] = Field(None, exclude=True)
    text_patch_base64: Optional[str] = Field(None, exclude=True)


class RegionGeoInfo(BaseModel):
    symbol: GeoSymbol = Field(default_factory=GeoSymbol)
    unit_name: str = Field("")
    age: Optional[str] = Field(None)
    genesis: Optional[str] = Field(None)
    legend_id: int = Field(-1)
    match_score: float = Field(0.0)
    legend_color_name: str = Field("unknown")
    legend_color_rgb: List[int] = Field(default_factory=list)


class RegionMatch(BaseModel):
    legend_id: int
    match_degree: float
    raw: float
    dE: float
    hist: float
    legend_rgb: List[int]
    legend_name: str


class RegionItem(BaseModel):
    id: int
    contour: List[List[float]] = Field(default_factory=list)
    centroid: List[float] = Field(default_factory=lambda: [-1.0, -1.0])
    area: float = Field(-1.0)
    matched_legend_id: int = Field(-1)
    match_score: float = Field(0.0)
    geo: RegionGeoInfo = Field(default_factory=RegionGeoInfo)
    region_color_rgb: List[int] = Field(default_factory=lambda: [0, 0, 0])
    top_matches: List[RegionMatch] = Field(default_factory=list)


class ImageAnalysisResult(BaseModel):
    summary: str
    error_code: Optional[str] = None
    features: List[Dict[str, Any]] = Field(default_factory=list)
    legends: List[LegendItem] = Field(default_factory=list)
    regions: List[RegionItem] = Field(default_factory=list)
    annotated_map_base64: Optional[str] = None
    cropped_legend_base64: Optional[str] = None
    artifacts_dir: Optional[str] = None


from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


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
