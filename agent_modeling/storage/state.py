"""Result serialization and checkpoint compatibility."""


import math


EXCLUDED_FIELDS = frozenset({
    "messages", "chat_history", "annotated_map_base64", "cropped_legend_base64",
    "color_patch_base64", "text_patch_base64",
    "parsed_documents", "documents", "pipeline",
})


def serialize_result_state(obj, depth=0, *, max_depth=20, max_items=None):
    """Preserve nested model fields; reject excessive nesting instead of corrupting it."""
    if obj is None or isinstance(obj, (str, int, bool)):
        return obj
    if isinstance(obj, float):
        return round(obj, 6) if math.isfinite(obj) else None
    if depth > max_depth:
        raise ValueError("Analysis result exceeds the supported nesting depth")
    if hasattr(obj, "model_dump"):
        obj = obj.model_dump()
    elif hasattr(obj, "dict"):
        obj = obj.dict()

    def convert(value):
        return serialize_result_state(
            value, depth + 1, max_depth=max_depth, max_items=max_items
        )

    if isinstance(obj, dict):
        return {key: convert(value) for key, value in obj.items() if key not in EXCLUDED_FIELDS}
    if isinstance(obj, (list, tuple)):
        return [convert(value) for value in obj[:max_items]]
    return str(obj)[:200]


import importlib
import sys

MODEL_TYPES = {
    "agent_modeling.models": (
        "ImageAnalysisResult", "LegendItem", "GeoSymbol", "RegionItem", "RegionGeoInfo", "RegionMatch",
        "TextInfoResult", "LLMExtraction", "ExtractedStratum", "ExtractedProfile", "StratumItem", "ProfileItem", "KnowledgeGraphData", "Node", "Edge",
    ),
}
LEGACY_MODEL_MODULES = {
    "image_agent": "agent_modeling.models",
    "text_info_agent": "agent_modeling.models",
    "kg_builder": "agent_modeling.models",
    "agent_modeling.schemas.image": "agent_modeling.models",
    "agent_modeling.schemas.text": "agent_modeling.models",
    "agent_modeling.schemas.knowledge_graph": "agent_modeling.models",
}
_IMAGE_TYPES = ("ImageAnalysisResult", "LegendItem", "GeoSymbol", "RegionItem", "RegionGeoInfo", "RegionMatch")
_TEXT_TYPES = ("TextInfoResult", "LLMExtraction", "StratumItem", "ProfileItem")
_GRAPH_TYPES = ("KnowledgeGraphData", "Node", "Edge")
_LEGACY_TYPES = {
    "image_agent": _IMAGE_TYPES, "agent_modeling.schemas.image": _IMAGE_TYPES,
    "text_info_agent": _TEXT_TYPES, "agent_modeling.schemas.text": _TEXT_TYPES,
    "kg_builder": _GRAPH_TYPES, "agent_modeling.schemas.knowledge_graph": _GRAPH_TYPES,
}


def trusted_checkpoint_types():
    """Allow only known business types; old module names resolve to pure schemas."""
    trusted = [(module, name) for module, names in MODEL_TYPES.items() for name in names]
    for old, new in LEGACY_MODEL_MODULES.items():
        sys.modules.setdefault(old, importlib.import_module(new))
        trusted.extend((old, name) for name in _LEGACY_TYPES[old])
    return trusted
