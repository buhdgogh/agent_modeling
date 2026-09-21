"""Bounded, JSON-safe conversion of analysis results for chat history."""

import math


EXCLUDED_FIELDS = frozenset({
    "messages", "chat_history", "annotated_map_base64", "cropped_legend_base64",
    "color_patch_base64", "text_patch_base64",
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
