"""Preserve source identities and serialize model-selected rows without guessing.

No automatic layer renumbering, thickness replacement or entity creation here.
The same serializer must be used in every experimental group.
"""
import csv
import io
import json
import re
from fusion_validation import CSV_COLUMNS, FusionOutputError
from fusion_matching import match_entities


def as_record(value):
    if hasattr(value, "model_dump"):
        return value.model_dump()
    return value if isinstance(value, dict) else {"description": str(value)}


def build_fusion_prompt(legends, strata, profiles):
    legend_fields = ("id", "legend_text", "formation", "lithology", "age_text", "symbol", "is_recognized")
    # Retain every entity but omit pixel histograms, local paths and image crops.
    compact_legends = [{k: value[k] for k in legend_fields if k in value}
                       for value in (as_record(v) for v in legends)]
    sources = {"legends": compact_legends,
               "strata": [as_record(v) for v in strata],
               "profiles": [as_record(v) for v in profiles]}
    sources["alignment_audit"] = match_entities(sources["strata"], compact_legends)
    example = {"rows": [dict(zip(CSV_COLUMNS, ["part_1", "示例名称（不可照抄）", 1, "", "", "", "", 30]))]}
    return (
        "你是地质数据对齐专家。依据下面的图例、地层实体和剖面记录生成融合配置。\n"
        "只输出JSON对象，顶层为rows数组；程序负责转换成CSV，不要自己拼接CSV。\n"
        f"每条记录必须包含全部8个键：{json.dumps(CSV_COLUMNS, ensure_ascii=False)}。"
        "年代字段缺失写空字符串，不省略键；formation和part_code不能为空。\n"
        "part_code固定part_1。formation_code是模型层序编号，不是原始地层代号；"
        "先依据原文年代和层序确定由新到老的排列，再按记录顺序编号1,2,3...，不可重复。"
        "不能为了通过检查重排真实层序；无法确定层序则返回空rows，交人工复核。\n"
        "厚度必须为有限正数，单位米。优先使用有明确对应关系的剖面厚度；缺少厚度时"
        "允许使用建模假设的默认30米，不把默认值当作实测值。不要虚构地层或实测厚度。"
        "名称和原始地层代号仅用于跨模态对齐；不将岩体、断层或其他图例自动当作地层。\n"
        "alignment_audit是程序计算的图文候选对应关系及评分，不是模型自评概率。"
        "仅accepted关系可作为已建立对应关系的依据；needs_review不能强行解释为已匹配。"
        "评分不证明地质层序正确，不用FC建模编号替代原始地层代号。\n"
        f"格式示例（仅展示键，不得照抄示例实体）：{json.dumps(example, ensure_ascii=False)}\n"
        "完整结构化输入：\n" + json.dumps(sources, ensure_ascii=False, separators=(",", ":"))
    )


def response_to_csv(raw):
    if not isinstance(raw, str):
        raise FusionOutputError("响应必须是JSON文本", str(raw))
    content = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()
    content = re.sub(r"^```(?:json)?\s*|\s*```$", "", content, flags=re.IGNORECASE).strip()
    try:
        payload = json.loads(content)
    except (ValueError, TypeError) as exc:
        raise FusionOutputError("JSON无法解析；只返回完整的{\"rows\":[...]}对象，不要CSV或说明文字", raw) from exc
    if not isinstance(payload, dict) or set(payload) != {"rows"} or not isinstance(payload["rows"], list):
        raise FusionOutputError("顶层必须仅包含rows数组", raw)
    output = io.StringIO(newline="")
    writer = csv.writer(output, lineterminator="\n")
    writer.writerow(CSV_COLUMNS)
    for index, row in enumerate(payload["rows"], 1):
        if not isinstance(row, dict) or set(row) != set(CSV_COLUMNS):
            keys = set(row) if isinstance(row, dict) else set()
            raise FusionOutputError(f"第{index}条记录键不完整：缺少{sorted(set(CSV_COLUMNS)-keys)}，"
                                    f"多余{sorted(keys-set(CSV_COLUMNS))}；8个键全部保留", raw)
        if any(isinstance(v, (list, dict, bool)) for v in row.values()):
            raise FusionOutputError(f"第{index}条记录含非标量或布尔字段", raw)
        # Empty optional ages are representational only; critical nulls remain
        # empty and fail the unchanged downstream identity/numeric checks.
        writer.writerow(["" if row[k] is None else row[k] for k in CSV_COLUMNS])
    return output.getvalue()


def generate_fusion_csv(llm, prompt, previous, issues):
    from langchain_core.messages import HumanMessage
    feedback = ""
    if issues:
        feedback = ("\n上次配置未通过程序校验。仅修正明确列出的错误，仍以完整原始输入为依据。"
                    "继续返回rows JSON，不要返回CSV。下面的CSV列名与JSON键一一对应；"
                    "formation_code必须按确定的层序从1递增，厚度缺失可用已声明的30米默认值。"
                    "禁止为通过检查添加虚构地层或任意改写有来源的厚度。\n错误："
                    + json.dumps(issues, ensure_ascii=False) + "\n上次输出：\n" + previous)
    response = llm.invoke([HumanMessage(content=prompt + feedback)])
    return response_to_csv(response.content if response else "")
