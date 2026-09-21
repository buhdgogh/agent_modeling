"""Auditable candidate alignment; scores are not calibrated probabilities.

No spatial score: current strata records have no comparable polygon adjacency.
Defaults are engineering settings, not fitted or experimentally optimal values.
"""
import re
import unicodedata

WEIGHTS = {"name": 0.5, "age": 0.3, "lithology": 0.2}
THRESHOLD = 0.8
MARGIN = 0.1
PERIODS = ("寒武", "奥陶", "志留", "泥盆", "石炭", "二叠", "三叠", "侏罗", "白垩", "古近", "新近", "第四")
AGE_GROUPS = {"古生": PERIODS[:6], "中生": PERIODS[6:9], "新生": PERIODS[9:],
              "元古": ("古元古", "中元古", "新元古"), "太古": ("太古",)}
ROCKS = ("砂岩", "页岩", "泥岩", "砾岩", "灰岩", "白云岩", "石英岩", "片岩",
         "片麻岩", "板岩", "千枚岩", "变粒岩", "大理岩", "凝灰岩", "玄武岩", "流纹岩", "花岗岩")
ROCK_ALIASES = {"流纹岩": ("流纹岩", "流纹质岩"), "凝灰岩": ("凝灰岩", "凝灰质岩")}
UNKNOWN = {"", "unknown", "none", "null", "未知", "不详"}


def record(value):
    return value.model_dump() if hasattr(value, "model_dump") else value


def normalize(value):
    value = unicodedata.normalize("NFKC", str(value or "")).casefold()
    return re.sub(r"[^\w\u3400-\u9fff]", "", value)


def present(value):
    return normalize(value) not in UNKNOWN


def name_score(left, right):
    left, right = normalize(left), normalize(right)
    if not present(left) or not present(right):
        return None
    if left == right:
        return 1.0
    a = {left[i:i+2] for i in range(len(left)-1)} or {left}
    b = {right[i:i+2] for i in range(len(right)-1)} or {right}
    return 2 * len(a & b) / (len(a) + len(b))


def age_units(value):
    """Recognize broad eras/periods only; do not invent numerical age ranges."""
    value = normalize(value)
    units = {p for p in PERIODS if p in value}
    proterozoic = {p for p in AGE_GROUPS["元古"] if p in value}
    if proterozoic:
        units |= proterozoic
    for group, children in AGE_GROUPS.items():
        if group in value and not any(p in value for p in children):
            units.update(children)
    return units


def age_score(left, right):
    if not present(left) or not present(right):
        return None
    a, b = age_units(left), age_units(right)
    if a and b:
        return len(a & b) / min(len(a), len(b))
    # Different unrecognized labels are missing evidence, not a hard conflict.
    return 1.0 if normalize(left) == normalize(right) else None


def lithology_score(left, right):
    def terms(value):
        value = str(value or "")
        found = {r for r in ROCKS if r in value}
        found.update(canonical for canonical, variants in ROCK_ALIASES.items()
                     if any(term in value for term in variants))
        return found
    a, b = terms(left), terms(right)
    return len(a & b) / len(a | b) if a and b else None


def legend_code(legend):
    symbol = record(legend.get("symbol") or {})
    for key in ("final", "symbol_final"):
        if present(symbol.get(key)):
            return normalize(symbol[key])
    return ""


def pair_score(text, legend):
    text, legend = record(text), record(legend)
    # Prefer the finer text age instead of concatenating broad and fine labels.
    age = next((text.get(k) for k in ("formation_age_2", "formation_age_1", "age_text")
                if present(text.get(k))), None)
    scores = {"name": name_score(text.get("formation"), legend.get("formation")),
              "age": age_score(age, legend.get("age_text")),
              "lithology": lithology_score(text.get("rock_features") or text.get("lithology"), legend.get("lithology"))}
    available = {k: v for k, v in scores.items() if v is not None}
    denominator = sum(WEIGHTS[k] for k in available)
    confidence = sum(WEIGHTS[k] * v for k, v in available.items()) / denominator if denominator else 0.0
    code = normalize(text.get("formation_code")) if present(text.get("formation_code")) else ""
    other = legend_code(legend)
    conflicts = []
    warnings = []
    if code and other and code != other:
        if scores["name"] == 1.0 and scores["lithology"] is not None and scores["lithology"] > 0:
            warnings.append("code_conflict_overridden_by_exact_name_and_lithology")
        else:
            conflicts.append("original_code_conflict")
    if scores["age"] == 0:
        conflicts.append("recognized_age_conflict")
    return {"scores": scores, "confidence": confidence, "conflicts": conflicts, "warnings": warnings,
            "code_equal": bool(code and other and code == other),
            "sufficient_evidence": scores["name"] is not None and
                (scores["age"] is not None or scores["lithology"] is not None)}


def match_entities(strata, legends):
    """One decision per text record, with every candidate retained for audit.

    Several text descriptions may legitimately refer to the same legend.
    Repeated legend entries remain ambiguous rather than silently deduplicated.
    """
    legends = [record(v) for v in legends]
    decisions = []
    for i, text in enumerate(strata):
        candidates = [dict(pair_score(text, legend), legend_index=j,
                           legend_id=legend.get("id", j)) for j, legend in enumerate(legends)]
        eligible = [c for c in candidates if not c["conflicts"]]
        exact = [c for c in eligible if c["code_equal"]]
        chosen, reason = None, "insufficient_or_low_score"
        if len(exact) == 1:
            chosen, reason = exact[0], "unique_original_code"
        elif len(exact) > 1:
            reason = "ambiguous_original_code"
        else:
            ranked = sorted(eligible, key=lambda c: c["confidence"], reverse=True)
            if ranked:
                best = ranked[0]
                gap = best["confidence"] - ranked[1]["confidence"] if len(ranked) > 1 else 1.0
                if best["sufficient_evidence"] and best["confidence"] >= THRESHOLD:
                    if gap >= MARGIN - 1e-12:
                        chosen, reason = best, "weighted_evidence"
                    else:
                        reason = "ambiguous_candidates"
        decisions.append({"text_id": f"text_{i}", "text_index": i,
                          "status": "accepted" if chosen else "needs_review",
                          "reason": reason, "legend_index": chosen["legend_index"] if chosen else None,
                          "legend_id": chosen["legend_id"] if chosen else None,
                          "confidence": chosen["confidence"] if chosen else None,
                          "candidates": candidates})
    return {"weights": WEIGHTS.copy(), "threshold": THRESHOLD, "margin": MARGIN,
            "spatial_score_used": False, "decisions": decisions}


def evaluate_matches(predicted, gold):
    """gold is independently annotated (text_id, legend_index) pairs."""
    predicted, gold = set(map(tuple, predicted)), set(map(tuple, gold))
    tp, fp, fn = len(predicted & gold), len(predicted - gold), len(gold - predicted)
    return {"tp": tp, "fp": fp, "fn": fn,
            "precision": tp / (tp + fp) if tp + fp else 0.0,
            "recall": tp / (tp + fn) if tp + fn else 0.0,
            "f1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0}
