"""Deterministic configuration checks and bounded, same-model fusion repair.

These checks validate a modelling configuration, not geological truth.
"""
import csv
import io
import math
import time
from collections import defaultdict

CSV_COLUMNS = ["part_code", "formation", "formation_code", "formation_age_1",
               "formation_age_code_1", "formation_age_2", "formation_age_code_2", "厚度"]


class FusionOutputError(ValueError):
    """Invalid model serialization, distinct from an API transport failure."""
    def __init__(self, message, raw):
        super().__init__(message)
        self.raw = raw


def validate_fusion_csv(content):
    issues = []
    try:
        reader = csv.reader(io.StringIO(content.lstrip("\ufeff")), strict=True)
        header = next(reader, [])
        if header != CSV_COLUMNS:
            return [{"code": "invalid_header", "message": "表头必须严格为：" + ",".join(CSV_COLUMNS)}]
        groups = defaultdict(list)
        count = 0
        for line, row in enumerate(reader, 2):
            if not row or not any(cell.strip() for cell in row):
                continue
            count += 1
            if len(row) != len(CSV_COLUMNS):
                issues.append({"code": "invalid_columns", "line": line,
                               "actual_columns": len(row), "expected_columns": CSV_COLUMNS,
                               "message": "记录必须包含8列，空年代字段也必须保留，不能省略"})
                continue
            row = [cell.strip() for cell in row]
            if not row[0] or not row[1]:
                issues.append({"code": "missing_identity", "line": line, "message": "区域编号和地层名称不能为空"})
            try:
                thickness = float(row[-1])
                if not math.isfinite(thickness) or thickness <= 0:
                    raise ValueError()
            except ValueError:
                issues.append({"code": "invalid_thickness", "line": line, "message": "厚度必须是有限的正数，单位米"})
            try:
                code = int(row[2])
                if code <= 0:
                    raise ValueError()
                groups[row[0]].append(code)
            except ValueError:
                issues.append({"code": "invalid_layer_index", "line": line, "message": "formation_code必须是正整数层序编号，不是原始地层代号"})
        if not count:
            issues.append({"code": "empty_configuration", "message": "至少提供一条地层配置"})
        for part, codes in groups.items():
            if codes != list(range(1, len(codes) + 1)):
                issues.append({"code": "invalid_layer_sequence", "part_code": part,
                               "message": "每个区域的层序编号必须按记录顺序从1连续递增，且不能重复"})
    except (csv.Error, TypeError, AttributeError) as exc:
        issues.append({"code": "invalid_csv", "message": "CSV无法解析：" + str(exc)})
    return issues


def repair_fusion(generate, max_repairs=2, on_check=None):
    """generate(previous_csv, issues) must reuse the same model and source inputs.

    Returns (accepted CSV or None, trace). Exceptions propagate to the caller;
    transport retries are not counted as repair rounds.
    """
    if not isinstance(max_repairs, int) or not 0 <= max_repairs <= 2:
        raise ValueError("max_repairs must be an integer between 0 and 2")
    previous, issues, trace = "", [], []
    for round_index in range(max_repairs + 1):
        started = time.perf_counter()
        try:
            content = generate(previous, issues)
        except FusionOutputError as exc:
            content = exc.raw
            issues = [{"code": "invalid_model_output", "message": str(exc)}]
        else:
            issues = validate_fusion_csv(content)
        trace.append({"round": round_index, "stage": "fusion", "passed": not issues,
                      "issues": issues, "csv": content,
                      "duration_ms": round((time.perf_counter() - started) * 1000, 2)})
        if on_check:
            on_check(trace)
        if not issues:
            return content, trace
        previous = content
    return None, trace
