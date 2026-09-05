"""
定性验证 2.1 — 地质叠加定律规则硬审查 (Superposition Principle Validation)

原理:
    虚拟钻孔的核心价值在于其必须符合基础地质力学与沉积规律。
    对生成的数十万虚拟钻孔坐标点进行遍历审查，检查层序倒置与厚度突变。

审查规则:
    1. 层序倒置校验 (Superposition Violation):
       检查同一个 (X, Y) 坐标下，随着深度 Z 的增加，地层的出现顺序是否严格遵循
       "由新到老"的字典序列。存在逻辑矛盾时标记为倒置违规。

    2. 厚度突变校验 (Thickness Anomaly):
       检查相邻两个间距为 100m 的虚拟钻孔中，同一地层单元的底界高程差是否异常
       （斜率是否超过该区域最大的岩层倾角极限）。

输出:
    - 违规率 (Violation Rate): 违规钻孔占比
    - 违规详细列表: 每处违规的坐标、地层、偏差值
    - 合格判定: 违规率 < 1% 视为通过

学术意义:
    在定性层面证明大模型逻辑链条的闭环稳定性——LLM 生成的文本厚度与空间组合
    遵循地质学基本定律。
"""

import os
import json
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Set
from dataclasses import dataclass, field
from datetime import datetime
from scipy.spatial import cKDTree


@dataclass
class SuperpositionResult:
    """叠置定律审查结果"""
    n_boreholes_total: int = 0                  # 总钻孔数
    n_boreholes_checked: int = 0                 # 实际检查的钻孔数
    n_points_total: int = 0                      # 总虚拟钻孔点数

    # 层序倒置
    superposition_violations: int = 0            # 层序倒置违规的钻孔数
    superposition_violation_rate: float = 0.0    # 层序倒置违规率
    superposition_details: List[Dict] = field(default_factory=list)  # 违规详情

    # 厚度突变
    thickness_anomalies: int = 0                 # 厚度突变违规的钻孔对数
    thickness_anomaly_rate: float = 0.0          # 厚度突变违规率
    thickness_details: List[Dict] = field(default_factory=list)

    # 综合
    overall_violation_rate: float = 0.0          # 综合违规率
    passed: bool = False                         # 是否通过
    grade: str = ""                              # 等级


class SuperpositionValidator:
    """
    地质叠加定律规则硬审查器

    使用方法:
        validator = SuperpositionValidator(max_dip_angle=60.0)
        result = validator.run(virtual_csv, formation_csv)
    """

    def __init__(self,
                 max_dip_angle: float = 60.0,
                 neighbor_distance: float = 100.0,
                 thickness_tolerance: float = 3.0):
        """
        Parameters
        ----------
        max_dip_angle : float
            区域最大岩层倾角 (度), 超出此角度的厚度变化视为异常
        neighbor_distance : float
            相邻钻孔判定距离 (m)
        thickness_tolerance : float
            同一地层在相邻钻孔中的厚度变异系数阈值
        """
        self.max_dip_angle = max_dip_angle
        self.max_dip_slope = np.tan(np.radians(max_dip_angle))
        self.neighbor_distance = neighbor_distance
        self.thickness_tolerance = thickness_tolerance

    def run(self,
            virtual_csv: str,
            formation_csv: Optional[str] = None) -> SuperpositionResult:
        """
        执行叠置定律审查

        Parameters
        ----------
        virtual_csv : str
            虚拟钻孔点云 CSV 路径
        formation_csv : str, optional
            地层特征 CSV (用于获取标准地层序列 "由新到老")

        Returns
        -------
        SuperpositionResult
        """
        df = pd.read_csv(virtual_csv)

        # 自动检测列名
        x_col = self._detect_col(df, ['x', 'x_m', 'lon', 'easting'])
        y_col = self._detect_col(df, ['y', 'y_m', 'lat', 'northing'])
        z_col = self._detect_col(df, ['z', 'z_m', 'elev', 'elevation'])
        fm_col = self._detect_col(df, ['formation_code', 'formation', 'surface'])
        bh_col = self._detect_col(df, ['borehole_id', 'bh_id', 'borehole'])

        result = SuperpositionResult()
        result.n_points_total = len(df)

        # 获取"由新到老"的标准地层序列
        age_order = None
        if formation_csv and os.path.exists(formation_csv):
            age_order = self._extract_age_order(formation_csv)

        # 按钻孔分组
        if bh_col and bh_col in df.columns:
            groups = df.groupby(bh_col)
        else:
            # 降级: 用 (x, y) 分组
            df = df.copy()
            df['_bh_key'] = df.apply(lambda r: f"{r[x_col]:.1f}_{r[y_col]:.1f}", axis=1)
            groups = df.groupby('_bh_key')
            bh_col = '_bh_key'

        result.n_boreholes_total = len(groups)

        # ========================================
        # 审查 1: 层序倒置校验
        # ========================================
        borehole_profiles = {}  # {bh_id: {x, y, z_top, layers: [(z, fm), ...]}}

        for bh_id, group in groups:
            group_sorted = group.sort_values(z_col, ascending=False)  # 从地表到深部

            x_pos = float(group_sorted[x_col].iloc[0])
            y_pos = float(group_sorted[y_col].iloc[0])
            z_top = float(group_sorted[z_col].max())

            layers = list(group_sorted[[z_col, fm_col]].itertuples(index=False, name=None))
            layers = [(float(z), str(fm)) for z, fm in layers]
            # 去重相邻的同一地层
            deduped = []
            for z, fm in layers:
                if not deduped or deduped[-1][1] != fm:
                    deduped.append((z, fm))
            layers = deduped

            borehole_profiles[bh_id] = {
                "x": x_pos,
                "y": y_pos,
                "z_top": z_top,
                "layers": layers,
            }

            # 检查层序
            if age_order:
                violations = self._check_superposition_order(layers, age_order)
                if violations:
                    result.superposition_violations += 1
                    for v in violations:
                        result.superposition_details.append({
                            "borehole_id": str(bh_id),
                            "x": x_pos,
                            "y": y_pos,
                            "type": "层序倒置",
                            "detail": v,
                        })
            else:
                # 无年代序列时,检查是否有明显的往返(新—老—新)
                violations = self._check_z_order_consistency(layers)
                if violations:
                    result.superposition_violations += 1
                    for v in violations:
                        result.superposition_details.append({
                            "borehole_id": str(bh_id),
                            "x": x_pos,
                            "y": y_pos,
                            "type": "Z值跳跃异常",
                            "detail": v,
                        })

        result.n_boreholes_checked = len(borehole_profiles)

        if result.n_boreholes_checked > 0:
            result.superposition_violation_rate = (
                result.superposition_violations / result.n_boreholes_checked
            )

        # ========================================
        # 审查 2: 厚度突变校验
        # ========================================
        bh_ids = list(borehole_profiles.keys())
        bh_coords = np.array([[borehole_profiles[bid]["x"],
                               borehole_profiles[bid]["y"]] for bid in bh_ids])

        if len(bh_coords) >= 2:
            tree = cKDTree(bh_coords)

            checked_pairs = set()
            for i, bid in enumerate(bh_ids):
                # 找 100m 范围内的邻居
                neighbors = tree.query_ball_point(bh_coords[i], self.neighbor_distance)

                for j in neighbors:
                    if j <= i:
                        continue

                    pair_key = (min(bid, bh_ids[j]), max(bid, bh_ids[j]))
                    if pair_key in checked_pairs:
                        continue
                    checked_pairs.add(pair_key)

                    anomaly = self._check_thickness_anomaly(
                        borehole_profiles[bid],
                        borehole_profiles[bh_ids[j]],
                        bid, bh_ids[j]
                    )
                    if anomaly:
                        result.thickness_anomalies += 1
                        result.thickness_details.append(anomaly)

            n_pairs = len(bh_ids) * (len(bh_ids) - 1) / 2
            if n_pairs > 0:
                result.thickness_anomaly_rate = (
                    result.thickness_anomalies / min(n_pairs, len(checked_pairs))
                    if checked_pairs else 0.0
                )

        # ========================================
        # 综合评定
        # ========================================
        result.overall_violation_rate = (
            result.superposition_violations + result.thickness_anomalies
        ) / max(result.n_boreholes_checked, 1)

        result.grade = self._assign_grade(result.overall_violation_rate)
        result.passed = result.overall_violation_rate < 0.01  # 违规率 < 1%

        return result

    def _extract_age_order(self, formation_csv: str) -> Dict[str, int]:
        """
        从特征 CSV 中提取地层由新到老的顺序

        优先级:
        1. formation_code 列 (1, 2, 3... 从新到老)
        2. formation_age_1 列 (根据地质年代排序)
        """
        df = pd.read_csv(formation_csv)

        age_order = {}

        if "formation_code" in df.columns:
            for _, row in df.iterrows():
                fm = str(row.get("formation", ""))
                code = row.get("formation_code")
                if pd.notna(code) and fm:
                    try:
                        age_order[fm] = int(code)
                    except (ValueError, TypeError):
                        pass

        # 也添加 formation_code 本身的映射
        if "formation_code" in df.columns:
            for _, row in df.iterrows():
                code = str(row.get("formation_code", ""))
                try:
                    age_order[code] = int(row["formation_code"])
                except (ValueError, TypeError):
                    pass

        return age_order

    def _check_superposition_order(self, layers: List[Tuple[float, str]],
                                    age_order: Dict[str, int]) -> List[str]:
        """
        检查层序是否符合由新到老（age_order 中数字越小越新）

        layers: [(z, formation_code), ...] 从地表到深部
        """
        violations = []

        for i in range(1, len(layers)):
            fm_above = layers[i - 1][1]
            fm_below = layers[i][1]

            age_above = age_order.get(fm_above)
            age_below = age_order.get(fm_below)

            if age_above is not None and age_below is not None:
                if age_above > age_below:
                    # 上方地层比下方老 → 倒置
                    violations.append(
                        f"地层 '{fm_above}' (序号{age_above}) 位于 '{fm_below}' "
                        f"(序号{age_below}) 之上 — 疑似倒置"
                    )

        return violations

    def _check_z_order_consistency(self, layers: List[Tuple[float, str]]) -> List[str]:
        """
        简易 Z 顺序一致性检查：同一 formation_code 不应在不同深度层次交替出现
        """
        violations = []
        fm_zones = {}  # {fm: [index_range]}

        for i, (z, fm) in enumerate(layers):
            if fm not in fm_zones:
                fm_zones[fm] = []
            fm_zones[fm].append(i)

        for fm, indices in fm_zones.items():
            if len(indices) > 1:
                # 检查是否连续
                for j in range(1, len(indices)):
                    if indices[j] - indices[j - 1] > 1:
                        violations.append(
                            f"地层 '{fm}' 在钻孔中出现不连续 (层位: {indices})"
                        )
                        break

        return violations

    def _check_thickness_anomaly(self,
                                  profile_a: Dict,
                                  profile_b: Dict,
                                  id_a: str, id_b: str) -> Optional[Dict]:
        """
        检查两个相邻钻孔间同一地层的厚度突变

        profile: {x, y, z_top, layers: [(z, fm), ...]}
        """
        # 提取每个钻孔的地层厚度
        def get_thicknesses(layers):
            thicknesses = {}
            for i in range(len(layers) - 1):
                fm = layers[i][1]
                thick = layers[i][0] - layers[i + 1][0]
                thicknesses[fm] = thick
            return thicknesses

        thick_a = get_thicknesses(profile_a["layers"])
        thick_b = get_thicknesses(profile_b["layers"])

        # 计算二维距离
        dx = profile_a["x"] - profile_b["x"]
        dy = profile_a["y"] - profile_b["y"]
        dist_2d = np.sqrt(dx * dx + dy * dy)

        if dist_2d < 1e-6:
            return None

        # 检查共同地层
        common_fms = set(thick_a.keys()) & set(thick_b.keys())
        for fm in common_fms:
            t_a = thick_a[fm]
            t_b = thick_b[fm]
            if t_a <= 0 or t_b <= 0:
                continue

            # 计算表观倾角
            dz = abs(t_a - t_b)
            apparent_dip = np.arctan(dz / dist_2d)
            apparent_dip_deg = np.degrees(apparent_dip)

            if apparent_dip_deg > self.max_dip_angle:
                return {
                    "borehole_a": str(id_a),
                    "borehole_b": str(id_b),
                    "distance_m": round(dist_2d, 2),
                    "formation": fm,
                    "thickness_a_m": round(t_a, 2),
                    "thickness_b_m": round(t_b, 2),
                    "apparent_dip_deg": round(apparent_dip_deg, 1),
                    "max_allowed_dip_deg": self.max_dip_angle,
                }

        return None

    def _detect_col(self, df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
        """从候选列名中检测存在的列"""
        for c in candidates:
            if c in df.columns:
                return c
        return candidates[0] if candidates else df.columns[0]

    def _assign_grade(self, violation_rate: float) -> str:
        if violation_rate < 0.005:
            return "A (优) — 地质规律违规率 < 0.5%，逻辑链高度稳定"
        elif violation_rate < 0.01:
            return "B (良) — 地质规律违规率 < 1%，逻辑链基本稳定"
        elif violation_rate < 0.05:
            return "C (中) — 存在一定地质规律违规，建议人工复核"
        else:
            return "D (差) — 违规率较高，LLM 地层推演逻辑链需优化"

    def format_report(self, result: SuperpositionResult) -> str:
        """格式化输出审查报告"""
        lines = [
            "=" * 60,
            "  地质叠加定律规则硬审查报告",
            "=" * 60,
            f"  总钻孔数:           {result.n_boreholes_total}",
            f"  有效检查钻孔数:      {result.n_boreholes_checked}",
            f"  总虚拟钻孔点数:      {result.n_points_total}",
            "-" * 60,
            f"  [层序倒置]",
            f"    违规钻孔数:        {result.superposition_violations}",
            f"    违规率:            {result.superposition_violation_rate:.4f} ({result.superposition_violation_rate * 100:.2f}%)",
            "-" * 60,
            f"  [厚度突变]",
            f"    违规钻孔对数:      {result.thickness_anomalies}",
            f"    违规率:            {result.thickness_anomaly_rate:.4f} ({result.thickness_anomaly_rate * 100:.2f}%)",
            "-" * 60,
            f"  综合违规率:          {result.overall_violation_rate:.4f} ({result.overall_violation_rate * 100:.2f}%)",
            f"  通过验证:            {result.passed}",
            f"  质量等级:            {result.grade}",
            "=" * 60,
        ]

        if result.superposition_details:
            lines.append(f"\n  [层序倒置详情] (前10条):")
            for detail in result.superposition_details[:10]:
                lines.append(f"    BH={detail['borehole_id']}, "
                             f"({detail['x']:.1f}, {detail['y']:.1f}): {detail['detail']}")

        if result.thickness_details:
            lines.append(f"\n  [厚度突变详情] (前10条):")
            for detail in result.thickness_details[:10]:
                lines.append(f"    {detail['borehole_a']}<->{detail['borehole_b']}: "
                             f"{detail['formation']}, "
                             f"Δ厚度={abs(detail['thickness_a_m'] - detail['thickness_b_m']):.1f}m, "
                             f"视倾角={detail['apparent_dip_deg']:.1f}°")

        return "\n".join(lines)


    def save_results(self, result: SuperpositionResult, output_dir: str,
                      prefix: str = "superposition") -> Dict[str, str]:
        """Save validation report, structured data, and violation details to output directory."""
        os.makedirs(output_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        saved = {}

        report_path = os.path.join(output_dir, f"{prefix}_report_{timestamp}.txt")
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(self.format_report(result))
        saved["report"] = report_path

        data_dict = {
            "n_boreholes_total": result.n_boreholes_total,
            "n_boreholes_checked": result.n_boreholes_checked,
            "n_points_total": result.n_points_total,
            "superposition_violations": result.superposition_violations,
            "superposition_violation_rate": result.superposition_violation_rate,
            "superposition_details": result.superposition_details[:100],
            "thickness_anomalies": result.thickness_anomalies,
            "thickness_anomaly_rate": result.thickness_anomaly_rate,
            "thickness_details": result.thickness_details[:100],
            "overall_violation_rate": result.overall_violation_rate,
            "passed": result.passed,
            "grade": result.grade,
        }
        json_path = os.path.join(output_dir, f"{prefix}_data_{timestamp}.json")
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(data_dict, f, ensure_ascii=False, indent=2, default=str)
        saved["json"] = json_path

        # Also save violation CSVs for detailed inspection
        if result.superposition_details:
            sp_csv = os.path.join(output_dir, f"{prefix}_superposition_violations_{timestamp}.csv")
            pd.DataFrame(result.superposition_details).to_csv(sp_csv, index=False, encoding="utf-8-sig")
            saved["superposition_csv"] = sp_csv
        if result.thickness_details:
            th_csv = os.path.join(output_dir, f"{prefix}_thickness_anomalies_{timestamp}.csv")
            pd.DataFrame(result.thickness_details).to_csv(th_csv, index=False, encoding="utf-8-sig")
            saved["thickness_csv"] = th_csv

        return saved


def demo_validation():
    """演示叠加定律审查流程（使用模拟数据）"""
    import tempfile

    tmpdir = tempfile.mkdtemp()
    print(f"[Demo] 使用临时目录: {tmpdir}")

    # 生成正常钻孔数据（绝大多数）
    np.random.seed(42)
    n_boreholes = 100
    rows = []

    for bh_idx in range(n_boreholes):
        x = np.random.uniform(0, 10000)
        y = np.random.uniform(0, 10000)
        z_top = np.random.uniform(200, 500)

        current_z = z_top
        # 正常层序: F1 → F2 → F3 → F4 (由新到老)
        for fm_idx in range(4):
            thickness = np.random.uniform(30, 80)
            rows.append({
                "x": x, "y": y, "z": current_z,
                "formation_code": f"F{fm_idx + 1}",
                "borehole_id": f"BH{bh_idx:03d}"
            })
            current_z -= thickness

    # 故意插入少量违规钻孔
    for bad_idx in range(3):
        x = np.random.uniform(0, 10000)
        y = np.random.uniform(0, 10000)
        z_top = np.random.uniform(200, 500)

        # 倒置层序: F1 → F3 → F2 → F4
        bad_order = ["F1", "F3", "F2", "F4"]
        current_z = z_top
        for fm in bad_order:
            thickness = np.random.uniform(30, 80)
            rows.append({
                "x": x, "y": y, "z": current_z,
                "formation_code": fm,
                "borehole_id": f"BH_BAD_{bad_idx:02d}"
            })
            current_z -= thickness

    df_virtual = pd.DataFrame(rows)
    csv_path = os.path.join(tmpdir, "virtual_boreholes.csv")
    df_virtual.to_csv(csv_path, index=False)

    # 执行验证
    validator = SuperpositionValidator(max_dip_angle=60.0, neighbor_distance=100.0)
    result = validator.run(csv_path)

    print(validator.format_report(result))

    # Save to output/
    output_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "output")
    saved = validator.save_results(result, output_dir, prefix="superposition_demo")
    print(f"\n[Demo] 报告已保存至: {saved['report']}")
    print(f"[Demo] 数据已保存至: {saved['json']}")
    if "superposition_csv" in saved:
        print(f"[Demo] 层序倒置详情: {saved['superposition_csv']}")
    if "thickness_csv" in saved:
        print(f"[Demo] 厚度突变详情: {saved['thickness_csv']}")
    return result


if __name__ == "__main__":
    demo_validation()
