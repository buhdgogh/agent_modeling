"""
定量验证 1.1 — 真实钻孔盲测对比法 (Blind Test / Leave-One-Out Validation)

原理:
    将研究区内极少量（3-5个）已知真实钻孔在虚拟推演阶段"物理隔离（隐匿）"。
    在全自动生成密集虚拟钻孔点阵后，提取与真实钻孔平面坐标 (X,Y) 最接近
    的虚拟钻孔序列，比较地层分界面深度的绝对误差。

评价指标:
    - RMSE (均方根误差): 衡量虚拟钻孔对各岩性段底界标高的推演精度
    - MAE  (平均绝对误差): 平均偏差幅度
    - R^2   (决定系数): 虚拟钻孔与真实钻孔的深度相关性

输入:
    - virtual_csv:  系统生成的虚拟钻孔点云 CSV (x, y, z, formation_code, value, surface, borehole_id)
    - real_csv:     真实勘探钻孔 CSV (x, y, z, formation_code, ...)
    - formation_csv: 地层特征配置文件 (含 formation_code -> 厚度、年代映射)
"""

import os
import json
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field, asdict
from scipy.spatial import cKDTree
from datetime import datetime


@dataclass
class BlindTestResult:
    """盲测对比结果"""
    n_pairs: int = 0                          # 匹配到的虚拟-真实钻孔对数
    n_formations_evaluated: int = 0            # 参与评估的地层界面总数
    rmse: float = 0.0                          # 均方根误差 (m)
    mae: float = 0.0                           # 平均绝对误差 (m)
    max_error: float = 0.0                     # 最大绝对误差 (m)
    r_squared: float = 0.0                     # 决定系数 R^2
    per_formation_rmse: Dict[str, float] = field(default_factory=dict)   # 每个地层代码的 RMSE
    per_formation_mae: Dict[str, float] = field(default_factory=dict)    # 每个地层代码的 MAE
    per_pair_errors: List[Dict] = field(default_factory=list)            # 每对钻孔的详细误差
    passed: bool = False                       # 是否通过验证 (RMSE < 阈值)
    grade: str = ""                            # 等级: A(优) / B(良) / C(中) / D(差)


class BlindTestValidator:
    """
    真实钻孔盲测对比验证器

    使用流程:
        1. 加载虚拟钻孔 CSV 和真实钻孔 CSV
        2. 通过空间 KD-Tree 匹配最近邻钻孔对
        3. 对齐地层序列，逐层比较分界面深度
        4. 计算 RMSE / MAE / R^2 并评定等级
    """

    def __init__(self, max_match_distance: float = 500.0):
        """
        Parameters
        ----------
        max_match_distance : float
            虚拟钻孔与真实钻孔匹配的最大空间距离 (m)，超出此距离视为无法匹配
        """
        self.max_match_distance = max_match_distance

    def run(self,
            virtual_csv: str,
            real_csv: str,
            formation_csv: Optional[str] = None,
            z_tolerance: float = 5.0) -> BlindTestResult:
        """
        执行盲测对比验证

        Parameters
        ----------
        virtual_csv : str
            虚拟钻孔点云 CSV 路径 (含 x, y, z, formation_code, borehole_id)
        real_csv : str
            真实钻孔 CSV 路径 (含 x, y, z, formation_code 或等效列)
        formation_csv : str, optional
            地层特征 CS V 路径, 用于获取地层标准序列
        z_tolerance : float
            Z 坐标匹配容差 (m), 同一地层的真实-虚拟深度差在此范围内视为同一界面

        Returns
        -------
        BlindTestResult
        """
        df_virtual = pd.read_csv(virtual_csv)
        df_real = pd.read_csv(real_csv)

        # 列名映射：尝试自动识别坐标列
        x_col_v, y_col_v, z_col_v, fm_col_v, bh_col_v = self._detect_columns(df_virtual, mode="virtual")
        x_col_r, y_col_r, z_col_r, fm_col_r, bh_col_r = self._detect_columns(df_real, mode="real")

        # Step 1: 提取每个虚拟钻孔的地层分界面深度序列
        virtual_profiles = self._extract_depth_profiles(df_virtual, x_col_v, y_col_v, z_col_v,
                                                        fm_col_v, bh_col_v, mode="virtual")

        # Step 2: 提取每个真实钻孔的地层分界面深度序列
        real_profiles = self._extract_depth_profiles(df_real, x_col_r, y_col_r, z_col_r,
                                                     fm_col_r, bh_col_r, mode="real")

        # Step 3: 空间匹配 — 为每个真实钻孔找最近的虚拟钻孔
        pairs = self._match_boreholes(virtual_profiles, real_profiles)

        if not pairs:
            result = BlindTestResult()
            result.grade = "F (无法匹配)"
            return result

        # Step 4: 逐对比较地层分界面深度
        all_errors = []
        per_formation_errors: Dict[str, List[float]] = {}

        for pair in pairs:
            v_profile = pair["virtual"]
            r_profile = pair["real"]
            distance_2d = pair["distance_2d"]

            # 对齐地层序列：找共同的地层代码
            v_interfaces = v_profile["interfaces"]  # {formation_code: bottom_z}
            r_interfaces = r_profile["interfaces"]

            common_formations = set(v_interfaces.keys()) & set(r_interfaces.keys())

            for fm in common_formations:
                z_v = v_interfaces[fm]
                z_r = r_interfaces[fm]
                error = abs(z_v - z_r)

                all_errors.append(error)
                per_formation_errors.setdefault(fm, []).append(error)

                self._record_pair_error(pair, fm, z_v, z_r, error, distance_2d)

        # Step 5: 计算统计指标
        result = BlindTestResult()
        result.n_pairs = len(pairs)
        result.n_formations_evaluated = len(all_errors)

        if all_errors:
            errors_arr = np.array(all_errors)
            result._all_errors = errors_arr
            result.rmse = float(np.sqrt(np.mean(errors_arr ** 2)))
            result.mae = float(np.mean(errors_arr))
            result.max_error = float(np.max(errors_arr))

            # R^2 计算：虚拟 Z vs 真实 Z 的线性回归
            z_v_all = []
            z_r_all = []
            for pair in pairs:
                v_profile = pair["virtual"]
                r_profile = pair["real"]
                v_interfaces = v_profile["interfaces"]
                r_interfaces = r_profile["interfaces"]
                for fm in set(v_interfaces.keys()) & set(r_interfaces.keys()):
                    z_v_all.append(v_interfaces[fm])
                    z_r_all.append(r_interfaces[fm])

            result._z_v_all = z_v_all
            result._z_r_all = z_r_all
            if len(z_v_all) > 2:
                corr_matrix = np.corrcoef(z_v_all, z_r_all)
                result.r_squared = float(corr_matrix[0, 1] ** 2)

            for fm, errs in per_formation_errors.items():
                result.per_formation_rmse[fm] = float(np.sqrt(np.mean(np.array(errs) ** 2)))
                result.per_formation_mae[fm] = float(np.mean(errs))

        # Step 6: 评定等级
        result.grade = self._assign_grade(result.rmse, result.mae, result.r_squared)
        result.passed = result.grade in ("A", "B")

        return result

    def _detect_columns(self, df: pd.DataFrame, mode: str) -> Tuple[str, str, str, str, str]:
        """自动检测列名映射"""
        cols = df.columns.tolist()

        x_col = next((c for c in cols if c.lower() in ('x', 'x_m', 'lon', 'longitude', 'easting')), cols[0])
        y_col = next((c for c in cols if c.lower() in ('y', 'y_m', 'lat', 'latitude', 'northing')), cols[1] if len(cols) > 1 else cols[0])
        z_col = next((c for c in cols if c.lower() in ('z', 'z_m', 'elev', 'elevation', 'depth', '标高', '高程')), cols[2] if len(cols) > 2 else cols[0])
        fm_col = next((c for c in cols if c.lower() in ('formation_code', 'formation', 'fm_code', 'surface', '地层代号', '地层')), cols[3] if len(cols) > 3 else cols[0])
        bh_col = next((c for c in cols if c.lower() in ('borehole_id', 'bh_id', 'borehole', 'well_id', '钻孔编号', '钻孔')), None)

        return x_col, y_col, z_col, fm_col, bh_col

    def _extract_depth_profiles(self, df: pd.DataFrame, x_col: str, y_col: str,
                                 z_col: str, fm_col: str, bh_col: Optional[str],
                                 mode: str) -> List[Dict]:
        """
        从钻孔点云中提取每个钻孔的地层分界面深度序列

        对于同一个 (x, y) 位置（即同一个钻孔），多个 depth (z) 值代表不同的地层界面。
        每个界面由 formation_code 标识。返回每个钻孔的顶面 (x, y, z_top) 和
        各层底界深度 {formation_code: z_bottom}。
        """
        profiles = []

        if bh_col and bh_col in df.columns:
            group_col = bh_col
        else:
            # 降级：用 (x, y) 分组
            df = df.copy()
            df['_bh_key'] = df.apply(lambda r: f"{r[x_col]:.1f}_{r[y_col]:.1f}", axis=1)
            group_col = '_bh_key'

        for bh_id, group in df.groupby(group_col):
            if len(group) < 1:
                continue

            # 取第一行的 x, y 作为钻孔位置
            x_pos = float(group[x_col].iloc[0])
            y_pos = float(group[y_col].iloc[0])

            # 按 z 降序排列（从地表往下）
            group_sorted = group.sort_values(z_col, ascending=False)

            # 地表高程 = 最高点
            z_top = float(group_sorted[z_col].iloc[0])

            # 每层底界：最低的 z 值
            interfaces = {}
            for fm, sub in group_sorted.groupby(fm_col):
                z_bottom = float(sub[z_col].min())  # 该地层的最低点 = 底界
                interfaces[str(fm)] = z_bottom

            profiles.append({
                "borehole_id": str(bh_id),
                "x": x_pos,
                "y": y_pos,
                "z_top": z_top,
                "interfaces": interfaces,
                "formations": list(interfaces.keys()),
                "n_layers": len(interfaces),
            })

        return profiles

    def _match_boreholes(self, virtual_profiles: List[Dict],
                         real_profiles: List[Dict]) -> List[Dict]:
        """
        使用 KD-Tree 为每个真实钻孔匹配最近的虚拟钻孔
        """
        if not virtual_profiles or not real_profiles:
            return []

        v_coords = np.array([[p["x"], p["y"]] for p in virtual_profiles])
        r_coords = np.array([[p["x"], p["y"]] for p in real_profiles])

        tree = cKDTree(v_coords)
        pairs = []

        for i, r_profile in enumerate(real_profiles):
            dist, idx = tree.query(r_coords[i], k=1)
            if dist <= self.max_match_distance:
                pairs.append({
                    "real": r_profile,
                    "virtual": virtual_profiles[idx],
                    "distance_2d": float(dist),
                    "real_x": r_profile["x"],
                    "real_y": r_profile["y"],
                    "virtual_x": virtual_profiles[idx]["x"],
                    "virtual_y": virtual_profiles[idx]["y"],
                })

        return pairs

    def _record_pair_error(self, pair: Dict, fm: str, z_v: float, z_r: float,
                           error: float, distance_2d: float):
        """记录每对钻孔的详细误差（供后续分析）"""
        # 结果中稍后填充
        pass

    def _assign_grade(self, rmse: float, mae: float, r_squared: float) -> str:
        """
        根据 RMSE / MAE / R^2 综合评定等级

        地质建模行业参考标准:
            A (优):  RMSE < 5m  且 R^2 > 0.90
            B (良):  RMSE < 15m 且 R^2 > 0.75
            C (中):  RMSE < 30m 且 R^2 > 0.50
            D (差):  不满足上述任一
        """
        if rmse < 5 and r_squared > 0.90:
            return "A (优) — 深度推演精度极高，可替代部分勘探钻孔"
        elif rmse < 15 and r_squared > 0.75:
            return "B (良) — 深度推演可靠，可作辅助工程参考"
        elif rmse < 30 and r_squared > 0.50:
            return "C (中) — 深度推演具备参考价值，需结合其他资料"
        else:
            return "D (差) — 深度推演偏差较大，建议优化模型参数"

    def format_report(self, result: BlindTestResult) -> str:
        """生成可读的验证报告"""
        lines = [
            "=" * 60,
            "  真实钻孔盲测对比验证报告 (Blind Test Report)",
            "=" * 60,
            f"  匹配钻孔对数:       {result.n_pairs}",
            f"  评估地层界面数:      {result.n_formations_evaluated}",
            f"  RMSE (均方根误差):   {result.rmse:.3f} m",
            f"  MAE  (平均绝对误差): {result.mae:.3f} m",
            f"  最大绝对误差:        {result.max_error:.3f} m",
            f"  R^2   (决定系数):     {result.r_squared:.4f}",
            f"  验证结论:            {result.passed}",
            f"  质量等级:            {result.grade}",
            "-" * 60,
        ]

        if result.per_formation_rmse:
            lines.append("  各地层 RMSE (按地层代码):")
            for fm, rmse_val in sorted(result.per_formation_rmse.items()):
                mae_val = result.per_formation_mae.get(fm, float("nan"))
                lines.append(f"    {fm}: RMSE={rmse_val:.3f}m, MAE={mae_val:.3f}m")

        lines.append("=" * 60)
        return "\n".join(lines)


    def save_results(self, result: BlindTestResult, output_dir: str,
                      prefix: str = "blind_test") -> Dict[str, str]:
        """Save validation report and structured data to output directory."""
        os.makedirs(output_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        saved = {}

        report_path = os.path.join(output_dir, f"{prefix}_report_{timestamp}.txt")
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(self.format_report(result))
        saved["report"] = report_path

        data_dict = {
            "n_pairs": result.n_pairs,
            "n_formations_evaluated": result.n_formations_evaluated,
            "rmse": result.rmse,
            "mae": result.mae,
            "max_error": result.max_error,
            "r_squared": result.r_squared,
            "per_formation_rmse": result.per_formation_rmse,
            "per_formation_mae": result.per_formation_mae,
            "passed": result.passed,
            "grade": result.grade,
        }
        json_path = os.path.join(output_dir, f"{prefix}_data_{timestamp}.json")
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(data_dict, f, ensure_ascii=False, indent=2)
        saved["json"] = json_path

        return saved


def demo_validation():
    """
    演示盲测验证流程（使用模拟数据）
    实际使用时替换为真实的 CSV 路径
    """
    import tempfile

    tmpdir = tempfile.mkdtemp()
    print(f"[Demo] 使用临时目录: {tmpdir}")

    # 生成模拟虚拟钻孔数据
    np.random.seed(42)
    n_boreholes = 100
    n_formations = 5

    virtual_rows = []
    for bh_idx in range(n_boreholes):
        x = np.random.uniform(5000, 15000)
        y = np.random.uniform(5000, 15000)
        z_top = np.random.uniform(100, 500)

        # 从地表往下堆 5 层
        current_z = z_top
        for fm_idx in range(n_formations):
            thickness = np.random.uniform(20, 80)
            virtual_rows.append({
                "x": x, "y": y, "z": current_z,
                "formation_code": f"F{fm_idx + 1}",
                "value": 1, "surface": f"F{fm_idx + 1}",
                "borehole_id": f"BH{bh_idx:03d}"
            })
            current_z -= thickness

    df_virtual = pd.DataFrame(virtual_rows)
    virtual_path = os.path.join(tmpdir, "virtual_boreholes.csv")
    df_virtual.to_csv(virtual_path, index=False)

    # 生成模拟真实钻孔数据（在虚拟钻孔基础上加噪声）
    real_rows = []
    for bh_idx in range(5):  # 仅 5 个真实钻孔
        x = np.random.uniform(5000, 15000)
        y = np.random.uniform(5000, 15000)
        z_top = np.random.uniform(100, 500)

        current_z = z_top
        for fm_idx in range(n_formations):
            thickness = np.random.uniform(20, 80)
            # 加噪声模拟真实值与推演值的差异
            real_z = current_z + np.random.normal(0, 8)  # σ = 8m
            real_rows.append({
                "x": x, "y": y, "z": real_z,
                "formation_code": f"F{fm_idx + 1}",
                "borehole_id": f"REAL_{bh_idx:02d}"
            })
            current_z -= thickness

    df_real = pd.DataFrame(real_rows)
    real_path = os.path.join(tmpdir, "real_boreholes.csv")
    df_real.to_csv(real_path, index=False)

    # 执行验证
    validator = BlindTestValidator(max_match_distance=500.0)
    result = validator.run(virtual_path, real_path)

    print(validator.format_report(result))

    # Save to output/
    output_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "output")
    saved = validator.save_results(result, output_dir, prefix="blind_test_demo")
    print(f"\n[Demo] 报告已保存至: {saved['report']}")
    print(f"[Demo] 数据已保存至: {saved['json']}")
    return result


if __name__ == "__main__":
    demo_validation()
