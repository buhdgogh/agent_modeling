"""
定性验证 2.2 — 多模态空间自洽性视觉评估 (Visual Spatial Consistency)

原理:
    通过数值计算替代（或辅助）人工目视判读，对虚拟钻孔与 DEM、地质底图的
    空间关系进行定量化自洽性检查。

评估维度:
    1. 孔口贴合度 (Orifice Fit):
       检查所有虚拟钻孔的最高点（孔口）是否贴合 DEM 地表网格。
       计算孔口高程与 DEM 采样值之间的偏差分布。

    2. 地层-地质图一致性 (Formation-Map Consistency):
       检查每个钻孔顶层的地层代码是否与 SHP 面文件中对应位置的地质单元属性一致。
       计算顶层匹配率。

    3. 地层起伏自然度 (Smoothness of Stratal Undulation):
       检查深部地层界面在空间上的起伏是否平滑连续。
       使用局部曲率分析检测异常的阶梯状断裂或噪点。

输出:
    - 孔口偏差统计 (均值、标准差、最大偏差)
    - 顶层地层匹配率
    - 地表起伏异常点列表
    - 综合自洽性评分

注意:
    此模块是对"人工专家打分制"的数值化补充——将主观目视判据转化为可重复的
    定量指标，但不完全替代人工判断。
"""

import os
import json
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field
from datetime import datetime
from scipy.spatial import cKDTree
from scipy.ndimage import uniform_filter
import warnings

try:
    import rasterio
    HAS_RASTERIO = True
except ImportError:
    HAS_RASTERIO = False

try:
    import geopandas as gpd
    HAS_GEOPANDAS = True
except ImportError:
    HAS_GEOPANDAS = False


@dataclass
class VisualConsistencyResult:
    """空间自洽性评估结果"""
    # 孔口贴合度
    n_orifices_checked: int = 0
    orifice_z_mean_error: float = 0.0           # 平均高程偏差 (m)
    orifice_z_std_error: float = 0.0            # 高程偏差标准差 (m)
    orifice_z_max_error: float = 0.0            # 最大偏差 (m)
    orifice_pass_rate: float = 0.0              # 孔口合格率 (偏差 < 阈值)
    n_floating_orifices: int = 0                 # 悬空孔口数 (高于DEM)
    n_buried_orifices: int = 0                   # 深埋孔口数 (低于DEM)

    # 地层-地质图一致性
    n_top_matched: int = 0
    n_top_total: int = 0
    top_formation_match_rate: float = 0.0        # 顶层地层匹配率

    # 地层起伏自然度
    n_anomalous_points: int = 0                  # 异常起伏点数
    smoothness_score: float = 0.0                # 平滑度评分 (0~1)
    anomalous_regions: List[Dict] = field(default_factory=list)

    # 综合
    overall_score: float = 0.0                   # 综合自洽性评分 (0~100)
    passed: bool = False
    grade: str = ""


class VisualConsistencyValidator:
    """
    多模态空间自洽性验证器

    使用流程:
        validator = VisualConsistencyValidator()
        result = validator.run(virtual_csv, dem_path=dem_path, shp_path=shp_path)
    """

    def __init__(self,
                 orifice_z_tolerance: float = 10.0,
                 curvature_threshold: float = 0.01,
                 outlier_std_threshold: float = 3.0):
        """
        Parameters
        ----------
        orifice_z_tolerance : float
            孔口高程偏差容忍阈值 (m)
        curvature_threshold : float
            地层界面局部曲率异常阈值
        outlier_std_threshold : float
            离群点标准差倍数阈值
        """
        self.orifice_z_tolerance = orifice_z_tolerance
        self.curvature_threshold = curvature_threshold
        self.outlier_std_threshold = outlier_std_threshold

    def run(self,
            virtual_csv: str,
            dem_path: Optional[str] = None,
            shp_path: Optional[str] = None) -> VisualConsistencyResult:
        """
        执行多模态空间自洽性评估

        Parameters
        ----------
        virtual_csv : str
            虚拟钻孔点云 CSV
        dem_path : str, optional
            DEM 栅格文件路径 (.tif)
        shp_path : str, optional
            地质图面文件路径 (.shp)

        Returns
        -------
        VisualConsistencyResult
        """
        df = pd.read_csv(virtual_csv)

        # 列检测
        x_col = self._detect_col(df, ['x', 'x_m', 'lon', 'easting'])
        y_col = self._detect_col(df, ['y', 'y_m', 'lat', 'northing'])
        z_col = self._detect_col(df, ['z', 'z_m', 'elev', 'elevation'])
        fm_col = self._detect_col(df, ['formation_code', 'surface'])
        bh_col = self._detect_col(df, ['borehole_id', 'bh_id'])

        result = VisualConsistencyResult()

        # ========================================
        # 评估 1: 孔口贴合度
        # ========================================
        if dem_path and os.path.exists(dem_path) and HAS_RASTERIO:
            orifice_result = self._check_orifice_fit(df, dem_path, x_col, y_col, z_col, bh_col)
            result.n_orifices_checked = orifice_result["n_checked"]
            result.orifice_z_mean_error = orifice_result["mean_error"]
            result.orifice_z_std_error = orifice_result["std_error"]
            result.orifice_z_max_error = orifice_result["max_error"]
            result.orifice_pass_rate = orifice_result["pass_rate"]
            result.n_floating_orifices = orifice_result["n_floating"]
            result.n_buried_orifices = orifice_result["n_buried"]
        else:
            # 无 DEM 时用钻孔自身的 z_top 变异度近似评估
            result = self._check_orifice_fit_fallback(df, x_col, y_col, z_col, bh_col, result)

        # ========================================
        # 评估 2: 地层-地质图一致性
        # ========================================
        if shp_path and os.path.exists(shp_path) and HAS_GEOPANDAS:
            top_result = self._check_formation_map_consistency(df, shp_path, x_col, y_col, fm_col, bh_col)
            result.n_top_matched = top_result["n_matched"]
            result.n_top_total = top_result["n_total"]
            result.top_formation_match_rate = top_result["match_rate"]

        # ========================================
        # 评估 3: 地层起伏自然度
        # ========================================
        smoothness_result = self._check_stratal_smoothness(df, x_col, y_col, z_col, fm_col, bh_col)
        result.n_anomalous_points = smoothness_result["n_anomalous"]
        result.smoothness_score = smoothness_result["score"]
        result.anomalous_regions = smoothness_result["details"]

        # ========================================
        # 综合评分
        # ========================================
        scores = []
        weights = []

        if result.n_orifices_checked > 0:
            scores.append(result.orifice_pass_rate * 100)
            weights.append(0.35)

        if result.n_top_total > 0:
            scores.append(result.top_formation_match_rate * 100)
            weights.append(0.35)

        scores.append(result.smoothness_score * 100)
        weights.append(0.30)

        if weights:
            total_weight = sum(weights)
            result.overall_score = sum(s * w / total_weight for s, w in zip(scores, weights))

        result.grade = self._assign_grade(result.overall_score)
        result.passed = result.overall_score >= 70

        return result

    # ─── 孔口贴合度 ───────────────────────────────────────

    def _check_orifice_fit(self, df: pd.DataFrame, dem_path: str,
                           x_col: str, y_col: str, z_col: str,
                           bh_col: str) -> Dict:
        """基于 DEM 的孔口贴合度检查"""
        with rasterio.open(dem_path) as src:
            dem_crs = src.crs
            nodata = src.nodata

            # 获取每个钻孔的孔口 (z 最大值)
            if bh_col in df.columns:
                orifices = df.groupby(bh_col).agg({
                    x_col: 'first',
                    y_col: 'first',
                    z_col: 'max'
                }).reset_index()
            else:
                df_copy = df.copy()
                df_copy['_bh_key'] = df_copy.apply(
                    lambda r: f"{r[x_col]:.1f}_{r[y_col]:.1f}", axis=1)
                orifices = df_copy.groupby('_bh_key').agg({
                    x_col: 'first',
                    y_col: 'first',
                    z_col: 'max'
                }).reset_index()

            # 采样 DEM
            xs = orifices[x_col].values
            ys = orifices[y_col].values
            z_virtual = orifices[z_col].values

            # 坐标转换 (假设虚拟钻孔在 EPSG:4326, DEM 可能在投影坐标)
            from pyproj import Transformer
            try:
                transformer = Transformer.from_crs("EPSG:4326", dem_crs, always_xy=True)
                xs_dem, ys_dem = transformer.transform(xs, ys)
            except Exception:
                xs_dem, ys_dem = xs, ys

            z_dem = []
            coords = list(zip(xs_dem, ys_dem))
            for val in src.sample(coords):
                v = float(val[0])
                if np.isnan(v) or (nodata is not None and np.isclose(v, nodata)):
                    z_dem.append(np.nan)
                else:
                    z_dem.append(v)

            z_dem = np.array(z_dem)

            # 计算偏差
            valid = np.isfinite(z_dem)
            errors = z_virtual[valid] - z_dem[valid]

            n_checked = valid.sum()
            if n_checked == 0:
                return {"n_checked": 0, "mean_error": 0, "std_error": 0,
                        "max_error": 0, "pass_rate": 0, "n_floating": 0, "n_buried": 0}

            mean_error = float(np.mean(errors))
            std_error = float(np.std(errors))
            max_error = float(np.max(np.abs(errors)))

            n_pass = (np.abs(errors) <= self.orifice_z_tolerance).sum()
            pass_rate = float(n_pass / n_checked)

            n_floating = (errors > self.orifice_z_tolerance).sum()
            n_buried = (errors < -self.orifice_z_tolerance).sum()

            return {
                "n_checked": int(n_checked),
                "mean_error": mean_error,
                "std_error": std_error,
                "max_error": max_error,
                "pass_rate": pass_rate,
                "n_floating": int(n_floating),
                "n_buried": int(n_buried),
            }

    def _check_orifice_fit_fallback(self, df: pd.DataFrame,
                                     x_col: str, y_col: str, z_col: str,
                                     bh_col: str, result: VisualConsistencyResult
                                     ) -> VisualConsistencyResult:
        """无 DEM 时的降级评估: 检查相邻钻孔孔口高程的连续性"""
        # 获取孔口
        if bh_col in df.columns:
            orifices = df.groupby(bh_col).agg({
                x_col: 'first', y_col: 'first', z_col: 'max'
            }).reset_index()
        else:
            df['_bh_key'] = df.apply(lambda r: f"{r[x_col]:.1f}_{r[y_col]:.1f}", axis=1)
            orifices = df.groupby('_bh_key').agg({
                x_col: 'first', y_col: 'first', z_col: 'max'
            }).reset_index()

        result.n_orifices_checked = len(orifices)

        # 检查相邻钻孔的高程差是否超出正常地形坡度
        coords = orifices[[x_col, y_col]].values
        z_vals = orifices[z_col].values

        if len(coords) >= 2:
            tree = cKDTree(coords)
            errors = []
            for i in range(len(coords)):
                dists, idxs = tree.query(coords[i], k=min(5, len(coords)))
                for j, d in zip(idxs, dists):
                    if j == i or d < 1e-6:
                        continue
                    dz = abs(z_vals[i] - z_vals[j])
                    # 正常地形坡度: 30° → tan(30°) ≈ 0.577
                    max_dz = d * 0.577
                    if dz > max_dz:
                        errors.append(dz - max_dz)
                        break

            result.orifice_z_mean_error = float(np.mean(errors)) if errors else 0.0
            result.orifice_z_std_error = float(np.std(errors)) if errors else 0.0
            result.orifice_z_max_error = float(np.max(errors)) if errors else 0.0
            result.orifice_pass_rate = 1.0 - min(len(errors) / len(orifices), 1.0)
            result.n_floating_orifices = len(errors)
            result.n_buried_orifices = 0  # 无法区分悬空/深埋
        else:
            result.orifice_pass_rate = 1.0

        return result

    # ─── 地层-地质图一致性 ─────────────────────────────────

    def _check_formation_map_consistency(self, df: pd.DataFrame, shp_path: str,
                                          x_col: str, y_col: str, fm_col: str,
                                          bh_col: str) -> Dict:
        """检查钻孔顶层地层与 SHP 地质图属性的一致性"""
        gdf = gpd.read_file(shp_path)

        # 获取每个钻孔的顶层地层
        if bh_col in df.columns:
            tops = df.sort_values(z_col, ascending=False).groupby(bh_col).first().reset_index()
        else:
            df['_bh_key'] = df.apply(lambda r: f"{r[x_col]:.1f}_{r[y_col]:.1f}", axis=1)
            tops = df.sort_values(z_col, ascending=False).groupby('_bh_key').first().reset_index()

        # 尝试匹配 SHP 属性中的地层字段
        shp_formation_col = None
        for candidate in ['formation', 'fm_code', 'formation_code', 'UNIT_NAME', 'GEO_UNIT',
                          '地层代号', '地质单元', 'NAME']:
            if candidate in gdf.columns:
                shp_formation_col = candidate
                break

        n_matched = 0
        n_total = 0

        for _, row in tops.iterrows():
            point = (row[x_col], row[y_col])
            top_fm = str(row[fm_col])

            # 查找包含该点的多边形
            for geom_idx, geom in enumerate(gdf.geometry):
                if geom.contains(gpd.points_from_xy([point[0]], [point[1]])[0]):
                    n_total += 1
                    if shp_formation_col:
                        shp_fm = str(gdf.iloc[geom_idx][shp_formation_col])
                        if top_fm.lower() == shp_fm.lower() or top_fm in shp_fm or shp_fm in top_fm:
                            n_matched += 1
                    break

        match_rate = n_matched / n_total if n_total > 0 else 0.0

        return {
            "n_matched": n_matched,
            "n_total": n_total,
            "match_rate": match_rate,
        }

    # ─── 地层起伏自然度 ────────────────────────────────────

    def _check_stratal_smoothness(self, df: pd.DataFrame,
                                   x_col: str, y_col: str, z_col: str,
                                   fm_col: str, bh_col: str) -> Dict:
        """
        检查地层界面在空间上的起伏平滑度

        方法: 对每种地层，构建其底界面的高程曲面，计算局部曲率。
        曲率异常大的点标记为异常起伏。
        """
        df_copy = df.copy()

        # 对每个钻孔，获取该钻孔中每种地层的底界深度
        # 使用 (borehole, formation) 分组取最小 z (底界)
        if bh_col not in df_copy.columns:
            df_copy['_bh_key'] = df_copy.apply(
                lambda r: f"{r[x_col]:.1f}_{r[y_col]:.1f}", axis=1)
            bh_col = '_bh_key'

        bottom_interfaces = df_copy.groupby([bh_col, fm_col])[z_col].min().reset_index()
        bottom_interfaces = bottom_interfaces.merge(
            df_copy.groupby(bh_col)[[x_col, y_col]].first().reset_index(),
            on=bh_col, how='left'
        )

        formations = bottom_interfaces[fm_col].unique()
        n_anomalous_total = 0
        all_scores = []
        anomalous_details = []

        for fm in formations:
            fm_data = bottom_interfaces[bottom_interfaces[fm_col] == fm]
            if len(fm_data) < 5:
                continue

            pts = fm_data[[x_col, y_col]].values
            z_vals = fm_data[z_col].values

            # 用局部邻域的 z 标准差检测异常
            if len(pts) < 2:
                continue

            tree = cKDTree(pts)
            local_std = []

            for i in range(len(pts)):
                neighbors = tree.query_ball_point(pts[i], r=200.0)  # 200m 邻域
                if len(neighbors) >= 3:
                    local_std.append(np.std(z_vals[neighbors]))
                else:
                    local_std.append(np.nan)

            local_std = np.array(local_std)
            valid_std = local_std[np.isfinite(local_std)]

            if len(valid_std) == 0:
                continue

            global_std = np.std(z_vals)
            mean_local_std = np.mean(valid_std)

            # 平滑度评分: 局部变异 / 全局变异
            if global_std > 0:
                smoothness = 1.0 - min(mean_local_std / (global_std * 2), 1.0)
            else:
                smoothness = 1.0
            all_scores.append(smoothness)

            # 检测异常点
            threshold = np.nanmean(local_std) + self.outlier_std_threshold * np.nanstd(local_std)
            for i in range(len(local_std)):
                if np.isfinite(local_std[i]) and local_std[i] > threshold:
                    n_anomalous_total += 1
                    anomalous_details.append({
                        "formation": str(fm),
                        "x": float(pts[i][0]),
                        "y": float(pts[i][1]),
                        "z": float(z_vals[i]),
                        "local_std": float(local_std[i]),
                        "threshold": float(threshold),
                    })

        smoothness_score = float(np.mean(all_scores)) if all_scores else 1.0

        return {
            "n_anomalous": n_anomalous_total,
            "score": smoothness_score,
            "details": anomalous_details[:50],  # 限50条
        }

    # ─── 辅助方法 ──────────────────────────────────────────

    def _detect_col(self, df: pd.DataFrame, candidates: List[str]) -> str:
        for c in candidates:
            if c in df.columns:
                return c
        return df.columns[0]

    def _assign_grade(self, score: float) -> str:
        if score >= 90:
            return "A (优) — 虚拟钻孔与 DEM/地质图高度自洽，空间形态可信"
        elif score >= 75:
            return "B (良) — 虚拟钻孔与 DEM/地质图基本自洽，偶有局部偏差"
        elif score >= 60:
            return "C (中) — 存在一定空间不自洽，建议检查区域边界或 DEM 精度"
        else:
            return "D (差) — 空间自洽性较差，需排查数据源或模型参数"

    def format_report(self, result: VisualConsistencyResult) -> str:
        """格式化输出评估报告"""
        lines = [
            "=" * 60,
            "  多模态空间自洽性评估报告",
            "=" * 60,
            f"  ─── 孔口贴合度 ───",
            f"  检查钻孔数:          {result.n_orifices_checked}",
            f"  平均高程偏差:         {result.orifice_z_mean_error:.2f} m",
            f"  偏差标准差:           {result.orifice_z_std_error:.2f} m",
            f"  最大偏差:             {result.orifice_z_max_error:.2f} m",
            f"  孔口合格率:           {result.orifice_pass_rate:.2%}",
            f"  悬空孔口:             {result.n_floating_orifices}",
            f"  深埋孔口:             {result.n_buried_orifices}",
            f"  ─── 地层-地质图一致性 ───",
            f"  匹配钻孔数:          {result.n_top_matched} / {result.n_top_total}",
            f"  顶层匹配率:           {result.top_formation_match_rate:.2%}",
            f"  ─── 地层起伏自然度 ───",
            f"  异常起伏点数:         {result.n_anomalous_points}",
            f"  平滑度评分:           {result.smoothness_score:.4f} (0~1)",
            f"  ─── 综合 ───",
            f"  综合自洽性评分:       {result.overall_score:.1f} / 100",
            f"  通过验证:             {result.passed}",
            f"  质量等级:             {result.grade}",
            "=" * 60,
        ]
        return "\n".join(lines)


    def save_results(self, result: VisualConsistencyResult, output_dir: str,
                      prefix: str = "visual_consistency") -> Dict[str, str]:
        """Save validation report, structured data, and anomaly details to output directory."""
        os.makedirs(output_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        saved = {}

        report_path = os.path.join(output_dir, f"{prefix}_report_{timestamp}.txt")
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(self.format_report(result))
        saved["report"] = report_path

        data_dict = {
            "n_orifices_checked": result.n_orifices_checked,
            "orifice_z_mean_error": result.orifice_z_mean_error,
            "orifice_z_std_error": result.orifice_z_std_error,
            "orifice_z_max_error": result.orifice_z_max_error,
            "orifice_pass_rate": result.orifice_pass_rate,
            "n_floating_orifices": result.n_floating_orifices,
            "n_buried_orifices": result.n_buried_orifices,
            "n_top_matched": result.n_top_matched,
            "n_top_total": result.n_top_total,
            "top_formation_match_rate": result.top_formation_match_rate,
            "n_anomalous_points": result.n_anomalous_points,
            "smoothness_score": result.smoothness_score,
            "anomalous_regions": result.anomalous_regions[:50],
            "overall_score": result.overall_score,
            "passed": result.passed,
            "grade": result.grade,
        }
        json_path = os.path.join(output_dir, f"{prefix}_data_{timestamp}.json")
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(data_dict, f, ensure_ascii=False, indent=2, default=str)
        saved["json"] = json_path

        # Save anomaly regions CSV
        if result.anomalous_regions:
            anom_csv = os.path.join(output_dir, f"{prefix}_anomalies_{timestamp}.csv")
            pd.DataFrame(result.anomalous_regions).to_csv(anom_csv, index=False, encoding="utf-8-sig")
            saved["anomalies_csv"] = anom_csv

        return saved


def demo_validation():
    """演示空间自洽性评估流程（使用模拟数据）"""
    import tempfile

    tmpdir = tempfile.mkdtemp()
    print(f"[Demo] 使用临时目录: {tmpdir}")

    # 生成模拟虚拟钻孔数据
    np.random.seed(42)
    n_boreholes = 150
    rows = []

    for bh_idx in range(n_boreholes):
        x = np.random.uniform(0, 10000)
        y = np.random.uniform(0, 10000)
        z_top = 300 + 100 * np.sin(x / 5000 * np.pi) * np.cos(y / 5000 * np.pi)  # 模拟起伏地形

        current_z = z_top
        for fm_idx in range(5):
            thickness = np.random.uniform(25, 70)
            rows.append({
                "x": x, "y": y, "z": current_z,
                "formation_code": f"F{fm_idx + 1}",
                "surface": f"F{fm_idx + 1}",
                "borehole_id": f"BH{bh_idx:03d}"
            })
            current_z -= thickness

    df_virtual = pd.DataFrame(rows)
    csv_path = os.path.join(tmpdir, "virtual_boreholes.csv")
    df_virtual.to_csv(csv_path, index=False)

    # 执行验证 (无 DEM/SHP 时使用降级方案)
    validator = VisualConsistencyValidator()
    result = validator.run(csv_path)

    print(validator.format_report(result))

    # Save to output/
    output_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "output")
    saved = validator.save_results(result, output_dir, prefix="visual_consistency_demo")
    print(f"\n[Demo] 报告已保存至: {saved['report']}")
    print(f"[Demo] 数据已保存至: {saved['json']}")
    if "anomalies_csv" in saved:
        print(f"[Demo] 异常起伏详情: {saved['anomalies_csv']}")
    return result


if __name__ == "__main__":
    demo_validation()
