"""
定量验证 1.2 — 专家地质剖面二维拓扑交并比 (Cross-Section Topology IoU)

原理:
    在"零真实钻孔"极端场景下，利用传统地质报告中附带的"人工手绘地质实测剖面图"作为
    参照基准。沿剖面走向从三维虚拟钻孔阵列中切出二维剖面，通过插值重构地层分界面曲线，
    与专家手绘剖面进行叠置分析。

评价指标:
    - IoU (交并比): 各地层多边形区域的重叠程度，IoU ∈ [0, 1]
    - Fréchet Distance: 地层分界曲线的空间形态相似度
    - Mean IoU: 所有地层的平均 IoU

适用场景:
    - 无真实钻孔可用
    - 拥有研究区地质报告中的手绘剖面图（A-A', B-B' 等）
    - 需要验证 LLM 推演的地层空间形态是否与人类专家判断一致

学术意义:
    证明多智能体推演出的空间三维形态在降维到二维时，与人类资深专家手绘推断保持
    高度拓扑一致性，验证算法"地质学意义上的正确性"。
"""

import os
import json
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field
from datetime import datetime
from scipy.interpolate import RBFInterpolator, griddata
from scipy.spatial import cKDTree
from shapely.geometry import Polygon, LineString, Point, box
from shapely.ops import unary_union
import warnings


@dataclass
class CrossSectionResult:
    """剖面拓扑验证结果"""
    section_name: str = ""                     # 剖面名称 (如 A-A')
    n_formations: int = 0                      # 参与评估的地层数
    mean_iou: float = 0.0                      # 平均 IoU
    per_formation_iou: Dict[str, float] = field(default_factory=dict)   # 各地层 IoU
    frechet_distances: Dict[str, float] = field(default_factory=dict)   # 各地层界线 Fréchet 距离
    overall_frechet: float = 0.0               # 整体 Fréchet 距离
    coverage: float = 0.0                      # 虚拟钻孔剖面对专家剖面的覆盖率
    grade: str = ""                            # 等级


class CrossSectionValidator:
    """
    专家地质剖面二维拓扑交并比验证器

    工作流程:
        1. 从三维虚拟钻孔阵列沿指定剖面线 (A-A') 提取二维切片
        2. 对切片点云进行克里金/最近邻插值，重构连续地层分界线
        3. 将重构剖面与专家手绘剖面（数字化为多边形）进行叠置
        4. 计算各地层的 IoU 和 Fréchet 距离

    输入格式:
        - 虚拟钻孔 CSV: x, y, z, formation_code, borehole_id
        - 剖面线: [(x1, y1), (x2, y2), ...] 定义剖面走向
        - 专家剖面多边形: GeoJSON 或 shapefile，每种地层一个 Polygon
    """

    def __init__(self,
                 section_line: List[Tuple[float, float]],
                 section_name: str = "A-A'",
                 swath_width: float = 100.0,
                 grid_resolution: float = 50.0):
        """
        Parameters
        ----------
        section_line : List[Tuple[float, float]]
            剖面线顶点序列 [(x1,y1), (x2,y2), ...]
        section_name : str
            剖面名称
        swath_width : float
            剖面切片的带宽 (m)，剖面线两侧各 swath_width/2 范围内的钻孔纳入切片
        grid_resolution : float
            二维插值网格分辨率 (m)
        """
        self.section_line = np.array(section_line)
        self.section_name = section_name
        self.swath_width = swath_width
        self.grid_resolution = grid_resolution

    def extract_section_from_boreholes(self,
                                        df_boreholes: pd.DataFrame,
                                        x_col: str = "x",
                                        y_col: str = "y",
                                        z_col: str = "z",
                                        fm_col: str = "formation_code") -> pd.DataFrame:
        """
        沿剖面线从三维钻孔点云中提取二维切片

        Returns
        -------
        pd.DataFrame with columns: distance_along_section, z, formation_code
        """
        points = df_boreholes[[x_col, y_col]].values
        z_vals = df_boreholes[z_col].values
        fm_vals = df_boreholes[fm_col].values

        # 计算每个点到剖面线的最近距离和沿剖面线的投影距离
        boreholes_in_swath = []
        for i in range(len(points)):
            pt = points[i]
            dist_to_line, proj_dist = self._point_to_polyline(pt, self.section_line)

            if dist_to_line <= self.swath_width / 2:
                boreholes_in_swath.append({
                    "distance_along_section": proj_dist,
                    "z": z_vals[i],
                    "formation_code": str(fm_vals[i]),
                    "x_orig": pt[0],
                    "y_orig": pt[1],
                    "dist_off_section": dist_to_line,
                })

        if not boreholes_in_swath:
            raise ValueError(f"剖面 {self.section_name} 带宽 {self.swath_width}m 内未找到任何虚拟钻孔！")

        return pd.DataFrame(boreholes_in_swath)

    def reconstruct_formation_boundaries(self,
                                          section_df: pd.DataFrame,
                                          x_range: Optional[Tuple[float, float]] = None,
                                          z_range: Optional[Tuple[float, float]] = None
                                          ) -> Dict[str, np.ndarray]:
        """
        通过插值重构各地层分界面的二维曲线

        使用 RBF (径向基函数) 插值对每种地层的底界深度进行空间插值

        Returns
        -------
        Dict[str, np.ndarray]: {formation_code: 2D_grid_of_bottom_z}
        """
        if x_range is None:
            d_min = section_df["distance_along_section"].min()
            d_max = section_df["distance_along_section"].max()
            x_range = (d_min, d_max)

        if z_range is None:
            z_min = section_df["z"].min()
            z_max = section_df["z"].max()
            z_range = (z_min, z_max)

        nx = int((x_range[1] - x_range[0]) / self.grid_resolution) + 1
        nz = int((z_range[1] - z_range[0]) / self.grid_resolution) + 1

        xi = np.linspace(x_range[0], x_range[1], nx)
        zi = np.linspace(z_range[0], z_range[1], nz)
        mesh_x, mesh_z = np.meshgrid(xi, zi)

        formations = section_df["formation_code"].unique()
        formation_grids = {}

        for fm in formations:
            fm_data = section_df[section_df["formation_code"] == fm]
            if len(fm_data) < 3:
                continue

            points = fm_data[["distance_along_section", "z"]].values
            # 用 1 表示该地层存在，0 表示不存在
            values = np.ones(len(points))

            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    grid = griddata(points, values, (mesh_x, mesh_z), method="linear", fill_value=0.0)
                formation_grids[fm] = grid
            except Exception:
                # 降级为最近邻
                grid = griddata(points, values, (mesh_x, mesh_z), method="nearest", fill_value=0.0)
                formation_grids[fm] = grid

        self._xi = xi
        self._zi = zi
        self._mesh_x = mesh_x
        self._mesh_z = mesh_z

        return formation_grids

    def compute_iou_with_expert(self,
                                 formation_grids: Dict[str, np.ndarray],
                                 expert_polygons: Dict[str, List[Polygon]],
                                 x_range: Tuple[float, float],
                                 z_range: Tuple[float, float]) -> CrossSectionResult:
        """
        将重构的二维地层网格与专家手绘剖面的地层多边形进行 IoU 计算

        Parameters
        ----------
        formation_grids : Dict[str, np.ndarray]
            各地层的插值网格 (formation_code -> 2D occupancy grid)
        expert_polygons : Dict[str, List[Polygon]]
            专家剖面的地层多边形 (formation_code -> list of shapely Polygons)
        x_range, z_range : Tuple[float, float]
            剖面的空间范围

        Returns
        -------
        CrossSectionResult
        """
        result = CrossSectionResult(section_name=self.section_name)
        result.n_formations = len(formation_grids)

        # 构建参考矩形 (剖面空间)
        ref_box = box(x_range[0], z_range[0], x_range[1], z_range[1])
        ref_area = ref_box.area

        all_ious = []
        all_frechet = []

        for fm, grid in formation_grids.items():
            if fm not in expert_polygons:
                continue

            # 从网格构建虚拟剖面的多边形
            virtual_poly = self._grid_to_polygon(grid, x_range, z_range)
            if virtual_poly is None or virtual_poly.is_empty:
                continue

            # 合并专家多边形的该地层区域
            expert_polys = expert_polygons[fm]
            expert_union = unary_union(expert_polys) if len(expert_polys) > 1 else expert_polys[0]

            if expert_union.is_empty:
                continue

            # IoU 计算
            try:
                intersection = virtual_poly.intersection(expert_union).area
                union = virtual_poly.union(expert_union).area
                iou = intersection / union if union > 0 else 0.0
            except Exception:
                iou = 0.0

            result.per_formation_iou[fm] = float(iou)
            all_ious.append(iou)

            # Fréchet 距离：比较地层边界线
            virtual_boundary = virtual_poly.boundary
            expert_boundary = expert_union.boundary
            fd = self._discrete_frechet_distance(virtual_boundary, expert_boundary)
            result.frechet_distances[fm] = float(fd)
            all_frechet.append(fd)

        if all_ious:
            result.mean_iou = float(np.mean(all_ious))
        if all_frechet:
            result.overall_frechet = float(np.mean(all_frechet))

        # 覆盖率
        if all_ious:
            result.coverage = len(all_ious) / len(expert_polygons) if expert_polygons else 1.0

        result.grade = self._assign_grade(result.mean_iou, result.overall_frechet)

        return result

    # ─── 辅助方法 ──────────────────────────────────────────

    def _point_to_polyline(self, pt: np.ndarray, polyline: np.ndarray) -> Tuple[float, float]:
        """计算点到折线的最近距离和投影距离"""
        min_dist = float("inf")
        cumulative_dist = 0.0
        best_proj = 0.0

        for i in range(len(polyline) - 1):
            seg_start = polyline[i]
            seg_end = polyline[i + 1]
            seg_vec = seg_end - seg_start
            seg_len = np.linalg.norm(seg_vec)

            if seg_len < 1e-10:
                dist = np.linalg.norm(pt - seg_start)
                if dist < min_dist:
                    min_dist = dist
                    best_proj = cumulative_dist
                continue

            seg_unit = seg_vec / seg_len
            vec_to_pt = pt - seg_start
            t = np.dot(vec_to_pt, seg_unit)
            t_clamped = max(0, min(seg_len, t))

            closest = seg_start + t_clamped * seg_unit
            dist = np.linalg.norm(pt - closest)

            if dist < min_dist:
                min_dist = dist
                best_proj = cumulative_dist + t_clamped

            cumulative_dist += seg_len

        return min_dist, best_proj

    def _grid_to_polygon(self, grid: np.ndarray,
                         x_range: Tuple[float, float],
                         z_range: Tuple[float, float]) -> Optional[Polygon]:
        """将二值网格转换为 Shapely 多边形"""
        from skimage import measure

        # 二值化
        binary = (grid > 0.5).astype(np.uint8)

        if binary.sum() < 4:
            return None

        try:
            contours = measure.find_contours(binary.astype(float), level=0.5)
            if not contours:
                return None

            # 取最大的轮廓
            largest = max(contours, key=len)

            # 将像素坐标映射回实际坐标
            ny, nx = grid.shape
            x_scale = (x_range[1] - x_range[0]) / nx
            z_scale = (z_range[1] - z_range[0]) / ny

            coords = [(x_range[0] + c[1] * x_scale,
                       z_range[0] + c[0] * z_scale) for c in largest]

            if len(coords) >= 3:
                return Polygon(coords)
        except ImportError:
            # skimage 不可用时的降级方案
            pass
        except Exception:
            pass

        return None

    def _discrete_frechet_distance(self, curve_a, curve_b) -> float:
        """
        计算两条曲线的离散 Fréchet 距离

        将 Shapely 几何对象采样为点序列后计算
        """
        # 采样曲线
        def sample_curve(geom, n_pts=100):
            if isinstance(geom, LineString):
                distances = np.linspace(0, geom.length, n_pts)
                pts = np.array([geom.interpolate(d).coords[0] for d in distances])
                return pts
            elif isinstance(geom, Polygon):
                boundary = geom.exterior
                distances = np.linspace(0, boundary.length, n_pts)
                pts = np.array([boundary.interpolate(d).coords[0] for d in distances])
                return pts
            else:
                # MultiLineString 等
                return np.zeros((0, 2))

        pts_a = sample_curve(curve_a)
        pts_b = sample_curve(curve_b)

        if len(pts_a) == 0 or len(pts_b) == 0:
            return float("inf")

        # 离散 Fréchet 距离的 DP 算法
        n, m = len(pts_a), len(pts_b)
        ca = np.full((n, m), np.inf)

        ca[0, 0] = np.linalg.norm(pts_a[0] - pts_b[0])

        for i in range(1, n):
            ca[i, 0] = max(ca[i - 1, 0], np.linalg.norm(pts_a[i] - pts_b[0]))
        for j in range(1, m):
            ca[0, j] = max(ca[0, j - 1], np.linalg.norm(pts_a[0] - pts_b[j]))

        for i in range(1, n):
            for j in range(1, m):
                d = np.linalg.norm(pts_a[i] - pts_b[j])
                ca[i, j] = max(min(ca[i - 1, j], ca[i, j - 1], ca[i - 1, j - 1]), d)

        return float(ca[n - 1, m - 1])

    def _assign_grade(self, mean_iou: float, frechet: float) -> str:
        """根据 IoU 和 Fréchet 距离评定等级"""
        if mean_iou >= 0.70:
            return "A (优) — 地层空间形态与专家剖面高度一致"
        elif mean_iou >= 0.50:
            return "B (良) — 地层空间形态与专家剖面基本吻合"
        elif mean_iou >= 0.30:
            return "C (中) — 部分地层形态与专家剖面一致，存在局部偏差"
        else:
            return "D (差) — 地层空间形态与专家剖面偏差较大"

    def format_report(self, result: CrossSectionResult) -> str:
        """格式化输出验证报告"""
        lines = [
            "=" * 60,
            f"  专家剖面拓扑交并比验证报告 (剖面 {result.section_name})",
            "=" * 60,
            f"  参与评估地层数:    {result.n_formations}",
            f"  平均 IoU:           {result.mean_iou:.4f}",
            f"  整体 Fréchet 距离:  {result.overall_frechet:.2f} m",
            f"  覆盖率:             {result.coverage:.2%}",
            f"  质量等级:           {result.grade}",
            "-" * 60,
        ]

        if result.per_formation_iou:
            lines.append("  各地层 IoU 与 Fréchet 距离:")
            for fm in sorted(result.per_formation_iou.keys()):
                iou = result.per_formation_iou[fm]
                fd = result.frechet_distances.get(fm, float("nan"))
                lines.append(f"    {fm}: IoU={iou:.4f}, Fréchet={fd:.2f} m")

        lines.append("=" * 60)
        return "\n".join(lines)


    def save_results(self, result: CrossSectionResult, output_dir: str,
                      prefix: str = "cross_section") -> Dict[str, str]:
        """Save validation report and structured data to output directory."""
        os.makedirs(output_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        saved = {}

        report_path = os.path.join(output_dir, f"{prefix}_report_{timestamp}.txt")
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(self.format_report(result))
        saved["report"] = report_path

        data_dict = {
            "section_name": result.section_name,
            "n_formations": result.n_formations,
            "mean_iou": result.mean_iou,
            "per_formation_iou": result.per_formation_iou,
            "frechet_distances": result.frechet_distances,
            "overall_frechet": result.overall_frechet,
            "coverage": result.coverage,
            "grade": result.grade,
        }
        json_path = os.path.join(output_dir, f"{prefix}_data_{timestamp}.json")
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(data_dict, f, ensure_ascii=False, indent=2)
        saved["json"] = json_path

        return saved


def demo_validation():
    """演示剖面拓扑验证流程（使用模拟数据）"""
    import tempfile
    import os

    tmpdir = tempfile.mkdtemp()
    print(f"[Demo] 使用临时目录: {tmpdir}")

    # 生成模拟虚拟钻孔数据
    np.random.seed(42)
    n_boreholes = 50
    section_line = [(5000, 5000), (15000, 15000)]  # 对角线剖面

    rows = []
    for i in range(n_boreholes):
        # 放置钻孔在剖面线附近
        t = np.random.uniform(0, 1)
        x = section_line[0][0] + t * (section_line[1][0] - section_line[0][0])
        y = section_line[0][1] + t * (section_line[1][1] - section_line[0][1])
        # 添加垂直剖面方向的偏移
        x += np.random.uniform(-50, 50)
        y += np.random.uniform(-50, 50)

        z_top = 300 - t * 200  # 地表沿剖面倾斜
        current_z = z_top
        for fm_idx, thick in enumerate([30, 50, 40, 60]):
            rows.append({
                "x": x, "y": y, "z": current_z,
                "formation_code": f"F{fm_idx + 1}",
                "borehole_id": f"BH_{i:03d}"
            })
            current_z -= thick + np.random.normal(0, 3)

    df_virtual = pd.DataFrame(rows)

    # 创建验证器并提取剖面
    validator = CrossSectionValidator(
        section_line=section_line,
        section_name="Demo A-A'",
        swath_width=100.0,
        grid_resolution=25.0,
    )

    section_df = validator.extract_section_from_boreholes(df_virtual)
    print(f"  从剖面带宽内提取了 {len(section_df)} 个钻孔点")

    grids = validator.reconstruct_formation_boundaries(section_df)
    print(f"  重构了 {len(grids)} 个地层的二维网格")

    # 模拟专家剖面多边形（简化为矩形测试）
    x_range = (section_df["distance_along_section"].min(),
               section_df["distance_along_section"].max())
    z_range = (section_df["z"].min(), section_df["z"].max())

    from shapely.geometry import box as shapely_box
    expert_polygons = {}
    formations = sorted(section_df["formation_code"].unique())
    z_step = (z_range[1] - z_range[0]) / len(formations)

    for i, fm in enumerate(formations):
        z_top_fm = z_range[1] - i * z_step
        z_bot_fm = z_top_fm - z_step
        expert_polygons[fm] = [shapely_box(x_range[0], z_bot_fm, x_range[1], z_top_fm)]

    result = validator.compute_iou_with_expert(grids, expert_polygons, x_range, z_range)
    print(validator.format_report(result))

    # Save to output/
    output_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "output")
    saved = validator.save_results(result, output_dir, prefix="cross_section_demo")
    print(f"\n[Demo] 报告已保存至: {saved['report']}")
    print(f"[Demo] 数据已保存至: {saved['json']}")
    return result


if __name__ == "__main__":
    demo_validation()
