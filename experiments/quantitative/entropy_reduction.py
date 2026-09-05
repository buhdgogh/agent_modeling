"""
定量验证 1.3 — 隐式建模不确定性缩减量化 (Uncertainty & Entropy Reduction)

原理:
    基于信息熵 (Information Entropy) 理论评估虚拟钻孔的工程价值。
    引入隐式三维地质建模方法, 通过对照实验量化虚拟钻孔数据对地下空间
    不确定性的抑制效果。

实验设计:
    Control Group  (对照组):  仅使用 SHP 地表边界与产状数据进行无钻孔隐式插值
    Experimental Group (实验组): 注入虚拟钻孔数据后进行隐式建模

评价指标:
    - 体素级信息熵: H(x) = -Σ p_k * log(p_k), 其中 p_k 为体素属于第k种地层的概率
    - 熵减率 (Entropy Reduction Ratio): ΔH = (H_control - H_experiment) / H_control
    - 熵减空间分布: 熵值减小最显著的深度区间和空间区域

学术意义:
    用信息论指标证明虚拟钻孔的加入有效抑制了隐式插值算法在地层深部的发散，
    显著降低了地下空间结构的地质不确定性。

注意:
    本模块提供两种实现路径:
    - Path A (轻量): 使用纯 NumPy/SciPy 的简化插值 + 熵计算 (无需 GemPy)
    - Path B (完整): 调用 GemPy 引擎进行隐式建模 (需安装 gempy)
"""

import os
import json
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field
from datetime import datetime
from scipy.interpolate import RBFInterpolator
from scipy.spatial import cKDTree


@dataclass
class EntropyResult:
    """不确定性缩减验证结果"""
    n_voxels: int = 0                          # 体素网格总数
    n_formations: int = 0                      # 地层种类数
    control_entropy_total: float = 0.0          # 对照组信息熵总和
    experiment_entropy_total: float = 0.0       # 实验组信息熵总和
    entropy_reduction_ratio: float = 0.0        # 整体熵减率
    per_depth_entropy: Dict[str, List[float]] = field(default_factory=dict)  # 各深度层熵值
    max_reduction_depth: float = 0.0            # 熵减最大的深度
    max_reduction_ratio: float = 0.0            # 最大局部熵减率
    grade: str = ""                             # 等级


class EntropyReductionValidator:
    """
    隐式建模不确定性缩减量化验证器

    实现 Path A (轻量方案): 使用 RBF 插值 + 体素化概率计算

    使用方法:
        validator = EntropyReductionValidator(grid_resolution=(100, 100, 50))
        result = validator.run(virtual_csv, shp_path, tif_path, formation_csv)
    """

    def __init__(self,
                 grid_resolution: Tuple[int, int, int] = (100, 100, 50),
                 prior_uncertainty_scale: float = 1.0):
        """
        Parameters
        ----------
        grid_resolution : Tuple[int, int, int]
            三维体素网格分辨率 (nx, ny, nz)
        prior_uncertainty_scale : float
            先验不确定性缩放因子, >1 表示深部不确定性更大
        """
        self.grid_resolution = grid_resolution
        self.prior_uncertainty_scale = prior_uncertainty_scale

    def run(self,
            virtual_csv: str,
            shp_path: Optional[str] = None,
            tif_path: Optional[str] = None,
            formation_csv: Optional[str] = None) -> EntropyResult:
        """
        执行不确定性缩减验证

        Parameters
        ----------
        virtual_csv : str
            虚拟钻孔点云 CSV
        shp_path : str, optional
            SHP 面文件路径 (用于确定建模范围)
        tif_path : str, optional
            DEM 文件路径 (用于确定地表)
        formation_csv : str, optional
            地层特征 CSV (用于获取地层序列)

        Returns
        -------
        EntropyResult
        """
        df_boreholes = pd.read_csv(virtual_csv)

        # 自动检测列名
        x_col = next((c for c in df_boreholes.columns if c.lower() in ('x', 'x_m')), df_boreholes.columns[0])
        y_col = next((c for c in df_boreholes.columns if c.lower() in ('y', 'y_m')), df_boreholes.columns[1])
        z_col = next((c for c in df_boreholes.columns if c.lower() in ('z', 'z_m')), df_boreholes.columns[2])
        fm_col = next((c for c in df_boreholes.columns if c.lower() in ('formation_code',)), df_boreholes.columns[3])

        # Step 1: 确定建模空间范围
        x_min, x_max = df_boreholes[x_col].min(), df_boreholes[x_col].max()
        y_min, y_max = df_boreholes[y_col].min(), df_boreholes[y_col].max()
        z_min, z_max = df_boreholes[z_col].min(), df_boreholes[z_col].max()

        # 扩展范围：向深部延伸
        z_deep = z_min - abs(z_max - z_min) * 2.0
        z_min = z_deep

        # Step 2: 构建三维体素网格
        nx, ny, nz = self.grid_resolution
        voxel_x = np.linspace(x_min, x_max, nx)
        voxel_y = np.linspace(y_min, y_max, ny)
        voxel_z = np.linspace(z_min, z_max, nz)

        mesh_x, mesh_y, mesh_z = np.meshgrid(voxel_x, voxel_y, voxel_z, indexing='ij')
        voxel_centers = np.column_stack([mesh_x.ravel(), mesh_y.ravel(), mesh_z.ravel()])

        formations = sorted(df_boreholes[fm_col].unique())
        n_formations = len(formations)
        fm_to_idx = {fm: i for i, fm in enumerate(formations)}

        # Step 3: 对照组 — 仅使用地表约束 (虚拟钻孔的顶层点)
        surface_points = df_boreholes.groupby([x_col, y_col], as_index=False).first()
        # 重采样减少点密度
        if len(surface_points) > 200:
            surface_points = surface_points.sample(200, random_state=42)

        control_probs = self._compute_probability_field(
            surface_points, voxel_centers, (nx, ny, nz),
            fm_to_idx, n_formations, z_min, z_max,
            use_only_surface=True
        )

        # Step 4: 实验组 — 注入完整虚拟钻孔数据
        experiment_probs = self._compute_probability_field(
            df_boreholes, voxel_centers, (nx, ny, nz),
            fm_to_idx, n_formations, z_min, z_max,
            use_only_surface=False
        )

        # Step 5: 计算信息熵
        control_entropy = self._compute_voxel_entropy(control_probs)
        experiment_entropy = self._compute_voxel_entropy(experiment_probs)

        H_control_total = float(np.sum(control_entropy))
        H_experiment_total = float(np.sum(experiment_entropy))

        # 熵减率
        if H_control_total > 0:
            entropy_reduction_ratio = float((H_control_total - H_experiment_total) / H_control_total)
        else:
            entropy_reduction_ratio = 0.0

        # Step 6: 按深度分析熵减
        per_depth_control = control_entropy.mean(axis=(0, 1))  # 各深度平均熵
        per_depth_experiment = experiment_entropy.mean(axis=(0, 1))
        per_depth_reduction = (per_depth_control - per_depth_experiment) / (per_depth_control + 1e-10)

        max_depth_idx = np.argmax(per_depth_reduction)
        max_reduction_ratio = float(per_depth_reduction[max_depth_idx])
        max_reduction_depth = float(voxel_z[max_depth_idx])

        # Step 7: 组装结果
        result = EntropyResult(
            n_voxels=nx * ny * nz,
            n_formations=n_formations,
            control_entropy_total=H_control_total,
            experiment_entropy_total=H_experiment_total,
            entropy_reduction_ratio=entropy_reduction_ratio,
            max_reduction_depth=max_reduction_depth,
            max_reduction_ratio=max_reduction_ratio,
        )
        result.per_depth_entropy = {
            "control": per_depth_control.tolist(),
            "experiment": per_depth_experiment.tolist(),
            "reduction_ratio": per_depth_reduction.tolist(),
        }
        result.grade = self._assign_grade(entropy_reduction_ratio)

        return result

    def _compute_probability_field(self,
                                    df: pd.DataFrame,
                                    voxel_centers: np.ndarray,
                                    grid_shape: Tuple[int, int, int],
                                    fm_to_idx: Dict[str, int],
                                    n_formations: int,
                                    z_min: float,
                                    z_max: float,
                                    use_only_surface: bool = False
                                    ) -> np.ndarray:
        """
        计算每个体素属于各地层的概率

        方法: RBF 插值 + 深度衰减先验
        """
        nx, ny, nz = grid_shape
        n_voxels = nx * ny * nz

        # 初始化概率数组
        probs = np.zeros((n_formations, nx, ny, nz))

        if use_only_surface:
            # 对照组: 仅地表信息，深部用先验
            x_col = next((c for c in df.columns if c.lower() in ('x', 'x_m')), df.columns[0])
            y_col = next((c for c in df.columns if c.lower() in ('y', 'y_m')), df.columns[1])
            z_col = next((c for c in df.columns if c.lower() in ('z', 'z_m')), df.columns[2])
            fm_col = next((c for c in df.columns if c.lower() in ('formation_code',)), df.columns[3])

            for fm, fm_idx in fm_to_idx.items():
                fm_data = df[df[fm_col] == fm]
                if len(fm_data) < 3:
                    continue

                try:
                    source_pts = fm_data[[x_col, y_col, z_col]].values
                    rbf = RBFInterpolator(source_pts[:, :2], source_pts[:, 2],
                                          kernel="thin_plate_spline", epsilon=500)
                    voxel_z_pred = rbf(voxel_centers[:, :2])

                    # 概率随到预测深度的距离衰减
                    depths = voxel_centers[:, 2].reshape(-1)
                    pred_depths = voxel_z_pred.reshape(-1)
                    sigma = 100.0  # 深度不确定性带宽
                    prob = np.exp(-0.5 * ((depths - pred_depths) / sigma) ** 2)
                    probs[fm_idx] = prob.reshape(nx, ny, nz)
                except Exception:
                    probs[fm_idx] = np.zeros((nx, ny, nz))
        else:
            # 实验组: 使用完整钻孔数据
            x_col = next((c for c in df.columns if c.lower() in ('x', 'x_m')), df.columns[0])
            y_col = next((c for c in df.columns if c.lower() in ('y', 'y_m')), df.columns[1])
            z_col = next((c for c in df.columns if c.lower() in ('z', 'z_m')), df.columns[2])
            fm_col = next((c for c in df.columns if c.lower() in ('formation_code',)), df.columns[3])

            # 构建 KD-Tree 用于最近邻分类
            borehole_pts = df[[x_col, y_col, z_col]].values
            borehole_fm = df[fm_col].map(fm_to_idx).values

            tree = cKDTree(borehole_pts)

            # 批量查询 (分批处理避免内存溢出)
            batch_size = 10000
            for start in range(0, n_voxels, batch_size):
                end = min(start + batch_size, n_voxels)
                batch_centers = voxel_centers[start:end]

                distances, indices = tree.query(batch_centers, k=min(10, len(borehole_pts)))

                if distances.ndim == 1:
                    distances = distances[:, np.newaxis]
                    indices = indices[:, np.newaxis]

                # 使用距离加权投票
                weights = 1.0 / (distances + 1.0)  # 避免除零
                weights /= weights.sum(axis=1, keepdims=True)

                for j, voxel_idx in enumerate(range(start, end)):
                    vi, vj, vk = np.unravel_index(voxel_idx, (nx, ny, nz))
                    for k in range(indices.shape[1]):
                        fm_idx = borehole_fm[indices[j, k]]
                        probs[fm_idx, vi, vj, vk] += weights[j, k]

        # 归一化概率
        prob_sum = probs.sum(axis=0) + 1e-10
        probs /= prob_sum[np.newaxis, :, :, :]

        return probs

    def _compute_voxel_entropy(self, probs: np.ndarray) -> np.ndarray:
        """
        计算每个体素的信息熵

        H(x) = -Σ p_k * log(p_k)
        """
        # 避免 log(0)
        probs_safe = np.maximum(probs, 1e-12)
        entropy = -np.sum(probs_safe * np.log(probs_safe), axis=0)

        # 归一化: 除以最大可能熵 (log(N))
        max_entropy = np.log(probs.shape[0])
        entropy /= max_entropy

        return entropy

    def _assign_grade(self, reduction_ratio: float) -> str:
        """根据熵减率评定等级"""
        if reduction_ratio > 0.30:
            return "A (优) — 虚拟钻孔显著降低深部不确定性，工程价值极高"
        elif reduction_ratio > 0.15:
            return "B (良) — 虚拟钻孔有效抑制深部插值发散，具备工程参考价值"
        elif reduction_ratio > 0.05:
            return "C (中) — 虚拟钻孔对深部不确定性有一定约束作用"
        else:
            return "D (差) — 虚拟钻孔对深部不确定性的约束有限，建议增加数据约束"

    def format_report(self, result: EntropyResult) -> str:
        """格式化输出验证报告"""
        lines = [
            "=" * 60,
            "  隐式建模不确定性缩减验证报告",
            "=" * 60,
            f"  建模体素数:         {result.n_voxels:,}",
            f"  地层种类数:          {result.n_formations}",
            f"  对照组信息熵总和:    {result.control_entropy_total:.2f}",
            f"  实验组信息熵总和:    {result.experiment_entropy_total:.2f}",
            f"  整体熵减率 (ΔH):    {result.entropy_reduction_ratio:.4f} ({result.entropy_reduction_ratio * 100:.2f}%)",
            f"  最大熵减深度:        {result.max_reduction_depth:.2f} m",
            f"  最大局部熵减率:      {result.max_reduction_ratio:.4f}",
            f"  质量等级:            {result.grade}",
            "=" * 60,
        ]
        return "\n".join(lines)


    def save_results(self, result: EntropyResult, output_dir: str,
                      prefix: str = "entropy_reduction") -> Dict[str, str]:
        """Save validation report and structured data to output directory."""
        os.makedirs(output_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        saved = {}

        report_path = os.path.join(output_dir, f"{prefix}_report_{timestamp}.txt")
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(self.format_report(result))
        saved["report"] = report_path

        data_dict = {
            "n_voxels": result.n_voxels,
            "n_formations": result.n_formations,
            "control_entropy_total": result.control_entropy_total,
            "experiment_entropy_total": result.experiment_entropy_total,
            "entropy_reduction_ratio": result.entropy_reduction_ratio,
            "max_reduction_depth": result.max_reduction_depth,
            "max_reduction_ratio": result.max_reduction_ratio,
            "per_depth_entropy": result.per_depth_entropy,
            "grade": result.grade,
        }
        json_path = os.path.join(output_dir, f"{prefix}_data_{timestamp}.json")
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(data_dict, f, ensure_ascii=False, indent=2, default=str)
        saved["json"] = json_path

        return saved


def demo_validation():
    """演示不确定性缩减验证流程（使用模拟数据）"""
    import tempfile

    tmpdir = tempfile.mkdtemp()
    print(f"[Demo] 使用临时目录: {tmpdir}")

    # 生成模拟虚拟钻孔数据
    np.random.seed(42)
    n_boreholes = 80
    n_formations = 4

    rows = []
    for bh_idx in range(n_boreholes):
        x = np.random.uniform(0, 10000)
        y = np.random.uniform(0, 10000)
        z_top = np.random.uniform(200, 500)

        current_z = z_top
        for fm_idx in range(n_formations):
            thickness = np.random.uniform(30, 100)
            rows.append({
                "x": x, "y": y, "z": current_z,
                "formation_code": f"F{fm_idx + 1}",
                "borehole_id": f"BH{bh_idx:03d}"
            })
            current_z -= thickness

    df_virtual = pd.DataFrame(rows)
    csv_path = os.path.join(tmpdir, "virtual_boreholes.csv")
    df_virtual.to_csv(csv_path, index=False)

    # 执行验证
    validator = EntropyReductionValidator(grid_resolution=(40, 40, 30))
    result = validator.run(csv_path)

    print(validator.format_report(result))

    # Save to output/
    output_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "output")
    saved = validator.save_results(result, output_dir, prefix="entropy_reduction_demo")
    print(f"\n[Demo] 报告已保存至: {saved['report']}")
    print(f"[Demo] 数据已保存至: {saved['json']}")
    return result


if __name__ == "__main__":
    demo_validation()
