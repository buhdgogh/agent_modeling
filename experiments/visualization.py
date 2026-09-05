"""
虚拟钻孔验证结果可视化模块

为每个验证模块生成发布级图表 (.png)，供论文章节使用。

可视化清单:
    Blind Test:           散点图 (virtual Z vs real Z)、地层 RMSE 柱状图、误差分布直方图
    Cross-Section IoU:    地层 IoU 柱状图、剖面二维地层对比图
    Entropy Reduction:    深度-熵值曲线、熵减率-深度曲线、体素熵值 3D 散点
    Superposition Check:  违规类型饼图、违规数量柱状图、违规空间分布
    Visual Consistency:   多维评分雷达图、孔口偏差直方图、综合评分柱状图

用法:
    from experiments.visualization import VisualizationEngine
    viz = VisualizationEngine(output_dir="experiments/output")
    viz.plot_blind_test(result)
    viz.plot_all(results_dict)
"""

import os
import numpy as np
import pandas as pd
from datetime import datetime
from typing import Dict, List, Optional, Any
import warnings

warnings.filterwarnings("ignore", category=UserWarning)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.patches import FancyBboxPatch
import matplotlib.font_manager as fm

# ─── attempt to use SimHei / Microsoft YaHei for CJK labels ───
_CJK_FONT = None
for _candidate in ["SimHei", "Microsoft YaHei", "WenQuanYi Micro Hei",
                    "Noto Sans CJK SC", "Arial Unicode MS", "sans-serif"]:
    try:
        fm.findfont(_candidate, fallback_to_default=False)
        _CJK_FONT = _candidate
        break
    except Exception:
        continue
if _CJK_FONT:
    plt.rcParams["font.family"] = _CJK_FONT
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["figure.dpi"] = 150
plt.rcParams["savefig.dpi"] = 300
plt.rcParams["savefig.bbox"] = "tight"
plt.rcParams["savefig.pad_inches"] = 0.1


# ═══════════════════════════════════════════════════════════════
#  Visualization Engine
# ═══════════════════════════════════════════════════════════════

class VisualizationEngine:
    """统一图表生成引擎"""

    def __init__(self, output_dir: str = None):
        if output_dir is None:
            from experiments.run_validation import get_output_dir
            output_dir = get_output_dir()
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        self._ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    def _save(self, name: str) -> str:
        path = os.path.join(self.output_dir, f"{name}_{self._ts}.png")
        plt.savefig(path)
        plt.close()
        return path

    # ─── 1.1 Blind Test ─────────────────────────────────────

    def plot_blind_test_scatter(self, result) -> str:
        """Virtual Z vs Real Z scatter with 1:1 reference line.

        Parameters
        ----------
        result : BlindTestResult or dict with keys:
            z_virtual, z_real (lists), rmse, mae, r_squared, n_pairs
        """
        fig, ax = plt.subplots(figsize=(7, 7))

        z_v = np.array(getattr(result, "z_virtual_all", []))
        z_r = np.array(getattr(result, "z_real_all", []))

        if len(z_v) == 0:
            # fallback: reconstruct from per-pair errors stored on result
            z_v = getattr(result, "_z_v_all", None)
            z_r = getattr(result, "_z_r_all", None)
            if z_v is None:
                ax.text(0.5, 0.5, "No scatter data — run validator with real CSV",
                        ha="center", transform=ax.transAxes, fontsize=14)
                return self._save("blind_test_scatter_empty")

        rmse = getattr(result, "rmse", 0)
        r2 = getattr(result, "r_squared", 0)

        ax.scatter(z_r, z_v, c="#2c7bb6", edgecolors="white", s=60, alpha=0.75, zorder=5)

        # 1:1 line
        all_vals = np.concatenate([z_v, z_r])
        lo, hi = all_vals.min() - 20, all_vals.max() + 20
        ax.plot([lo, hi], [lo, hi], "--", color="#d7191c", linewidth=2,
                label="1:1 (perfect match)", zorder=3)
        ax.fill_between([lo, hi], [lo - rmse, hi - rmse], [lo + rmse, hi + rmse],
                        alpha=0.10, color="#d7191c", label=f"+/- RMSE ({rmse:.1f} m)")

        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        ax.set_xlabel("Real Z (m)", fontsize=13)
        ax.set_ylabel("Virtual Z (m)", fontsize=13)
        ax.set_title(f"Blind Test: Virtual vs Real Depth\nRMSE={rmse:.1f}m, "
                     f"R^2={r2:.3f}, n={len(z_v)} interfaces",
                     fontsize=14, fontweight="bold")
        ax.legend(loc="upper left", framealpha=0.9)
        ax.set_aspect("equal")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        return self._save("blind_test_scatter")

    def plot_blind_test_per_formation(self, result) -> str:
        """Grouped bar chart: RMSE / MAE per formation."""
        per_fm_rmse = getattr(result, "per_formation_rmse", {})
        per_fm_mae = getattr(result, "per_formation_mae", {})

        if not per_fm_rmse:
            fig, ax = plt.subplots(figsize=(8, 5))
            ax.text(0.5, 0.5, "No per-formation data", ha="center",
                    transform=ax.transAxes, fontsize=14)
            return self._save("blind_test_per_formation_empty")

        formations = sorted(per_fm_rmse.keys())
        x = np.arange(len(formations))
        w = 0.35

        fig, ax = plt.subplots(figsize=(max(8, len(formations) * 1.2), 5.5))
        rmse_vals = [per_fm_rmse[f] for f in formations]
        mae_vals = [per_fm_mae.get(f, 0) for f in formations]

        bars1 = ax.bar(x - w / 2, rmse_vals, w, color="#d7191c", alpha=0.85,
                       edgecolor="white", label="RMSE (m)")
        bars2 = ax.bar(x + w / 2, mae_vals, w, color="#2c7bb6", alpha=0.85,
                       edgecolor="white", label="MAE (m)")

        for bar in bars1:
            ax.text(bar.get_x() + bar.get_width() / 2., bar.get_height() + 1,
                    f"{bar.get_height():.1f}", ha="center", fontsize=8)
        for bar in bars2:
            ax.text(bar.get_x() + bar.get_width() / 2., bar.get_height() + 1,
                    f"{bar.get_height():.1f}", ha="center", fontsize=8)

        ax.set_xticks(x)
        ax.set_xticklabels(formations, fontsize=11)
        ax.set_ylabel("Error (m)", fontsize=13)
        ax.set_title("Per-Formation Depth Error", fontsize=14, fontweight="bold")
        ax.legend(framealpha=0.9, fontsize=11)
        ax.grid(axis="y", alpha=0.3)
        fig.tight_layout()
        return self._save("blind_test_per_formation")

    def plot_blind_test_histogram(self, result) -> str:
        """Error distribution histogram with KDE overlay."""
        errors = getattr(result, "_all_errors", None)
        if errors is None or len(errors) == 0:
            fig, ax = plt.subplots()
            ax.text(0.5, 0.5, "No error data", ha="center",
                    transform=ax.transAxes, fontsize=14)
            return self._save("blind_test_histogram_empty")

        errors = np.array(errors)
        fig, ax = plt.subplots(figsize=(8, 4.5))

        ax.hist(errors, bins=25, density=True, color="#2c7bb6", alpha=0.7,
                edgecolor="white", label="Error distribution")
        ax.axvline(np.mean(errors), color="#d7191c", linewidth=2, linestyle="--",
                   label=f"Mean = {np.mean(errors):.1f} m")
        ax.axvline(np.median(errors), color="#fdae61", linewidth=2, linestyle="-.",
                   label=f"Median = {np.median(errors):.1f} m")

        ax.set_xlabel("Absolute Error (m)", fontsize=13)
        ax.set_ylabel("Density", fontsize=13)
        ax.set_title(f"Depth Error Distribution (n={len(errors)})",
                     fontsize=14, fontweight="bold")
        ax.legend(fontsize=11)
        ax.grid(axis="y", alpha=0.3)
        fig.tight_layout()
        return self._save("blind_test_histogram")

    # ─── 1.2 Cross-Section IoU ──────────────────────────────

    def plot_cross_section_iou(self, result) -> str:
        """Bar chart: IoU per formation."""
        per_fm = getattr(result, "per_formation_iou", {})
        per_fd = getattr(result, "frechet_distances", {})
        mean_iou = getattr(result, "mean_iou", 0)

        if not per_fm:
            fig, ax = plt.subplots()
            ax.text(0.5, 0.5, "No IoU data", ha="center",
                    transform=ax.transAxes, fontsize=14)
            return self._save("cross_section_iou_empty")

        formations = sorted(per_fm.keys())
        x = np.arange(len(formations))
        iou_vals = [per_fm[f] for f in formations]

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(max(10, len(formations) * 1.3), 5))

        colors = ["#1a9641" if v >= 0.7 else "#a6d96a" if v >= 0.5
                  else "#fdae61" if v >= 0.3 else "#d7191c" for v in iou_vals]

        ax1.bar(x, iou_vals, color=colors, edgecolor="white", width=0.6)
        ax1.axhline(0.7, color="#1a9641", linestyle="--", linewidth=1.5, alpha=0.7)
        ax1.axhline(0.5, color="#fdae61", linestyle="--", linewidth=1.5, alpha=0.7)
        ax1.axhline(mean_iou, color="#2c7bb6", linestyle="-", linewidth=2,
                     label=f"Mean IoU = {mean_iou:.3f}")
        for i, (fm, v) in enumerate(zip(formations, iou_vals)):
            ax1.text(i, v + 0.02, f"{v:.3f}", ha="center", fontsize=9)
        ax1.set_xticks(x)
        ax1.set_xticklabels(formations, fontsize=11)
        ax1.set_ylabel("IoU", fontsize=13)
        ax1.set_title(f"Cross-Section IoU per Formation\n(section: {getattr(result, 'section_name', '')})",
                      fontsize=12, fontweight="bold")
        ax1.set_ylim(0, 1.1)
        ax1.legend(fontsize=9)
        ax1.grid(axis="y", alpha=0.3)

        # Fréchet distances
        fd_vals = [per_fd.get(f, 0) for f in formations]
        ax2.bar(x, fd_vals, color="#5e3c99", edgecolor="white", width=0.6)
        for i, (fm, v) in enumerate(zip(formations, fd_vals)):
            ax2.text(i, v + max(fd_vals) * 0.02, f"{v:.0f}", ha="center", fontsize=9)
        ax2.set_xticks(x)
        ax2.set_xticklabels(formations, fontsize=11)
        ax2.set_ylabel("Fréchet Distance (m)", fontsize=13)
        ax2.set_title("Fréchet Distance per Formation", fontsize=12, fontweight="bold")
        ax2.grid(axis="y", alpha=0.3)

        fig.tight_layout()
        return self._save("cross_section_iou")

    # ─── 1.3 Entropy Reduction ──────────────────────────────

    def plot_entropy_vs_depth(self, result) -> str:
        """Dual line plot: entropy vs depth for control and experiment groups."""
        per_depth = getattr(result, "per_depth_entropy", {})
        control = per_depth.get("control", [])
        experiment = per_depth.get("experiment", [])
        reduction_ratio = per_depth.get("reduction_ratio", [])
        max_red_depth = getattr(result, "max_reduction_depth", 0)
        overall_ratio = getattr(result, "entropy_reduction_ratio", 0)

        if not control or not experiment:
            fig, ax = plt.subplots()
            ax.text(0.5, 0.5, "No entropy data", ha="center",
                    transform=ax.transAxes, fontsize=14)
            return self._save("entropy_vs_depth_empty")

        depths = np.linspace(0, -500, len(control))  # approximate

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

        ax1.plot(control, depths, color="#d7191c", linewidth=2.5, label="Control (surface only)")
        ax1.plot(experiment, depths, color="#2c7bb6", linewidth=2.5, label="Experiment (+ boreholes)")
        ax1.set_xlabel("Normalized Entropy H(x)", fontsize=12)
        ax1.set_ylabel("Approximate Depth Index", fontsize=12)
        ax1.set_title("Entropy vs Depth:\nControl vs Experiment",
                      fontsize=13, fontweight="bold")
        ax1.legend(fontsize=10, framealpha=0.9)
        ax1.grid(True, alpha=0.3)
        ax1.invert_yaxis()

        ax2.plot(reduction_ratio, depths, color="#5e3c99", linewidth=2.5)
        ax2.axvline(0, color="gray", linestyle="--", linewidth=1, alpha=0.5)
        ax2.axhline(max_red_depth, color="#d7191c", linestyle="--", linewidth=1.5,
                    alpha=0.7, label=f"Max reduction at depth idx={max_red_depth:.0f}")
        ax2.set_xlabel("Entropy Reduction Ratio", fontsize=12)
        ax2.set_title(f"Entropy Reduction Ratio vs Depth\n"
                      f"(overall ratio = {overall_ratio:.3f})",
                      fontsize=13, fontweight="bold")
        ax2.legend(fontsize=10)
        ax2.grid(True, alpha=0.3)
        ax2.invert_yaxis()

        fig.tight_layout()
        return self._save("entropy_vs_depth")

    # ─── 2.1 Superposition Check ────────────────────────────

    def plot_superposition_violations(self, result) -> str:
        """Pie chart + bar chart: violation distribution."""
        sp_v = getattr(result, "superposition_violations", 0)
        th_a = getattr(result, "thickness_anomalies", 0)
        n_total = getattr(result, "n_boreholes_checked", 1)
        n_pass = n_total - sp_v - th_a if n_total > (sp_v + th_a) else 0

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.8))

        # Pie chart
        labels = ["Passed", "Sequence Violations", "Thickness Anomalies"]
        sizes = [max(n_pass, 0), sp_v, th_a]
        colors_pie = ["#1a9641", "#d7191c", "#fdae61"]
        explode = (0, 0.05, 0.05)

        wedges, texts, autotexts = ax1.pie(
            sizes, labels=labels, colors=colors_pie, explode=explode,
            autopct="%1.1f%%", startangle=90, pctdistance=0.6,
            wedgeprops=dict(edgecolor="white", linewidth=1.5))
        for at in autotexts:
            at.set_fontsize(10)
            at.set_fontweight("bold")
        ax1.set_title(f"Borehole Validation Results\n(n={n_total})",
                      fontsize=13, fontweight="bold")

        # Bar chart
        categories = ["Seq. Violations", "Thickness Anomalies"]
        counts = [sp_v, th_a]
        colors_bar = ["#d7191c", "#fdae61"]
        bars = ax2.bar(categories, counts, color=colors_bar, edgecolor="white", width=0.5)
        for bar, cnt in zip(bars, counts):
            ax2.text(bar.get_x() + bar.get_width() / 2., bar.get_height() + max(1, max(counts) * 0.02),
                     str(cnt), ha="center", fontsize=12, fontweight="bold")
        ax2.set_ylabel("Count", fontsize=12)
        ax2.set_title("Violation Breakdown", fontsize=13, fontweight="bold")
        ax2.grid(axis="y", alpha=0.3)

        fig.tight_layout()
        return self._save("superposition_violations")

    # ─── 2.2 Visual Consistency ─────────────────────────────

    def plot_visual_consistency_radar(self, result) -> str:
        """Radar chart of multi-dimensional spatial consistency scores."""
        overall = getattr(result, "overall_score", 0)
        orifice = getattr(result, "orifice_pass_rate", 0) * 100
        formation_match = getattr(result, "top_formation_match_rate", 0) * 100
        smoothness = getattr(result, "smoothness_score", 0) * 100

        categories = ["Orifice Fit", "Formation Match", "Stratal Smoothness", "Overall"]
        values = [orifice, formation_match, smoothness, overall]
        N = len(categories)

        angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
        values += values[:1]
        angles += angles[:1]

        fig, ax = plt.subplots(figsize=(6.5, 6.5), subplot_kw=dict(polar=True))
        ax.fill(angles, values, color="#2c7bb6", alpha=0.3)
        ax.plot(angles, values, color="#2c7bb6", linewidth=2.5, marker="o", markersize=8)

        for a, v in zip(angles[:-1], values[:-1]):
            ax.text(a, v + 3, f"{v:.1f}", ha="center", fontsize=10, fontweight="bold")

        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(categories, fontsize=11)
        ax.set_ylim(0, 105)
        ax.set_yticks([20, 40, 60, 80, 100])
        ax.set_yticklabels(["20", "40", "60", "80", "100"], fontsize=8, color="gray")
        ax.set_title(f"Spatial Consistency Multi-Dimensional Score\n"
                     f"(Overall = {overall:.1f}/100)",
                     fontsize=14, fontweight="bold", pad=25)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        return self._save("visual_consistency_radar")

    def plot_visual_consistency_bars(self, result) -> str:
        """Horizontal bar chart of individual dimension scores."""
        overall = getattr(result, "overall_score", 0)
        orifice = getattr(result, "orifice_pass_rate", 0) * 100
        formation_match = getattr(result, "top_formation_match_rate", 0) * 100
        smoothness = getattr(result, "smoothness_score", 0) * 100

        dimensions = [
            "Overall Score",
            "Orifice Fit (vs DEM)",
            "Formation-Map Match",
            "Stratal Smoothness",
        ]
        scores = [overall, orifice, formation_match, smoothness]
        colors = ["#2c7bb6", "#1a9641", "#fdae61", "#5e3c99"]

        fig, ax = plt.subplots(figsize=(9, 3.5))
        y_pos = np.arange(len(dimensions))
        bars = ax.barh(y_pos, scores, color=colors, edgecolor="white", height=0.6)

        for bar, score in zip(bars, scores):
            ax.text(bar.get_width() + 1.5, bar.get_y() + bar.get_height() / 2.,
                    f"{score:.1f}", va="center", fontsize=11, fontweight="bold")

        ax.set_yticks(y_pos)
        ax.set_yticklabels(dimensions, fontsize=12)
        ax.set_xlim(0, 110)
        ax.set_xlabel("Score (0-100)", fontsize=12)
        ax.set_title("Visual Spatial Consistency — Dimension Scores",
                     fontsize=14, fontweight="bold")

        # Add grade thresholds
        for thresh, label, color in [(90, "Grade A (90+)", "#1a9641"),
                                       (75, "Grade B (75+)", "#a6d96a"),
                                       (60, "Grade C (60+)", "#fdae61")]:
            ax.axvline(thresh, color=color, linestyle="--", linewidth=1, alpha=0.4)
            ax.text(thresh + 0.3, len(dimensions) - 0.3, label, fontsize=8,
                    color=color, alpha=0.8)

        ax.grid(axis="x", alpha=0.3)
        fig.tight_layout()
        return self._save("visual_consistency_bars")

    @staticmethod
    def _get(obj, key, default=None):
        """Safely get attribute or dict key from result objects."""
        if obj is None:
            return default
        if isinstance(obj, dict):
            return obj.get(key, default)
        return getattr(obj, key, default)

    # ─── Composite / Summary ────────────────────────────────

    def plot_summary_dashboard(self, validation_results: Dict[str, Any]) -> str:
        """Generate a single multi-panel summary figure."""
        fig = plt.figure(figsize=(14, 12))
        gs = fig.add_gridspec(3, 3, hspace=0.35, wspace=0.35)

        # ── Panel 1: Blind Test grade ──
        ax1 = fig.add_subplot(gs[0, 0])
        bt = self._get(validation_results, "blind_test")
        if bt:
            self._draw_grade_card(ax1, "Blind Test\n(Real vs Virtual)", bt)
        else:
            ax1.text(0.5, 0.5, "N/A", ha="center", va="center",
                     transform=ax1.transAxes, fontsize=14, color="gray")
            ax1.set_title("Blind Test", fontsize=12, fontweight="bold")

        # ── Panel 2: Entropy Reduction grade ──
        ax2 = fig.add_subplot(gs[0, 1])
        er = self._get(validation_results, "entropy_reduction")
        if er:
            self._draw_grade_card(ax2, "Entropy\nReduction", er)
        else:
            ax2.text(0.5, 0.5, "N/A", ha="center", va="center",
                     transform=ax2.transAxes, fontsize=14, color="gray")
            ax2.set_title("Entropy Reduction", fontsize=12, fontweight="bold")

        # ── Panel 3: Cross-Section grade ──
        ax3 = fig.add_subplot(gs[0, 2])
        cs = self._get(validation_results, "cross_section")
        if cs and self._get(cs, "n_section_points"):
            self._draw_grade_card(ax3, "Cross-Section\nIoU", cs)
        else:
            ax3.text(0.5, 0.5, "N/A", ha="center", va="center",
                     transform=ax3.transAxes, fontsize=14, color="gray")
            ax3.set_title("Cross-Section IoU", fontsize=12, fontweight="bold")

        # ── Panel 4: Superposition pie ──
        ax4 = fig.add_subplot(gs[1, :2])
        sp = self._get(validation_results, "superposition")
        if sp:
            n_total = self._get(sp, "n_boreholes_checked", 0)
            sp_v = self._get(sp, "superposition_violations", 0)
            th_a = self._get(sp, "thickness_anomalies", 0)
            n_pass = max(n_total - sp_v - th_a, 0)

            labels = ["Passed", "Sequence Viol.", "Thickness Anom."]
            sizes = [n_pass, sp_v, th_a]
            colors_pie = ["#1a9641", "#d7191c", "#fdae61"]
            ax4.pie(sizes, labels=labels, colors=colors_pie, autopct="%1.1f%%",
                    startangle=90, explode=(0, 0.05, 0.05),
                    wedgeprops=dict(edgecolor="white", linewidth=1.2))
            ax4.set_title(f"Superposition Check (n={n_total} boreholes)",
                          fontsize=12, fontweight="bold")
        else:
            ax4.text(0.5, 0.5, "N/A", ha="center", va="center",
                     transform=ax4.transAxes, fontsize=14, color="gray")
            ax4.set_title("Superposition Check", fontsize=12, fontweight="bold")

        # ── Panel 5: Visual Consistency barh ──
        ax5 = fig.add_subplot(gs[1, 2])
        vc = self._get(validation_results, "visual_consistency")
        if vc:
            dims = ["Orifice Fit", "Form. Match", "Smoothness"]
            scores_vc = [
                self._get(vc, "orifice_pass_rate", 0) * 100,
                self._get(vc, "top_formation_match_rate", 0) * 100,
                self._get(vc, "smoothness_score", 0) * 100,
            ]
            y = np.arange(len(dims))
            ax5.barh(y, scores_vc, color=["#1a9641", "#fdae61", "#5e3c99"],
                     edgecolor="white", height=0.5)
            ax5.set_yticks(y)
            ax5.set_yticklabels(dims, fontsize=10)
            ax5.set_xlim(0, 110)
            for i, sv in enumerate(scores_vc):
                ax5.text(sv + 1.5, i, f"{sv:.1f}", va="center", fontsize=10, fontweight="bold")
            ax5.set_title("Visual Consistency Scores", fontsize=12, fontweight="bold")
            ax5.grid(axis="x", alpha=0.3)
        else:
            ax5.text(0.5, 0.5, "N/A", ha="center", va="center",
                     transform=ax5.transAxes, fontsize=14, color="gray")
            ax5.set_title("Visual Consistency", fontsize=12, fontweight="bold")

        # ── Panel 6: Summary table ──
        ax6 = fig.add_subplot(gs[2, :])
        ax6.axis("off")
        table_data, col_labels = self._build_summary_table(validation_results)
        if table_data:
            tbl = ax6.table(cellText=table_data, colLabels=col_labels,
                            cellLoc="center", loc="center",
                            colWidths=[0.2, 0.22, 0.18, 0.4])
            tbl.auto_set_font_size(False)
            tbl.set_fontsize(10)
            tbl.scale(1, 1.6)
            # Color-code rows
            for i in range(len(table_data)):
                grade = table_data[i][2]
                color = "#c8e6c9" if "A" in grade else "#fff9c4" if "B" in grade \
                    else "#ffe0b2" if "C" in grade else "#ffcdd2"
                for j in range(len(col_labels)):
                    tbl[(i + 1, j)].set_facecolor(color)
            # Header
            for j in range(len(col_labels)):
                tbl[(0, j)].set_facecolor("#37474f")
                tbl[(0, j)].set_text_props(color="white", fontweight="bold")
        ax6.set_title("Validation Summary", fontsize=14, fontweight="bold", pad=60)

        fig.suptitle("Virtual Borehole Quality Validation Dashboard",
                     fontsize=17, fontweight="bold", y=1.01)
        return self._save("summary_dashboard")

    def _draw_grade_card(self, ax, title, data):
        """Draw a grade-card style panel."""
        grade = self._get(data, "grade", "N/A")
        passed = self._get(data, "passed", None)

        if "A" in str(grade):
            color = "#1a9641"
        elif "B" in str(grade):
            color = "#a6d96a"
        elif "C" in str(grade):
            color = "#fdae61"
        else:
            color = "#d7191c"

        ax.text(0.5, 0.7, str(grade)[:2] if len(str(grade)) > 2 else str(grade),
                ha="center", va="center", fontsize=36, fontweight="bold", color=color)
        ax.text(0.5, 0.3, str(grade) if len(str(grade)) > 3 else "",
                ha="center", va="center", fontsize=7, color="gray")

        status_text = "PASS" if passed else ("FAIL" if passed is False else "---")
        ax.text(0.85, 0.92, status_text, ha="center", va="center", fontsize=10,
                fontweight="bold",
                color="#1a9641" if passed else ("#d7191c" if passed is False else "gray"),
                transform=ax.transAxes,
                bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.8))

        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis("off")
        ax.set_title(title, fontsize=11, fontweight="bold", pad=8)

    def _build_summary_table(self, vr: Dict) -> tuple:
        """Build summary table rows from validation results dict or dataclasses."""
        rows = []
        col_labels = ["Module", "Key Metric", "Grade", "Detail"]

        bt = self._get(vr, "blind_test")
        if bt:
            rows.append(["Blind Test", f"RMSE={self._get(bt, 'rmse', 0):.1f}m",
                         str(self._get(bt, 'grade', 'N/A'))[:8],
                         str(self._get(bt, 'grade', ''))])

        cs = self._get(vr, "cross_section")
        if cs:
            rows.append(["Cross-Section IoU",
                         f"n_points={self._get(cs, 'n_section_points', 0)}",
                         str(self._get(cs, 'grade', 'N/A'))[:8] if self._get(cs, 'grade') else "N/A",
                         "need expert polygons for full IoU"])

        er = self._get(vr, "entropy_reduction")
        if er:
            rows.append(["Entropy Reduction",
                         f"ratio={self._get(er, 'entropy_reduction_ratio', 0):.3f}",
                         str(self._get(er, 'grade', 'N/A'))[:8], ""])

        sp = self._get(vr, "superposition")
        if sp:
            rows.append(["Superposition",
                         f"viol_rate={self._get(sp, 'violation_rate', 0):.3%}",
                         str(self._get(sp, 'grade', 'N/A'))[:8],
                         f"{self._get(sp, 'superposition_violations', 0)} seq + "
                         f"{self._get(sp, 'thickness_anomalies', 0)} thick"])

        vc = self._get(vr, "visual_consistency")
        if vc:
            rows.append(["Visual Consistency",
                         f"score={self._get(vc, 'overall_score', 0):.1f}",
                         str(self._get(vc, 'grade', 'N/A'))[:8],
                         f"orifice={self._get(vc, 'orifice_pass_rate', 0):.1%}"])

        return rows, col_labels

    # ─── Batch runner ───────────────────────────────────────

    def plot_all(self, validation_results: Dict[str, Any]) -> List[str]:
        """Generate all applicable charts from a validation results dict.

        Parameters
        ----------
        validation_results : dict with keys like:
            "blind_test": BlindTestResult | dict
            "cross_section": CrossSectionResult | dict
            "entropy_reduction": EntropyResult | dict
            "superposition": SuperpositionResult | dict
            "visual_consistency": VisualConsistencyResult | dict

        Returns
        -------
        List[str] : paths to generated PNG files
        """
        paths = []

        print("\n>>> 生成验证可视化图表...")

        # Blind Test
        bt = validation_results.get("blind_test")
        if bt:
            print("  1.1 Blind Test scatter...")
            paths.append(self.plot_blind_test_scatter(bt))
            print("  1.1 Blind Test per-formation...")
            paths.append(self.plot_blind_test_per_formation(bt))
            print("  1.1 Blind Test histogram...")
            paths.append(self.plot_blind_test_histogram(bt))

        # Cross-Section
        cs = validation_results.get("cross_section")
        if cs and getattr(cs, "per_formation_iou", None):
            print("  1.2 Cross-Section IoU...")
            paths.append(self.plot_cross_section_iou(cs))

        # Entropy Reduction
        er = validation_results.get("entropy_reduction")
        if er:
            print("  1.3 Entropy vs Depth...")
            paths.append(self.plot_entropy_vs_depth(er))

        # Superposition
        sp = validation_results.get("superposition")
        if sp:
            print("  2.1 Superposition violations...")
            paths.append(self.plot_superposition_violations(sp))

        # Visual Consistency
        vc = validation_results.get("visual_consistency")
        if vc:
            print("  2.2 Visual Consistency radar...")
            paths.append(self.plot_visual_consistency_radar(vc))
            print("  2.2 Visual Consistency bars...")
            paths.append(self.plot_visual_consistency_bars(vc))

        # Summary Dashboard
        print("  Summary dashboard...")
        paths.append(self.plot_summary_dashboard(validation_results))

        print(f"\n共生成 {len(paths)} 张图表:")
        for p in paths:
            print(f"  {p}")

        return paths


# ═══════════════════════════════════════════════════════════════
#  Demo
# ═══════════════════════════════════════════════════════════════

def demo_visualization():
    """Run full demo: generate all validation results then plot them."""
    print("=" * 70)
    print("  Visualization Demo — generating all validation charts")
    print("=" * 70)

    from experiments.run_validation import get_output_dir
    output_dir = get_output_dir()

    # Generate validation results using demo functions
    print("\n>>> Running validation demos to get result objects...")
    from experiments.quantitative.blind_test import demo_validation as bt_demo
    from experiments.quantitative.entropy_reduction import demo_validation as er_demo
    from experiments.qualitative.superposition_check import demo_validation as sp_demo
    from experiments.qualitative.visual_consistency import demo_validation as vc_demo

    bt_result = bt_demo()
    er_result = er_demo()
    sp_result = sp_demo()
    vc_result = vc_demo()

    # Also run cross-section demo
    from experiments.quantitative.cross_section_iou import demo_validation as cs_demo
    cs_result = cs_demo()

    validation_results = {
        "blind_test": bt_result,
        "cross_section": cs_result,
        "entropy_reduction": er_result,
        "superposition": sp_result,
        "visual_consistency": vc_result,
    }

    viz = VisualizationEngine(output_dir=output_dir)
    paths = viz.plot_all(validation_results)

    print("\n" + "=" * 70)
    print("  Visualization demo complete.")
    print(f"  Output directory: {output_dir}")
    print("=" * 70)


if __name__ == "__main__":
    demo_visualization()
