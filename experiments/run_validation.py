"""
虚拟钻孔数据质量评价与验证体系 — 主入口

构建了"定性（逻辑自洽性）— 定量（拓扑逼近度）"相结合的多维交叉验证体系。

使用方法:
    # 1. 运行所有验证 (自动保存到 output/)
    python run_validation.py --all \
        --virtual_csv path/to/virtual_boreholes_points.csv \
        --formation_csv path/to/auto_generated_formation.csv

    # 2. 使用模拟数据演示 (自动保存到 output/)
    python run_validation.py --demo

    # 3. 仅运行定量验证
    python run_validation.py --quantitative \
        --virtual_csv path/to/virtual_boreholes_points.csv \
        --real_csv path/to/real_boreholes.csv

    # 4. 仅运行定性验证
    python run_validation.py --qualitative \
        --virtual_csv path/to/virtual_boreholes_points.csv \
        --dem_path path/to/dem.tif --shp_path path/to/boundary.shp

所有验证报告 (.txt) 和结构化数据 (.json, .csv) 均自动保存至:
    experiments/output/
"""

import os
import sys
import json
import argparse
import numpy as np
import pandas as pd
from datetime import datetime
from typing import Dict, Any, Optional

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)


def get_output_dir():
    p = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
    os.makedirs(p, exist_ok=True)
    return p


def run_quantitative_validation(virtual_csv, formation_csv=None, real_csv=None,
                                 dem_path=None, shp_path=None, section_line=None,
                                 output_dir=None):
    """Run all quantitative validations, save reports and data to output_dir."""
    if output_dir is None:
        output_dir = get_output_dir()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results = {}

    print("\n" + "=" * 70)
    print("  定量验证模块 (Quantitative Validation)")
    print("=" * 70)

    # 1.1 Blind Test
    print("\n>>> 1.1 真实钻孔盲测对比法...")
    from experiments.quantitative.blind_test import BlindTestValidator

    if real_csv and os.path.exists(real_csv):
        validator = BlindTestValidator(max_match_distance=500.0)
        result = validator.run(virtual_csv, real_csv, formation_csv)
        print(validator.format_report(result))
        saved = validator.save_results(result, output_dir,
                                        prefix=f"blind_test_{timestamp}")
        print(f"  [保存] 报告 -> {saved['report']}")
        results["blind_test"] = {
            "rmse": result.rmse, "mae": result.mae,
            "r_squared": result.r_squared, "grade": result.grade,
            "passed": result.passed, "files": saved,
        }
    else:
        print("  [跳过] 未提供真实钻孔 CSV (--real_csv)")

    # 1.2 Cross-Section IoU
    print("\n>>> 1.2 专家地质剖面二维拓扑交并比...")
    from experiments.quantitative.cross_section_iou import CrossSectionValidator

    cross_section_info = {}
    if section_line:
        validator = CrossSectionValidator(
            section_line=section_line, section_name="A-A'", swath_width=100.0,
        )
        df_virtual = pd.read_csv(virtual_csv)
        section_df = validator.extract_section_from_boreholes(df_virtual)
        grids = validator.reconstruct_formation_boundaries(section_df)
        print(f"  从剖面带宽内提取了 {len(section_df)} 个钻孔点，重构了 {len(grids)} 个地层网格")
        print("  (需要 expert_polygons 参数来完成完整 IoU 计算)")

        section_csv_path = os.path.join(output_dir, f"cross_section_slice_{timestamp}.csv")
        section_df.to_csv(section_csv_path, index=False, encoding="utf-8-sig")
        print(f"  [保存] 剖面切片 -> {section_csv_path}")

        cross_section_info = {
            "n_section_points": len(section_df),
            "n_formation_grids": len(grids),
            "section_csv": section_csv_path,
        }
        results["cross_section"] = cross_section_info
    else:
        print("  [跳过] 未提供剖面线 (--section_line)")

    # 1.3 Entropy Reduction
    print("\n>>> 1.3 隐式建模不确定性缩减...")
    from experiments.quantitative.entropy_reduction import EntropyReductionValidator

    validator = EntropyReductionValidator(grid_resolution=(50, 50, 30))
    result = validator.run(virtual_csv, shp_path, dem_path, formation_csv)
    print(validator.format_report(result))
    saved = validator.save_results(result, output_dir,
                                    prefix=f"entropy_reduction_{timestamp}")
    print(f"  [保存] 报告 -> {saved['report']}")
    results["entropy_reduction"] = {
        "entropy_reduction_ratio": result.entropy_reduction_ratio,
        "grade": result.grade, "files": saved,
    }

    return results


def run_qualitative_validation(virtual_csv, formation_csv=None, dem_path=None,
                                shp_path=None, output_dir=None):
    """Run all qualitative validations, save reports and data to output_dir."""
    if output_dir is None:
        output_dir = get_output_dir()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results = {}

    print("\n" + "=" * 70)
    print("  定性验证模块 (Qualitative Validation)")
    print("=" * 70)

    # 2.1 Superposition Check
    print("\n>>> 2.1 地质叠加定律规则硬审查...")
    from experiments.qualitative.superposition_check import SuperpositionValidator

    validator = SuperpositionValidator(max_dip_angle=60.0, neighbor_distance=100.0)
    result = validator.run(virtual_csv, formation_csv)
    print(validator.format_report(result))
    saved = validator.save_results(result, output_dir,
                                    prefix=f"superposition_{timestamp}")
    print(f"  [保存] 报告 -> {saved['report']}")
    if "superposition_csv" in saved:
        print(f"  [保存] 层序倒置CSV -> {saved['superposition_csv']}")
    if "thickness_csv" in saved:
        print(f"  [保存] 厚度突变CSV -> {saved['thickness_csv']}")
    results["superposition"] = {
        "violation_rate": result.overall_violation_rate,
        "superposition_violations": result.superposition_violations,
        "thickness_anomalies": result.thickness_anomalies,
        "grade": result.grade, "passed": result.passed, "files": saved,
    }

    # 2.2 Visual Consistency
    print("\n>>> 2.2 多模态空间自洽性评估...")
    from experiments.qualitative.visual_consistency import VisualConsistencyValidator

    validator = VisualConsistencyValidator()
    result = validator.run(virtual_csv, dem_path, shp_path)
    print(validator.format_report(result))
    saved = validator.save_results(result, output_dir,
                                    prefix=f"visual_consistency_{timestamp}")
    print(f"  [保存] 报告 -> {saved['report']}")
    if "anomalies_csv" in saved:
        print(f"  [保存] 异常点CSV -> {saved['anomalies_csv']}")
    results["visual_consistency"] = {
        "overall_score": result.overall_score,
        "orifice_pass_rate": result.orifice_pass_rate,
        "top_formation_match_rate": result.top_formation_match_rate,
        "smoothness_score": result.smoothness_score,
        "grade": result.grade, "passed": result.passed, "files": saved,
    }

    return results


def run_all_validations(virtual_csv, **kwargs):
    """Run all validations and save comprehensive report."""
    output_dir = kwargs.pop("output_dir", None) or get_output_dir()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    print("\n" + "#" * 70)
    print("#  虚拟钻孔数据质量评价与验证体系")
    print(f"#  执行时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"#  数据文件: {virtual_csv}")
    print(f"#  输出目录: {output_dir}")
    print("#" * 70)

    all_results = {
        "metadata": {
            "timestamp": datetime.now().isoformat(),
            "virtual_csv": virtual_csv,
            "output_dir": output_dir,
        },
    }

    all_results["quantitative"] = run_quantitative_validation(
        virtual_csv=virtual_csv,
        formation_csv=kwargs.get("formation_csv"),
        real_csv=kwargs.get("real_csv"),
        dem_path=kwargs.get("dem_path"),
        shp_path=kwargs.get("shp_path"),
        section_line=kwargs.get("section_line"),
        output_dir=output_dir,
    )

    all_results["qualitative"] = run_qualitative_validation(
        virtual_csv=virtual_csv,
        formation_csv=kwargs.get("formation_csv"),
        dem_path=kwargs.get("dem_path"),
        shp_path=kwargs.get("shp_path"),
        output_dir=output_dir,
    )

    # ─── Summary ──────────────────────────────────────────
    print("\n" + "=" * 70)
    print("  验证汇总")
    print("=" * 70)

    summary_lines = []
    for category, modules in [("定量", all_results["quantitative"]),
                                ("定性", all_results["qualitative"])]:
        for module_name, module_result in modules.items():
            if module_result:
                grade = module_result.get("grade", "N/A")
                passed = module_result.get("passed", None)
                summary_lines.append(
                    f"  [{category}] {module_name:25s}  {grade[:60]}"
                )

    for line in summary_lines:
        print(line)

    print("=" * 70)

    # Save JSON summary
    json_path = os.path.join(output_dir, f"validation_summary_{timestamp}.json")
    serializable = _make_serializable(all_results)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(serializable, f, ensure_ascii=False, indent=2)
    print(f"\n验证汇总已保存至: {json_path}")

    # Save text summary
    summary_path = os.path.join(output_dir, f"validation_summary_{timestamp}.txt")
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(f"虚拟钻孔数据质量评价与验证体系 — 汇总报告\n")
        f.write(f"执行时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"数据文件: {virtual_csv}\n")
        f.write("=" * 70 + "\n")
        for line in summary_lines:
            f.write(line + "\n")
        f.write("=" * 70 + "\n")
    print(f"汇总报告已保存至: {summary_path}")

    # ─── Generate visualization charts ─────────────────────
    print("\n" + "=" * 70)
    print("  生成可视化图表...")
    print("=" * 70)
    try:
        from experiments.visualization import VisualizationEngine
        viz = VisualizationEngine(output_dir=output_dir)
        viz_paths = viz.plot_all({
            "blind_test": all_results["quantitative"].get("blind_test"),
            "cross_section": all_results["quantitative"].get("cross_section"),
            "entropy_reduction": all_results["quantitative"].get("entropy_reduction"),
            "superposition": all_results["qualitative"].get("superposition"),
            "visual_consistency": all_results["qualitative"].get("visual_consistency"),
        })
    except Exception as e:
        print(f"  可视化生成失败: {e}")

    return all_results


def _make_serializable(obj):
    if isinstance(obj, dict):
        return {k: _make_serializable(v) for k, v in obj.items()
                if not k.startswith("_")}
    elif isinstance(obj, (list, tuple)):
        return [_make_serializable(item) for item in obj]
    elif isinstance(obj, (np.integer,)):
        return int(obj)
    elif isinstance(obj, (np.floating,)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (datetime,)):
        return obj.isoformat()
    elif hasattr(obj, '__dict__'):
        return str(obj)
    else:
        return obj


def demo_all():
    """Run all validation demos, then generate visualization charts."""
    output_dir = get_output_dir()
    print("\n" + "#" * 70)
    print("#  虚拟钻孔数据质量评价与验证体系 — 演示模式")
    print("#  所有数据均为模拟生成，用于展示验证流程")
    print(f"#  输出目录: {output_dir}")
    print("#" * 70)

    # Run all demos once, store results
    from experiments.quantitative.blind_test import demo_validation as demo1
    from experiments.quantitative.cross_section_iou import demo_validation as demo2
    from experiments.quantitative.entropy_reduction import demo_validation as demo3
    from experiments.qualitative.superposition_check import demo_validation as demo4
    from experiments.qualitative.visual_consistency import demo_validation as demo5

    print("\n>>> [定量] 1.1 真实钻孔盲测对比法")
    bt_result = demo1()
    print("\n>>> [定量] 1.2 专家地质剖面二维拓扑交并比")
    cs_result = demo2()
    print("\n>>> [定量] 1.3 隐式建模不确定性缩减")
    er_result = demo3()
    print("\n>>> [定性] 2.1 地质叠加定律规则硬审查")
    sp_result = demo4()
    print("\n>>> [定性] 2.2 多模态空间自洽性评估")
    vc_result = demo5()

    # ─── Generate visualization charts ─────────────────────
    print("\n" + "=" * 70)
    print("  生成可视化图表...")
    print("=" * 70)

    from experiments.visualization import VisualizationEngine
    viz = VisualizationEngine(output_dir=output_dir)
    viz.plot_all({
        "blind_test": bt_result,
        "cross_section": cs_result,
        "entropy_reduction": er_result,
        "superposition": sp_result,
        "visual_consistency": vc_result,
    })

    # List all generated files
    print("\n" + "=" * 70)
    print("  演示验证全部完成。输出文件列表:")
    print("=" * 70)
    all_files = sorted(os.listdir(output_dir))
    for fn in all_files:
        fpath = os.path.join(output_dir, fn)
        size_kb = os.path.getsize(fpath) / 1024
        print(f"  {fn:50s}  ({size_kb:.1f} KB)")
    print("=" * 70)
    print(f"  所有文件已保存至: {output_dir}")
    print("=" * 70)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="虚拟钻孔数据质量评价与验证体系",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--all", action="store_true", help="运行全部验证")
    parser.add_argument("--quantitative", action="store_true", help="仅运行定量验证")
    parser.add_argument("--qualitative", action="store_true", help="仅运行定性验证")
    parser.add_argument("--demo", action="store_true", help="使用模拟数据演示所有验证流程")
    parser.add_argument("--virtual_csv", type=str, help="虚拟钻孔点云 CSV 路径")
    parser.add_argument("--formation_csv", type=str, help="地层特征配置 CSV 路径")
    parser.add_argument("--real_csv", type=str, help="真实钻孔 CSV 路径 (盲测对比用)")
    parser.add_argument("--dem_path", type=str, help="DEM 栅格文件路径 (.tif)")
    parser.add_argument("--shp_path", type=str, help="SHP 面文件路径 (.shp)")
    parser.add_argument("--section_line", type=str,
                        help="剖面线坐标 JSON, 如 '[[x1,y1],[x2,y2]]'")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="输出目录 (默认 experiments/output/)")
    args = parser.parse_args()

    if args.demo:
        demo_all()
        sys.exit(0)

    if not args.virtual_csv:
        print("错误: 必须提供 --virtual_csv 参数，或使用 --demo 运行演示模式。")
        parser.print_help()
        sys.exit(1)

    if not os.path.exists(args.virtual_csv):
        print(f"错误: 虚拟钻孔文件不存在: {args.virtual_csv}")
        sys.exit(1)

    section_line = None
    if args.section_line:
        try:
            section_line = json.loads(args.section_line)
        except Exception:
            print(f"警告: 无法解析剖面线 JSON: {args.section_line}")

    output_dir = args.output_dir or get_output_dir()
    kwargs = {
        "formation_csv": args.formation_csv,
        "real_csv": args.real_csv,
        "dem_path": args.dem_path,
        "shp_path": args.shp_path,
        "section_line": section_line,
        "output_dir": output_dir,
    }

    if args.quantitative:
        run_quantitative_validation(args.virtual_csv, **kwargs)
    elif args.qualitative:
        run_qualitative_validation(args.virtual_csv, **kwargs)
    else:
        run_all_validations(args.virtual_csv, **kwargs)
