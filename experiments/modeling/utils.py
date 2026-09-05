import os, gempy as gp, gempy_viewer as gpv, pyvista as pv

def print_model_info(geo_model):
    print("=========================================")
    # 打印 structural_elements（所有元素，包括 basement）
    print("模型中所有的地质元素 (名称 + 是否激活):")
    for e in geo_model.structural_frame.structural_elements:
        print("-", getattr(e, "name", str(e)), " active:", getattr(e, "is_active", "N/A"))

    # 打印 groups 与 group 内元素
    print("\n模型中的地质组和对应地质组中的元素:")
    for group in geo_model.structural_frame.structural_groups:
        print("Group:", group.name)
        for e in group.elements:
            print("   └──", e.name)
    print("=========================================")


def make_empty_elements_list(names_list, colors_list):
    """
    接收一个名称列表 (names_list) 和一个颜色列表 (colors_list),
    返回一个 StructuralElement 对象的列表。
    假定 names_list 和 colors_list 长度相同。
    """
    # 使用列表推导式 (list comprehension) 和 zip 来同时遍历两个列表
    element_list = [
        gp.data.StructuralElement(
            name=name,
            color=color,
            surface_points=gp.data.SurfacePointsTable.initialize_empty(),
            orientations=gp.data.OrientationsTable.initialize_empty(),
        )
        for name, color in zip(names_list, colors_list)
    ]

    return element_list


def clean_plot_3d(geo_model, **kwargs):
    """
    一个 "干净" 的 3D 绘图函数：
    - 移除坐标轴、边界框、网格
    - **默认强制** 绘制岩性 (plot_lith=True) - 这会创建图例
    - **保留** 地层图例 (legend)
    """
    # opacity 控制模型透明度
    # if 'opacity' not in kwargs:
    #     kwargs['opacity'] = 1.0

    # show_data 是否显示用于建模的数据点（地质界面点surface_points + 产状点orientation_points）
    # if 'show_data' not in kwargs:
    #     kwargs['show_data'] = True

    # show_lith 控制是否显示建模的外部方块轮廓
    if 'show_lith' not in kwargs:
        kwargs['show_lith'] = False

    # show_surfaces 控制建模方式
    # if 'show_surfaces' not in kwargs:
    #     kwargs['show_surfaces'] = True

    # show_boundaries 控制是否显示建模边界
    # if 'show_boundaries' not in kwargs:
    #     kwargs['show_boundaries'] = False

    plotter = gpv.plot_3d(geo_model, show=False, **kwargs).p

    # 控制不显示坐标轴
    for func in [plotter.hide_axes, plotter.remove_bounds_axes]:
        try:
            func()
        except:
            pass

    for name, actor in list(plotter.renderer.actors.items()):
        if any(k in name.lower() for k in ["axes"]):
            plotter.remove_actor(actor)

    plotter.set_background("white")
    plotter.render()
    plotter.show()
    return plotter

import gempy_viewer as gpv

def clean_plot_2d(geo_model, **kwargs):
    """
    Clean 2D plot compatible with current GemPy version.
    Plot2D object has .create_figure() instead of .plotter.
    """
    kwargs.setdefault("plot_lith", True)
    kwargs.setdefault("legend", True)

    plot2d = gpv.plot_2d(geo_model, show=False, **kwargs)

    # Newer GemPy: Plot2D has create_figure() method, not .plotter
    if hasattr(plot2d, 'create_figure'):
        fig, axes = plot2d.create_figure()
        import matplotlib.pyplot as plt
        out_dir = os.path.dirname(os.path.abspath(__file__))
        fig.savefig(os.path.join(out_dir, "gempy_2d_section.png"), dpi=150,
                    bbox_inches='tight', facecolor='white')
        plt.close(fig)
        print("2D section saved to gempy_2d_section.png")
        return fig
    elif hasattr(plot2d, 'plotter'):
        plotter = plot2d.plotter
        plotter.set_background("white")
        if hasattr(plotter, "remove_bounds_axes"):
            plotter.remove_bounds_axes()
        if hasattr(plotter, "hide_axes"):
            plotter.hide_axes()
        plotter.show()
        return plotter
    else:
        print("Warning: plot_2d returned unexpected type:", type(plot2d))
        return plot2d