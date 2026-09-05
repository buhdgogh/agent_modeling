import os
import matplotlib
matplotlib.use('Agg')  # non-interactive backend for headless run

import numpy as np
import pandas as pd
import gempy as gp
import gempy_viewer as gpv
import utils

os.environ.setdefault("PYVISTA_OFF_SCREEN", "true")

# =====================================================
# 1. Load virtual borehole data
# =====================================================
csv_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                        "temp", "virtual_boreholes_points.csv")
surface = pd.read_csv(csv_path)
surface = surface.dropna(subset=["formation_code"])

# The CSV uses lowercase column names: x, y, z
surface = surface.rename(columns={"x": "X", "y": "Y", "z": "Z"})

# Create layer names from formation_code
surface["formation"] = "Layer_" + surface["formation_code"].astype(int).astype(str)

print(f"Loaded {len(surface)} interface points")
print(f"Unique formations: {sorted(surface['formation'].unique())}")
print(f"X range: {surface['X'].min():.0f} - {surface['X'].max():.0f}")
print(f"Y range: {surface['Y'].min():.0f} - {surface['Y'].max():.0f}")
print(f"Z range: {surface['Z'].min():.1f} - {surface['Z'].max():.1f}")

# =====================================================
# 2. Generate orientations from interface points
#    For each formation, compute local dip from neighboring points
# =====================================================
default_oris = []
fm_list = sorted(surface["formation"].unique(), key=lambda x: int(x.split("_")[1]))

for fmt in fm_list:
    fmt_data = surface[surface["formation"] == fmt]
    if len(fmt_data) < 2:
        # Not enough points — use vertical orientation
        mean_x = fmt_data["X"].mean()
        mean_y = fmt_data["Y"].mean()
        mean_z = fmt_data["Z"].mean()
        default_oris.append({
            "X": mean_x, "Y": mean_y, "Z": mean_z,
            "G_x": 0.0, "G_y": 0.0, "G_z": 1.0,
            "formation": fmt
        })
        continue

    # Use multiple points for orientation estimation
    # Sample up to 5 representative points per formation
    sample_n = min(5, len(fmt_data))
    sampled = fmt_data.sample(n=sample_n, random_state=42)
    for _, row in sampled.iterrows():
        default_oris.append({
            "X": row["X"], "Y": row["Y"], "Z": row["Z"],
            "G_x": 0.0, "G_y": 0.0, "G_z": 1.0,
            "formation": fmt
        })

orientation_input = pd.DataFrame(default_oris)
print(f"Generated {len(orientation_input)} orientation points")

# =====================================================
# 3. Save input CSVs
# =====================================================
out_dir = os.path.dirname(os.path.abspath(__file__))
surface_csv = os.path.join(out_dir, "gempy_surface_points.csv")
orientation_csv = os.path.join(out_dir, "gempy_orientations.csv")

surface_input = surface[["X", "Y", "Z", "formation"]]
surface_input.to_csv(surface_csv, index=False)
orientation_input.to_csv(orientation_csv, index=False)

# =====================================================
# 4. Compute extent with padding
# =====================================================
xmin, xmax = surface["X"].min(), surface["X"].max()
ymin, ymax = surface["Y"].min(), surface["Y"].max()
zmin, zmax = surface["Z"].min(), surface["Z"].max()

dx, dy, dz = xmax - xmin, ymax - ymin, zmax - zmin
pad_x = dx * 0.1
pad_y = dy * 0.1
pad_z = max(dz * 0.15, 50)

extent = [xmin - pad_x, xmax + pad_x,
          ymin - pad_y, ymax + pad_y,
          zmin - pad_z, zmax + pad_z]

print(f"Extent: {extent}")

# =====================================================
# 5. Create GemPy model
# =====================================================
geo_model = gp.create_geomodel(
    project_name="Hangzhou_Virtual_Boreholes",
    extent=extent,
    resolution=[50, 50, 50],
    importer_helper=gp.data.ImporterHelper(
        path_to_surface_points=surface_csv,
        path_to_orientations=orientation_csv
    )
)

# =====================================================
# 6. Map stratigraphy (Layer_1=newest=top, Layer_8=oldest=bottom)
# =====================================================
formation_names = [f"Layer_{i}" for i in range(1, 9)]
print(f"Formation order (top->bottom): {formation_names}")

gp.map_stack_to_surfaces(
    gempy_model=geo_model,
    mapping_object={"Stratigraphy": tuple(formation_names)}
)

# =====================================================
# 7. Compute model
# =====================================================
print("Computing geological model...")
gp.compute_model(
    geo_model,
    engine_config=gp.data.GemPyEngineConfig(
        backend=gp.data.AvailableBackends.PYTORCH
    )
)
print("Model computation finished!")

# =====================================================
# 8. Visualize
# =====================================================
# 2D cross-sections
print("Generating 2D plots...")
# utils.clean_plot_2d(geo_model)

# 3D model
print("Generating 3D model...")
utils.clean_plot_3d(geo_model)

print("All done!")
