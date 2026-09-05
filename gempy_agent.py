import os
import time
import math
import json
import numpy as np
import pandas as pd
import traceback
import warnings

# === 屏蔽底层警告 ===
warnings.filterwarnings("ignore")

import geopandas as gpd
import rasterio
from rasterio.features import rasterize
from rasterio.transform import from_origin
import rasterio.mask
from pyproj import Transformer


def safe_to_records(df):
    """将 DataFrame 转换为 100% 兼容 MySQL JSON 的格式，供前端无错预览"""
    if df is None or df.empty: return []
    df_clean = df.copy()
    records = df_clean.to_dict('records')
    safe_records = []
    for row in records:
        safe_row = {}
        for k, v in row.items():
            if pd.isna(v):
                safe_row[k] = None
            elif isinstance(v, (bool, np.bool_)):
                safe_row[k] = bool(v)
            elif isinstance(v, (np.integer, int)):
                safe_row[k] = int(v)
            elif isinstance(v, (np.floating, float)):
                val = float(v)
                if math.isnan(val) or math.isinf(val):
                    safe_row[k] = None
                else:
                    safe_row[k] = val
            else:
                safe_row[k] = str(v)
        safe_records.append(safe_row)
    return safe_records


class GempyAgent:
    """支持解析附带多重年代与代号的全新特征组合矩阵，并精简最终输出字段"""

    def __init__(self, api_key: str = "", model_name: str = ""):
        self.temp_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "temp")
        if not os.path.exists(self.temp_dir):
            os.makedirs(self.temp_dir)

    def run(self, text: str, file_paths: str = ""):
        result_dict = {
            "status": "Started",
            "virtual_boreholes_data": [],
            "csv_file_path": "",
            "total_rows": 0
        }

        shp_path = None
        tif_path = None
        csv_path = None
        regions_json_path = None

        # ==========================================
        # 1. 动态获取用户上传的文件地址
        # ==========================================
        if file_paths:
            paths = file_paths.split("|")
            for p in paths:
                p = p.strip()
                if not os.path.exists(p): continue
                lower_p = p.lower()

                if lower_p.endswith('.shp'):
                    shp_path = p
                elif lower_p.endswith(('.tif', '.tiff')):
                    tif_path = p
                elif lower_p.endswith('.csv'):
                    csv_path = p
                elif lower_p.endswith('.json') and ('region' in lower_p or 'regions_ui' in lower_p):
                    regions_json_path = p

        if not shp_path or not csv_path:
            result_dict["status"] = "需同时提供【SHP面文件】和【特征组合.csv】。"
            return {"borehole_result": result_dict}

        print(f">>> [BoreholeAgent] 正在根据特征配置文件与空间边界下钻...")
        try:
            formation_df = pd.read_csv(csv_path)
            gdf = gpd.read_file(shp_path)
            pixel_size = 0.005

            # ==========================================
            # Step 1: Vectorized raster grid extraction
            # ==========================================
            minx, miny, maxx, maxy = gdf.total_bounds
            width = int(np.ceil((maxx - minx) / pixel_size))
            height = int(np.ceil((maxy - miny) / pixel_size))
            transform = from_origin(minx, maxy, pixel_size, pixel_size)

            shapes = [(geom, val) for geom, val in zip(gdf.geometry, gdf['Id'])]
            raster = rasterize(shapes, out_shape=(height, width), transform=transform, fill=0, dtype="int32")

            # Vectorized: find all valid cells and compute coords in one shot
            rows, cols = np.where(raster > 0)
            values = raster[rows, cols]
            xs = transform[2] + (cols + 0.5) * transform[0]
            ys = transform[5] + (rows + 0.5) * transform[4]
            n_cells = len(xs)

            # ==========================================
            # Step 2: DEM sampling (vectorized)
            # ==========================================
            if n_cells == 0:
                result_dict["status"] = "SHP 范围内无有效栅格单元。"
                return {"borehole_result": result_dict}

            if tif_path:
                with rasterio.open(tif_path) as src:
                    dem_crs = src.crs
                    nodata = src.nodata
                    t_dem = Transformer.from_crs("EPSG:4326", dem_crs, always_xy=True)
                    xs_dem, ys_dem = t_dem.transform(xs, ys)
                    z_raw = np.array([float(v[0]) for v in src.sample(np.column_stack([xs_dem, ys_dem]))])
                    # Only filter truly invalid values (NaN/Inf), keep all valid and near-zero
                    valid = np.isfinite(z_raw)
                    if nodata is not None:
                        valid &= ~np.isclose(z_raw, nodata, atol=1e-3)
                    # For invalid cells, use median of valid ones as fallback
                    fallback_z = float(np.median(z_raw[valid])) if valid.any() else 100.0
                    z_vals = np.where(valid, z_raw, fallback_z)
                    z_vals = np.maximum(z_vals, 1.0)  # minimum 1m elevation
                    print(f">>> [BoreholeAgent] DEM: {valid.sum()}/{n_cells} cells valid, fallback Z={fallback_z:.0f}m")
            else:
                z_vals = np.full(n_cells, 100.0)  # default 100m if no DEM
                print(f">>> [BoreholeAgent] No DEM, using default Z=100m")

            # ==========================================
            # Step 3a: Surface formation from region polygons (if available)
            # ==========================================
            surface_formation = np.full(n_cells, -1, dtype=int)  # -1 = no region constraint

            if regions_json_path and os.path.exists(regions_json_path):
                try:
                    with open(regions_json_path, 'r', encoding='utf-8') as f:
                        regions_data = json.load(f)

                    # Get image dimensions for pixel→WGS84 transform
                    # Image pixels (0..img_w, 0..img_h) → WGS84 (minx..maxx, maxy..miny)
                    img_w, img_h = 1, 1
                    meta_json = regions_json_path.replace('regions_ui.json', 'meta.json')
                    if os.path.exists(meta_json):
                        with open(meta_json, 'r', encoding='utf-8') as f:
                            meta = json.load(f)
                        sz = meta.get('size', {})
                        img_w = sz.get('width', 1) or 1
                        img_h = sz.get('height', 1) or 1

                    # Affine: pixel_x → lon, pixel_y → lat
                    # Image top-left = (minx, maxy), bottom-right = (maxx, miny)
                    def px_to_wgs84(px, py):
                        lon = minx + (px / img_w) * (maxx - minx)
                        lat = maxy - (py / img_h) * (maxy - miny)
                        return lon, lat

                    # Build polygon shapes in WGS84 coordinates
                    region_shapes = []
                    for r in regions_data:
                        contour = r.get('contour', [])
                        if len(contour) < 3:
                            continue
                        try:
                            wgs84_contour = [px_to_wgs84(pt[0], pt[1]) for pt in contour]
                            from shapely.geometry import Polygon as ShapelyPolygon
                            poly = ShapelyPolygon(wgs84_contour)
                            if poly.is_valid and not poly.is_empty:
                                legend_id = r.get('matched_legend_id', -1)
                                region_shapes.append((poly, legend_id))
                        except Exception:
                            continue

                    if region_shapes:
                        region_raster = rasterize(
                            region_shapes, out_shape=(height, width),
                            transform=transform, fill=-1, dtype="int32"
                        )
                        surface_formation = region_raster[rows, cols]
                        n_matched = np.sum(surface_formation >= 0)
                        print(f">>> [BoreholeAgent] Region constraint: {len(region_shapes)} polygons, "
                              f"{n_matched}/{n_cells} cells ({100*n_matched/max(n_cells,1):.0f}%) constrained")
                except Exception as e:
                    print(f">>> [BoreholeAgent] Region load failed: {str(e)[:80]}, using uniform stack")

            # Build legend_id → formation_code lookup (bucketed by Y-rank)
            legend_to_fc = {}
            num_fms = len(formation_df)
            if regions_json_path and num_fms > 0:
                legend_json = regions_json_path.replace('regions_ui.json', 'legend_info.json')
                if os.path.exists(legend_json):
                    try:
                        with open(legend_json, 'r', encoding='utf-8') as f:
                            _jl = json.load(f)
                        sorted_legends = sorted(_jl, key=lambda x: x['color_bbox'][1]
                                               if x['color_bbox'][1] != -1 else 99999)
                        n_leg = len(sorted_legends)
                        for rank, leg in enumerate(sorted_legends):
                            # Bucket ranks into num_fms bins: rank 0→FC1, rank n_leg-1→FC{num_fms}
                            fc = min(int(rank * num_fms / n_leg) + 1, num_fms)
                            legend_to_fc[leg['id']] = fc
                        print(f">>> [BoreholeAgent] Region constraint: {len(legend_to_fc)} legend IDs "
                              f"mapped to {num_fms} formation codes (Y-rank bucketing)")
                    except Exception as e:
                        print(f">>> [BoreholeAgent] legend_info load failed: {str(e)[:80]}")

            # ==========================================
            # Step 3b: UTM coordinate transform (vectorized)
            # ==========================================
            lon0, lat0 = xs.mean(), ys.mean()
            zone = int((lon0 + 180) // 6) + 1
            utm_epsg = 32600 + zone if lat0 >= 0 else 32700 + zone
            t_utm = Transformer.from_crs("EPSG:4326", f"EPSG:{utm_epsg}", always_xy=True)
            x_m, y_m = t_utm.transform(xs, ys)
            x_m, y_m = np.array(x_m), np.array(y_m)

            # ==========================================
            # Step 4: Build formation profile config
            # ==========================================
            if 'part_code' in formation_df.columns:
                groups = formation_df.groupby('part_code')
            else:
                groups = [('part_1', formation_df)]

            # Unified profile: part_code -> [(formation_code, thickness_m), ...]
            profile_data = {}
            for pc, grp in groups:
                layers = []
                for _, r in grp.iterrows():
                    try:
                        tv = float(r.get('厚度', 30))
                        if pd.isna(tv) or tv <= 0:
                            tv = 30.0
                    except Exception:
                        tv = 30.0
                    fc = str(r['formation_code']) if pd.notna(r.get('formation_code')) else 'Unknown'
                    layers.append((fc, tv))
                profile_data[pc] = layers

            # Map each cell's part_code to its layer list
            cell_part_codes = np.array([f'part_{int(v)}' for v in values])
            # Default to part_1 if code not found
            for i, pc in enumerate(cell_part_codes):
                if pc not in profile_data:
                    cell_part_codes[i] = 'part_1'

            # ==========================================
            # Step 5: Vectorized geological variation pre-computation
            # ==========================================
            np.random.seed(42)
            rng = np.random.RandomState(42)

            # Spatial factors (mild edge thinning: 0.80~1.0)
            cx_m, cy_m = x_m.mean(), y_m.mean()
            max_dist = np.sqrt((x_m - cx_m)**2 + (y_m - cy_m)**2).max() or 1.0
            dists = np.sqrt((x_m - cx_m)**2 + (y_m - cy_m)**2)
            edge_factors = 1.0 - 0.20 * np.minimum(dists / max_dist, 1.0)

            # XY jitter (one per cell)
            xy_jitter = pixel_size * 111320.0 * 0.35
            x_jittered = x_m + rng.uniform(-xy_jitter, xy_jitter, n_cells)
            y_jittered = y_m + rng.uniform(-xy_jitter, xy_jitter, n_cells)

            # Z surface noise
            z_surface_noise = rng.uniform(-3.0, 3.0, n_cells)

            # Build all borehole records using numpy
            # Pre-allocate: max total records = n_cells * max_layers_per_part
            max_layers = max(len(v) for v in profile_data.values()) if profile_data else 0
            if max_layers == 0:
                result_dict["status"] = "CSV 中无有效地层配置。"
                return {"borehole_result": result_dict}

            # Pre-generate random numbers
            pinch_rand = rng.random((n_cells, max_layers))
            perturb_rand = np.clip(rng.uniform(0.80, 1.60, (n_cells, max_layers)), 0.5, 3.0)
            iface_rand = rng.uniform(-3.0, 3.0, (n_cells, max_layers))

            rec_x, rec_y, rec_z, rec_fc, rec_val = [], [], [], [], []

            for i in range(n_cells):
                layers = profile_data.get(cell_part_codes[i], profile_data.get('part_1', []))
                if not layers:
                    layers = profile_data.get('part_1', [])
                    if not layers:
                        continue

                # Get per-cell parameters with explicit type conversion and safety
                z_cur = float(np.nan_to_num(z_vals[i], nan=50.0)) + float(np.nan_to_num(z_surface_noise[i], nan=0.0))
                ef = float(np.nan_to_num(edge_factors[i], nan=0.9))
                xj = float(np.nan_to_num(x_jittered[i], nan=230000.0))
                yj = float(np.nan_to_num(y_jittered[i], nan=3365000.0))
                area_val = int(values[i])

                start_j = 0
                sf = int(surface_formation[i]) if surface_formation is not None else -1
                if sf >= 0 and sf in legend_to_fc:
                    target_fc = legend_to_fc[sf]
                    for j, (fc, _) in enumerate(layers):
                        try:
                            if int(fc) == target_fc:
                                start_j = j; break
                        except: pass

                for j, (fc, base_t) in enumerate(layers):
                    if j < start_j:
                        continue

                    # Spatially-varying pinch-out
                    pinch_p = 0.03 + 0.03 * j + 0.08 * (1.0 - ef)
                    if float(pinch_rand[i, j]) < pinch_p:
                        continue

                    # Thickness with perturbation
                    thick = base_t * float(perturb_rand[i, j]) * ef * (1.0 - 0.015 * (j + 1))
                    thick = max(thick, 5.0)

                    # Interface wave
                    z_cur -= float(iface_rand[i, j])

                    # Record point
                    rec_x.append(round(xj, 3))
                    rec_y.append(round(yj, 3))
                    rec_z.append(round(z_cur, 3))
                    rec_fc.append(fc)
                    rec_val.append(area_val)

                    # Move down
                    z_cur -= thick

            stacked_df = pd.DataFrame({
                'x': rec_x, 'y': rec_y, 'z': rec_z,
                'formation_code': rec_fc, 'value': rec_val
            })

            if not stacked_df.empty:
                # Vectorized borehole_id: use factorize on (x, y) tuples
                xy_tuples = list(zip(stacked_df['x'], stacked_df['y']))
                codes, _ = pd.factorize(xy_tuples)
                stacked_df['borehole_id'] = [f'BH{c:03d}' for c in codes]
                stacked_df['surface'] = stacked_df['formation_code'].astype(str)

                target_cols = ['x', 'y', 'z', 'formation_code', 'value', 'surface', 'borehole_id']
                df_bh = stacked_df[target_cols]

                bh_csv_path = os.path.join(self.temp_dir, "virtual_boreholes_points.csv")
                df_bh.to_csv(bh_csv_path, index=False, encoding='utf-8-sig')

                # Fast preview: use .values conversion instead of row-by-row
                preview = df_bh.head(200)
                result_dict["virtual_boreholes_data"] = preview.where(pd.notna(preview), None).to_dict('records')
                result_dict["csv_file_path"] = bh_csv_path
                result_dict["total_rows"] = len(df_bh)
                result_dict["status"] = "Success"

                print(f">>> [BoreholeAgent] 提取结束！生成 {len(df_bh)} 行严格标准格式数据。")
            else:
                result_dict["status"] = "未能提取到数据，请检查特征边界与区域匹配。"

        except Exception as e:
            result_dict["status"] = f"提取崩溃: {traceback.format_exc()}"

        return {"borehole_result": result_dict}