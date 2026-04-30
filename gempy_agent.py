import os
import time
import math
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

        if not shp_path or not csv_path:
            result_dict["status"] = "需同时提供【SHP面文件】和【特征组合.csv】。"
            return {"borehole_result": result_dict}

        print(f">>> [BoreholeAgent] 正在根据特征配置文件与空间边界下钻...")
        try:
            def infer_utm_epsg(lon, lat):
                zone = int((lon + 180) // 6) + 1
                if lat >= 0:
                    return 32600 + zone  # WGS84 / UTM 北半球
                else:
                    return 32700 + zone  # WGS84 / UTM 南半球

            formation_df = pd.read_csv(csv_path)
            gdf = gpd.read_file(shp_path)
            pixel_size = 0.01

            minx, miny, maxx, maxy = gdf.total_bounds
            width = int(np.ceil((maxx - minx) / pixel_size))
            height = int(np.ceil((maxy - miny) / pixel_size))

            transform = from_origin(minx, maxy, pixel_size, pixel_size)
            shapes = [(geom, val) for geom, val in zip(gdf.geometry, gdf['Id'])]
            raster = rasterize(
                shapes,
                out_shape=(height, width),
                transform=transform,
                fill=0,
                dtype="int32"
            )
            n_rows, n_cols = raster.shape

            data = []
            for row in range(n_rows):
                for col in range(n_cols):
                    value = raster[row, col]
                    if value == 0:
                        continue

                    x = transform[2] + (col + 0.5) * transform[0]
                    y = transform[5] + (row + 0.5) * transform[4]
                    data.append((x, y, value))

            df = pd.DataFrame(data, columns=['x', 'y', 'value'])

            # === 1. 打开 DEM ===
            if tif_path:
                with rasterio.open(tif_path) as src:
                    dem_crs = src.crs
                    transformer_to_dem = Transformer.from_crs("EPSG:4326", dem_crs, always_xy=True)
                    xs_dem, ys_dem = transformer_to_dem.transform(df['x'].values, df['y'].values)
                    coords_for_dem = list(zip(xs_dem, ys_dem))

                    z_values = []
                    nodata = src.nodata
                    for val in src.sample(coords_for_dem):
                        v = float(val[0])
                        if np.isnan(v) or (nodata is not None and np.isclose(v, nodata)):
                            z_values.append(np.nan)
                        else:
                            z_values.append(v)

                df['z'] = z_values
                df = df[np.isfinite(df['z'])].copy()
                df = df[df['z'] > 0].copy()
            else:
                df['z'] = 0.0

            if not df.empty:
                lon0, lat0 = df['x'].mean(), df['y'].mean()
                utm_epsg = infer_utm_epsg(lon0, lat0)
                transformer_to_utm = Transformer.from_crs("EPSG:4326", f"EPSG:{utm_epsg}", always_xy=True)
                x_m, y_m = transformer_to_utm.transform(df['x'].values, df['y'].values)
                df['x_m'] = x_m
                df['y_m'] = y_m
            else:
                df['x_m'] = []
                df['y_m'] = []

            # === 读取 CSV 中的年代与特征配置 ===
            profile_data = {}
            if 'part_code' in formation_df.columns:
                group_iter = formation_df.groupby('part_code')
            else:
                group_iter = [('part_1', formation_df)]

            for part_code, group in group_iter:
                profile_data[part_code] = []
                for _, row_f in group.iterrows():
                    try:
                        thick_val = float(row_f.get('厚度', 10))
                    except:
                        thick_val = 10.0

                    profile_data[part_code].append({
                        'formation_code': str(row_f.get('formation_code', 'Unknown')) if pd.notna(
                            row_f.get('formation_code')) else 'Unknown',
                        'thickness_m': thick_val
                    })

            # === 下钻堆层 ===
            records = []
            for _, row in df.iterrows():
                x = row['x_m']
                y = row['y_m']
                z_top = float(row['z'])
                part_code = f'part_{int(row["value"])}'

                if part_code not in profile_data:
                    if 'part_1' in profile_data:
                        part_code = 'part_1'
                    else:
                        continue

                current_z = z_top
                for layer in profile_data[part_code]:
                    thickness_m = layer['thickness_m']
                    area_id = row['value']
                    records.append({
                        'x': x,
                        'y': y,
                        'z': round(current_z, 6),
                        'formation_code': layer['formation_code'],
                        'value': area_id
                    })
                    current_z -= thickness_m

            stacked_df = pd.DataFrame(records)

            if not stacked_df.empty:
                # 补充 surface 和 borehole_id 字段
                stacked_df['surface'] = stacked_df['formation_code'].astype(str)
                stacked_df['borehole_id'] = stacked_df.groupby(['x', 'y']).ngroup().apply(lambda i: f'BH{i:03d}')

                df_bh = stacked_df
                # ==========================================
                # 🌟 重新编排导出的列顺序，严格匹配目标字段
                # ==========================================
                target_cols = [
                    'x', 'y', 'z', 'formation_code', 'value', 'surface', 'borehole_id'
                ]

                for c in target_cols:
                    if c not in df_bh.columns:
                        df_bh[c] = None

                # 提取并排序
                df_bh = df_bh[target_cols]

                bh_csv_path = os.path.join(self.temp_dir, "virtual_boreholes_points.csv")
                df_bh.to_csv(bh_csv_path, index=False, encoding='utf-8-sig')

                result_dict["virtual_boreholes_data"] = safe_to_records(df_bh.head(200))
                result_dict["csv_file_path"] = bh_csv_path
                result_dict["total_rows"] = len(df_bh)
                result_dict["status"] = "Success"

                print(f">>> [BoreholeAgent] 提取结束！生成 {len(df_bh)} 行严格标准格式数据。")
            else:
                result_dict["status"] = "未能提取到数据，请检查特征边界与区域匹配。"

        except Exception as e:
            result_dict["status"] = f"提取崩溃: {traceback.format_exc()}"

        return {"borehole_result": result_dict}