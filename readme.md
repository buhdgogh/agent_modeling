# 多智能体多模态虚拟钻孔生成系统

面向三维地质建模中真实钻孔获取成本高、空间覆盖有限的问题，本项目构建了一套多智能体虚拟钻孔数据生成原型。系统能够解析地质报告和地质图件，融合地层、岩性、年代、图例、地质边界及 DEM 高程信息，在空间与地层规则约束下生成标准化虚拟钻孔点云，并提供知识图谱、实验性 GemPy 建模和数据质量验证能力。

> 项目当前定位为科研原型。生成结果用于三维地质建模研究和方法验证，不能替代真实钻探、测绘或工程勘察结论。

## 功能概览

- 地质文本抽取：从非结构化报告中提取地层、年代、岩性、厚度、剖面及上下覆关系。
- 地质图件解析：检测图例与地图区域，识别地层符号，并按照颜色和纹理匹配图例与地质区域。
- 多模态融合：并行处理文本和图像信息，通过大模型完成地层实体对齐，生成标准地层配置。
- 虚拟钻孔生成：结合 SHP 边界、DEM 高程、地层厚度、空间扰动和尖灭规则生成钻孔点云。
- 知识图谱构建：抽取地层、年代、岩性、构造和空间分布关系，可选写入 Neo4j。
- 质量评价：提供真实钻孔盲测、剖面 IoU、不确定性缩减、地层叠覆规则和空间一致性验证。
- 交互与持久化：通过 Streamlit 提供文件上传、任务执行与结果展示，并使用 MySQL 保存会话记录。

## 技术体系

- Agent 编排：LangChain、LangGraph
- 大模型：Qwen、Qwen-VL、DeepSeek（通过兼容的模型服务调用）
- 视觉处理：PyTorch、YOLO、OpenCV、PEACE
- 地理计算：GeoPandas、Rasterio、PyProj、Shapely
- 三维建模：GemPy、PyVista
- 数据与界面：MySQL、Neo4j、NetworkX、Streamlit

## 系统架构

```text
用户指令 + TXT/JPG/PNG/CSV/SHP/TIF
                  |
                  v
        MasterAgent（意图路由）
                  |
        +---------+----------+----------------+
        |                    |                |
        v                    v                v
 TextInfoAgent          ImageAgent       KGBuilderAgent
 文本结构化抽取       图例与区域解析       实体关系抽取
        |                    |                |
        +---------+----------+                +--> Neo4j（可选）
                  |
                  v
          LLM 地层对齐与融合
                  |
                  v
       标准地层配置 CSV + 空间约束
                  |
                  v
             GempyAgent
                  |
                  v
         虚拟钻孔点云 CSV
                  |
        +---------+----------+
        |                    |
        v                    v
 实验性 GemPy 建模        质量验证
```

### Agent 通信方式

当前系统采用中心化 Supervisor 模式：`MasterAgent` 将专业 Agent 封装为 LangChain Tool，通过 LangGraph 状态完成任务路由，再以 Python 函数调用和结构化对象传递结果。各 Agent 运行在同一进程内，不依赖消息队列或跨服务网络通信。

自动钻孔任务内部使用线程池并行执行文本抽取和图像解析，两个结果返回总控 Agent 后进入融合步骤。该设计保持了较低的通信开销，也便于在单机环境中调试和复现实验。

## 处理流程

### 1. 文本信息抽取

长篇地质报告经过重叠分块后并发提交给语言模型，并按照 Pydantic Schema 输出：

- 地层名称与代号
- 主、次地质年代及代号
- 分布区域与岩石特征
- 剖面位置、厚度和岩石组合
- 上覆与下伏地层
- 条目置信度

### 2. 地质图件解析

图像 Agent 使用 PEACE/YOLO 检测图例和地图主体，调用视觉语言模型识别图例文字与符号，再通过颜色距离和纹理直方图完成图例—区域匹配。解析结果包括图例信息、区域轮廓、匹配置信度和可视化图件。

### 3. 图文融合

系统并行获取文本地层实体和图像图例序列，随后由大模型完成名称、代号、年代与层序对齐，输出八列地层配置：

```text
part_code, formation, formation_code, formation_age_1,
formation_age_code_1, formation_age_2, formation_age_code_2, 厚度
```

### 4. 虚拟钻孔生成

虚拟下钻模块将 SHP 地质边界栅格化，采样 DEM 获取地表高程，并结合地层配置和图像区域约束生成三维点。输出字段为：

```text
x, y, z, formation_code, value, surface, borehole_id
```

其中 `x/y/z` 为三维坐标，`formation_code` 和 `surface` 表示地层，`value` 表示地质分区编号，`borehole_id` 表示钻孔编号。

## 项目结构

```text
agent_modeling/
|-- main.py                         # Streamlit 应用入口
|-- master_agent.py                 # 总控 Agent、任务路由和多模态融合
|-- text_info_agent.py              # 地质文本结构化抽取
|-- image_agent.py                  # 地质图例识别、区域分割与匹配
|-- kg_builder.py                   # 地质知识图谱构建
|-- gempy_agent.py                  # 空间约束下的虚拟钻孔点生成
|-- db_manager.py                   # MySQL 会话持久化
|-- neo4j_manager.py                # Neo4j 图谱持久化
|-- db.sql                          # MySQL 数据表结构
|-- requirements.txt                # Python 依赖锁定文件
|-- PEACE/                          # 图件检测模型、权重和地学工具
|-- experiments/
|   |-- modeling/                   # GemPy 建模实验
|   |-- quantitative/               # 定量验证
|   |-- qualitative/                # 定性验证
|   `-- run_validation.py           # 验证统一入口
|-- temp/                           # 中间数据与生成 CSV
`-- output/                         # 图像分析及论文图件输出
```

## 环境准备

建议在独立环境中安装依赖。GDAL、GeoPandas、Rasterio、VTK 和 GemPy 含本地二进制依赖，需要保证 Python 版本与安装包兼容。

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

如果直接通过 `pip` 安装地理计算依赖失败，可以先使用 Conda 创建包含 GDAL、GeoPandas 和 Rasterio 的环境，再安装其余依赖。

## 配置服务

复制配置模板：

```powershell
Copy-Item .api_key.env.example api_key.env
```

需要配置以下环境变量：

| 配置项 | 是否必需 | 用途 |
|---|---:|---|
| `DASHSCOPE_API_KEY` | 是 | Qwen/Qwen-VL 等模型调用 |
| `DB_HOST`、`DB_PORT` | 是 | MySQL 地址与端口 |
| `DB_USER`、`DB_PASSWORD`、`DB_NAME` | 是 | MySQL 认证与数据库名称 |
| `NEO4J_URI`、`NEO4J_USER`、`NEO4J_PASSWORD` | 否 | 知识图谱持久化 |

`db.sql` 只负责创建数据表。请先创建与 `DB_NAME` 一致的数据库，再导入表结构。默认数据库名为 `agent_chat_db`：

```powershell
mysql -u root -p -e "CREATE DATABASE IF NOT EXISTS agent_chat_db CHARACTER SET utf8mb4;"
cmd /c "mysql -u root -p agent_chat_db < db.sql"
```

不要提交包含真实密钥或密码的 `api_key.env`。

## 启动应用

```powershell
streamlit run main.py
```

浏览器打开 Streamlit 输出的本地地址后，可以在侧栏选择模型、管理历史会话并上传任务文件。

### 支持的输入

| 数据类型 | 扩展名 | 说明 |
|---|---|---|
| 地质报告 | `.txt` | 支持 UTF-8，并兼容常见 GBK 文本 |
| 地质图件 | `.jpg`、`.png` | 默认使用第一张图片执行图像任务 |
| 地层配置 | `.csv` | 可跳过自动图文融合，直接用于虚拟下钻 |
| 地质边界 | `.shp`、`.shx`、`.dbf`、`.prj`、`.cpg` | 辅助文件应与 SHP 同名并放在同一目录 |
| 高程模型 | `.tif`、`.tiff` | 用于采样钻孔口高程 |

### 输入约束

- 当前虚拟下钻实现要求 SHP 属性表包含 `Id` 字段。
- 当前空间计算按 SHP 坐标为 WGS84 经纬度处理；其他坐标系应先转换或扩展 CRS 处理逻辑。
- 自动生成钻孔至少需要 SHP 和地层配置 CSV；DEM 缺失时会使用默认高程。
- 图像轮廓到地理空间的映射目前依据图像范围和 SHP 包围盒进行线性转换，不等同于严格的地理配准。

### 输出位置

- `temp/auto_generated_formation.csv`：图文融合生成的地层配置。
- `temp/virtual_boreholes_points.csv`：生成的虚拟钻孔点云。
- `output/<图件名_时间戳>/`：图例、区域匹配和可视化结果。
- `experiments/output/`：验证报告、JSON 数据及统计图。

## 质量验证

运行内置模拟数据演示：

```powershell
python experiments/run_validation.py --demo
```

对真实输入运行全部验证：

```powershell
python experiments/run_validation.py --all `
  --virtual_csv temp/virtual_boreholes_points.csv `
  --formation_csv temp/auto_generated_formation.csv `
  --shp_path "path/to/geology.shp" `
  --dem_path "path/to/dem.tif"
```

如需执行真实钻孔盲测，可增加：

```powershell
--real_csv "path/to/real_boreholes.csv"
```

查看所有参数：

```powershell
python experiments/run_validation.py --help
```

验证模块覆盖：

1. 真实钻孔盲测：计算 RMSE、MAE 和决定系数等指标。
2. 地质剖面验证：提取指定剖面带内的钻孔点并支持二维拓扑 IoU 分析。
3. 不确定性评价：评估加入虚拟钻孔前后的信息熵变化。
4. 地层规则检查：检测层序倒置和厚度异常。
5. 空间一致性：评价孔口位置、表层地层匹配和空间平滑性。

## 实现边界

- 主应用中的 `GempyAgent` 当前负责虚拟钻孔点云生成，并未直接执行 GemPy 隐式模型求解。
- GemPy 建模代码位于 `experiments/modeling/`，用于读取钻孔点并开展实验性三维建模。
- Agent 编排目前是单次路由和工具执行，不是多个 Agent 自由对话或多轮协商。
- MySQL 保存会话和分析结果；Neo4j 不可用时，知识图谱仍可在内存中生成和展示。
- 模型输出、图像匹配和规则生成均存在不确定性，重要结果应结合专家审查和真实数据验证。

## 安全与版本管理

- `api_key.env`、运行缓存和生成目录已列入 `.gitignore`。
- 如果敏感配置曾经提交到 Git，仅添加忽略规则不能清除历史记录，应及时轮换凭据并清理仓库历史。
- PEACE 模型权重和部分实验数据体积较大，团队协作时可以考虑使用 Git LFS 或独立模型存储。
