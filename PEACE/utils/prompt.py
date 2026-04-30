import common
from enum import Enum

system_prompt = "You are an expert in geology and cartography with a focus on geological maps."

components = (
    "title",
    "main_map",
    "scale",
    "legend",
    "index_map", 
    "cross_section",
    "stratigraphic_column", 
    "others",
)

class question_type(Enum):
    multiple_choice = 1
    true_false = 2
    fill_in_the_blank = 3
    essay = 4

question_ability2type = {
    # extracting
    "extracting-sheet_name": question_type.fill_in_the_blank,
    "extracting-scale": question_type.fill_in_the_blank,
    "extracting-lonlat": question_type.fill_in_the_blank,
    "extracting-index_map": question_type.fill_in_the_blank,

    # grounding
    "grounding-title_by_name": question_type.fill_in_the_blank,
    "grounding-main_map_by_name": question_type.fill_in_the_blank,
    "grounding-scale_by_name": question_type.fill_in_the_blank,
    "grounding-legend_by_name": question_type.fill_in_the_blank,
    "grounding-index_map_by_name": question_type.fill_in_the_blank,
    "grounding-cross_section_by_name": question_type.fill_in_the_blank,
    "grounding-stratigraphic_column_by_name": question_type.fill_in_the_blank,

    "grounding-title_by_intention": question_type.fill_in_the_blank,
    "grounding-main_map_by_intention": question_type.fill_in_the_blank,
    "grounding-scale_by_intention": question_type.fill_in_the_blank,
    "grounding-legend_by_intention": question_type.fill_in_the_blank,
    "grounding-index_map_by_intention": question_type.fill_in_the_blank,
    "grounding-cross_section_by_intention": question_type.fill_in_the_blank,
    "grounding-stratigraphic_column_by_intention": question_type.fill_in_the_blank,

    # referring
    "referring-rock_by_color": question_type.multiple_choice,
    "referring-color_by_rock": question_type.multiple_choice,

    # reasoning
    "reasoning-lonlat_localization": question_type.multiple_choice,
    "reasoning-fault_existence": question_type.true_false,
    "reasoning-area_comparison": question_type.multiple_choice,
    "reasoning-lithology_composition": question_type.multiple_choice,

    # analyzing
    "analyzing-earthquake_risk": question_type.essay,
}

grounding_format =              '这是一道填空题，结合提供的文本信息和图片（宽：$width，高：$height），回答答案并给出分析过程，仅返回JSON格式结果，例如：{"answer": [x_min, y_min, x_max, y_max], "reason", "XXX"}' if common.dataset_source == "cgs" else \
                                'This is a fill-in-the-blank question. Based on the provided text and image (width: $width, height: $height), analyze and answer the question in JSON format only, for example: {"answer": [x_min, y_min, x_max, y_max], "reason", "XXX"}'
mcq_format =                    '这是一道选择题，结合提供的文本信息和图片（宽：$width，高：$height），回答答案并给出分析过程，仅返回JSON格式结果，例如：{"answer": "C", "reason": "XXX"}' if common.dataset_source == "cgs" else \
                                'This is a multiple-choice question. Based on the provided text and image (width: $width, height: $height), analyze and answer the question in JSON format only, for example: {"answer": "C", "reason": "XXX"}'
yes_no_format =                 '这是一道判断题，结合提供的文本信息和图片（宽：$width，高：$height），回答答案并给出分析过程，仅返回JSON格式结果，例如：{"answer": True, "reason": "XXX"}' if common.dataset_source == "cgs" else \
                                'This is a true/false question. Based on the provided text and image (width: $width, height: $height), analyze and answer the question in JSON format only, for example: {"answer": True, "reason": "XXX"}'
essay_format  =                 '这是一道问答题，结合提供的文本信息和图片（宽：$width，高：$height），回答答案并从不同方面给出详细的分析过程，仅返回JSON格式结果，例如：{"answer": "high risk", "reason": "XXX"}' if common.dataset_source == "cgs" else \
                                'This is a essay question. Based on the provided text and image (width: $width, height: $height), detailedly anylyze and answer the question from different aspects in JSON format only, for example: {"answer": "high risk", "reason": "1. XXX; 2. XXX; 3. XXX; ..."}'
question_ability2format = {
    # extracting
    "extracting-sheet_name": '这是一道填空题，结合提供的文本信息和图片（宽：$width，高：$height），回答答案并给出分析过程，仅返回JSON格式结果，例如：{"answer": "K-49-65(义县幅)", "reason": "XXX"}' if common.dataset_source == "cgs" else \
                             'This is a fill-in-the-blank question. Based on the provided text and image (width: $width, height: $height), analyze and answer the question in JSON format only, for example: {"answer": "Geologic map of the Valle 30\' x 60\' quadrangle, Coconino County, northern Arizona", "reason": "XXX"}',
    "extracting-scale":      '这是一道填空题，结合提供的文本信息和图片（宽：$width，高：$height），回答答案并给出分析过程，仅返回JSON格式结果，例如：{"answer": "1:100000", "reason": "XXX"}' if common.dataset_source == "cgs" else \
                             'This is a fill-in-the-blank question. Based on the provided text and image (width: $width, height: $height), analyze and answer the question in JSON format only, for example: {"answer": "1:100000", "reason": "XXX"}',
    "extracting-lonlat":     '这是一道填空题，结合提供的文本信息和图片（宽：$width，高：$height），回答答案并给出分析过程，经度范围在前，从西到东顺序，纬度范围在后，从南到北顺序，仅返回JSON格式结果，例如：{"answer": "%s", "reason": "XXX"}' % ("109°40'E-110°40'E,19°20'N-20°00'N'") if common.dataset_source == "cgs" else \
                             'This is a fill-in-the-blank question. Based on the provided text and image (width: $width, height: $height), analyze and answer the question in JSON format only, longitude range comes first, from west to east, followed by latitude range, from south to north, for example: {"answer": "%s", "reason": "XXX"}' % ("109°40'W-110°40'W,19°20'N-20°00'N'"),
    "extracting-index_map":  '这是一道填空题，结合提供的文本信息和图片（宽：$width，高：$height），回答答案并给出分析过程，仅返回JSON格式结果，例如：{"answer": ["乌石港", "海口市", "七洲列岛", "万宁市", "乐东县", "儋州市"], "reason": "XXX"}' if common.dataset_source == "cgs" else \
                             'This is a fill-in-the-blank question. Based on the provided text and image (width: $width, height: $height), analyze and answer the question in JSON format only, for example: {"answer": ["SHELBY", "WACO and BESSEMER CITY", "GASTONIA NORTH", "BLACKSBURG NORTH", "CLOVER"], "reason": "XXX"}',

    # grounding
    "grounding-title_by_name": grounding_format,
    "grounding-main_map_by_name": grounding_format,
    "grounding-scale_by_name": grounding_format,
    "grounding-legend_by_name": grounding_format,
    "grounding-index_map_by_name": grounding_format,
    "grounding-cross_section_by_name": grounding_format,
    "grounding-stratigraphic_column_by_name": grounding_format,

    "grounding-title_by_intention": grounding_format,
    "grounding-main_map_by_intention": grounding_format,
    "grounding-scale_by_intention": grounding_format,
    "grounding-legend_by_intention": grounding_format,
    "grounding-index_map_by_intention": grounding_format,
    "grounding-cross_section_by_intention": grounding_format,
    "grounding-stratigraphic_column_by_intention": grounding_format,

    # referring
    "referring-rock_by_color": mcq_format,
    "referring-color_by_rock": mcq_format,

    # reasoning
    "reasoning-lonlat_localization": mcq_format,
    "reasoning-fault_existence": yes_no_format,
    "reasoning-area_comparison": mcq_format,
    "reasoning-lithology_composition": mcq_format,

    # analyzing
    "analyzing-earthquake_risk": essay_format,
}

def ability2instruction(ability, image_size):
    height, width = image_size
    instruction = question_ability2format[ability]
    instruction = instruction.replace("$width", f"{width}").replace("$height", f"{height}")
    return instruction

def remove_format_requirement(question):
    pattern_cgs = "？"
    pattern_usgs = "?"
    s1 = question.find(pattern_cgs)
    s2 = question.find(pattern_usgs)
    if s1 >= 0 and s2 >= 2:
        s = min(s1, s2)
    elif s1 >= 0 and s2 == -1:
        s = s1
    elif s1 == -1 and s2 >= 0:
        s = s2
    else:
        s = len(question) - 1
    question_prefix = question[:s+1]
    return question_prefix

def format_question(question, meta):
    if meta["mcq"]:
        for key in ("A", "B", "C", "D"):
            value = meta[key]
            choice = f"\n{key}. {value}"
            question += choice
    return question

def get_component_instruction(component):
    component_keys = {
        "title": ("图幅名",) if common.dataset_source == "cgs" else \
                 ("title_name",),
        "scale": ("比例尺",) if common.dataset_source == "cgs" else \
                 ("scale",),
        "lonlat": ("经度范围", "纬度范围",) if common.dataset_source == "cgs" else \
                  ("longitude_range", "latitude_range",),
        "index_map": ("WN", "N", "EN", "W", "E", "WS", "S", "ES",),
    }
    component_examples = {
        "title": '{"图幅名": "G-47-04(中甸幅)"}' if common.dataset_source == "cgs" else \
                 '{"title_name": Geologic map of the Valle 30\' x 60\' quadrangle, Coconino County, northern Arizona"}',
        "scale": '{"比例尺": "1:XXX"}' if common.dataset_source == "cgs" else \
                 '{"scale": "1:XXX"}',
        "lonlat": '{"经度范围": "(X°X\'E, X°X\'E)", "纬度范围": "(X°X\'N, X°X\'N)"}' if common.dataset_source == "cgs" else \
                  '{"longitude_range": "(X°X\'W, X°X\'W)", "latitude_range": "(X°X\'N, X°X\'N)"}',
        "index_map": '{"WN": "弥勒 (G-48-32)", "N": "", "EN": "广南 (G-48-34)", "W": "个旧 (F-48-02)", "E": "", "WS": "金平 (F-48-08)", "S": "马关 (F-48-09)", "ES": "保乐 (F-48-10)"}' if common.dataset_source == "cgs" else \
                     '{"WN": "SHELBY", "N": "WACO and BESSEMER CITY", "EN": "GASTONIA NORTH", "W": "BLACKSBURG NORTH", "E": "GASTONIA SOUTH", "WS": "BLACKSBURG SOUTH", "S": "KINGS CREEK and FILBERT", "ES": "CLOVER"}',
    }
    component_instructions = {
        "title": "给定地质图的标题局部图片，以JSON格式，返回 图幅名，例如：${example}" if common.dataset_source == "cgs" else \
                 "Given a image of the title region of a geologic map, return the title name in JSON format, for example: ${example}",
        "scale": "给定地质图的比例尺局部图片，以JSON格式，返回 比例尺，例如：${example}" if common.dataset_source == "cgs" else \
                 "Given a image of the scale region of a geologic map, return the scale in JSON format, for example: ${example}",
        "lonlat": "给定中国（东北半球）地质图的经纬度局部图片，表示经度的文字一般横向排列，表示纬度的文字竖向排列，以JSON格式，返回 经度范围和纬度范围，例如：${example}" if common.dataset_source == "cgs" else \
                  "Given a image of the longitude and latitude of a geologic map, which locates in the United States (Northwestern Hemisphere), longitude is represented horizontally in text, while latitude is represented vertically, return the longitude range and latitude range in JSON format, for example: ${example}",
        "index_map": "给定地质图的接图表局部图片，以JSON格式，返回 8个方位的邻接区域信息，例如：${example}" if common.dataset_source == "cgs" else \
                     "Given a image of the index map region of a geologic map, return the names of adjacent areas in 8 directions in JSON format, for example: ${example}",
    }

    keys = component_keys[component]
    example = component_examples[component]
    instruction = component_instructions[component].replace("${example}", example)
    return keys, instruction

def get_basic_information(component, infos):
    if component == "title":
        title_name = infos.get("图幅名") if common.dataset_source == "cgs" else infos.get("title_name")
        return (("title", title_name),)
    elif component == "scale":
        scale = infos.get("比例尺") if common.dataset_source == "cgs" else infos.get("scale")
        scale = scale.replace(",", "").replace(" ", "")
        scale = list(map(int, scale.split(":")))
        return (("scale", scale),)
    elif component == "lonlat":
        longitude = infos.get("经度范围") if common.dataset_source == "cgs" else infos.get("longitude_range")
        if isinstance(longitude, str):
            longitude = longitude.replace(" ", "").strip("()").split(",")
        latitude = infos.get("纬度范围") if common.dataset_source == "cgs" else infos.get("latitude_range")
        if isinstance(latitude, str):
            latitude = latitude.replace(" ", "").strip("()").split(",")
        return (("longitude", longitude), ("latitude", latitude))
    elif component == "index_map":
        connection = infos
        connection["C"] = ""
        return (("connection", connection),)

def polish_information(meta):
    del meta["date"]
    #del meta["name"]
    del meta["version"]
    del meta["source"]
    del meta["regions"]["others"]
    del meta["faults"]
    
    lon_range = meta["information"]["longitude"]
    lon_range = [common.polish_lonlat(lon) for lon in lon_range]
    lon_range.sort()
    meta["information"]["longitude"] = lon_range

    lat_range = meta["information"]["latitude"]
    lat_range = [common.polish_lonlat(lat) for lat in lat_range]
    lat_range.sort()
    meta["information"]["latitude"] = lat_range

    legends = meta["legend"].values()
    meta["legend"] = sorted(list(filter(lambda legend: \
                                        legend["color_hex"] != "#FFFFFF" and \
                                        legend["color_name"] != "White" and \
                                        legend["text"] != "unknown",
                                        legends)), key=lambda legend: legend["area"], reverse=True)
    for legend in meta["legend"]:
        legend["rock_type"] = legend["text"]
        del legend["text"]
        legend["rock_area_in_main_map"] = legend["area"]
        del legend["area"]
        legend["rock_color_in_main_map"] = legend["color_hex"]
        del legend["color_hex"]

def get_final_answer(answer, question_type):
    question_type = question_ability2type[question_type]
    if question_type == question_type.true_false:
        if answer["answer"]:
            final_answer = "是" if common.dataset_source == "cgs" else "Yes"
        else:
            final_answer = "否" if common.dataset_source == "cgs" else "No"
    elif question_type == question_type.essay:
        try:
            final_answer = str(answer["reason"])
        except:
            final_answer = str(answer["answer"])
    else:
        final_answer = str(answer["answer"])
    return final_answer.strip()
