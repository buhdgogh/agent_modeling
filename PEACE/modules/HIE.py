import os
os.sys.path.append(f"{os.path.dirname(os.path.realpath(__file__))}/..")
import cv2
import json
import collections
from ..tool_pool import geological_knwoledge_type
from ..utils import api, prompt, vision, common
from ..agents import geologist_agent

class hierarchical_information_extraction:
    def __init__(self):
        self.geologist = geologist_agent()

    def digitalize(self, image_path):
        image = cv2.imread(image_path)
        name = common.path2name(image_path)
        meta_path = os.path.join(common.cache_path(), "meta", name + ".json")
        if os.path.exists(meta_path):
            meta = json.loads(open(meta_path).read())
            return meta

        # get headers of geologic map.
        meta = dict()
        meta["date"] = common.today_date()
        meta["name"] = name
        meta["version"] = "v1.0"
        meta["source"] = common.dataset_source
        meta["size"] = {"width": image.shape[1], "height": image.shape[0]}
        meta["regions"] = dict()
        meta["legend"] = dict()
        meta["information"] = dict()
        meta["faults"] = None

        # get layout of geologic map.
        map_layout = self.geologist.get_map_layout(image_path)
        regions = map_layout["regions"]
        meta["regions"] = regions

        # crop each component in geologic map.
        region_path_and_bbox = collections.defaultdict(list)
        for region_name, region_bndboxes in regions.items():
            for i, region_bndbox in enumerate(region_bndboxes):
                region_path = os.path.join(common.cache_path(), "det", name, f"{region_name}_{i}.png")
                common.create_folder_by_file_path(region_path)
                vision.crop_and_save_image(image, region_bndbox, region_path)
                region_path_and_bbox[region_name].append((region_path, region_bndbox))

                if "main_map" == region_name:
                    # crop latitude and longitude region of geologic map.
                    lonlat_name = "lonlat"
                    lonlat_region_path = os.path.join(common.cache_path(), "det", name, f"{lonlat_name}_{i}.png")
                    common.create_folder_by_file_path(lonlat_region_path)
                    vision.crop_corners_and_save_image(region_path, lonlat_region_path)
                    region_path_and_bbox[lonlat_name].append((lonlat_region_path, None))
                elif "index_map" == region_name:
                    # extend index map and add vision prompt.
                    vision.annotate_image_with_directions(region_path, region_path)

        # get metadata of legend.
        if len(region_path_and_bbox["legend"]) > 0:
            legend_path, legend_bndbox = region_path_and_bbox["legend"][0]
            legend_metadata = self.geologist.get_legend_metadata(legend_path, legend_bndbox)
            legends = legend_metadata["legend"]
            for legend in legends.values():
                legend["lithology"] = self.geologist.get_knowledge(geological_knwoledge_type.Rock_Type, legend["text"])["rock_type"]
                legend["stratigraphic_age"] = self.geologist.get_knowledge(geological_knwoledge_type.Rock_Age, legend["text"])["rock_age"]
            meta["legend"] = legends

        # get basic information of geologic map.
        region_names = ["title", "scale", "lonlat", "index_map"]
        for region_name in region_names:
            if region_name not in region_path_and_bbox:
                continue
            region_path, region_bndbox = region_path_and_bbox[region_name][0]
            keys, instruction = prompt.get_component_instruction(region_name)
            prompt_content = [
                {"type": "image_url", "image_url": {"url": api.local_image_to_data_url(region_path)}},
                {"type": "text", "text": instruction},
            ]
            messages = [
                {"role": "system", "content": prompt.system_prompt},
                {"role": "user", "content": prompt_content},
            ]

            answer = api.answer_wrapper(messages, structured=True)
            infos = eval(answer)
            key_value_pairs = prompt.get_basic_information(region_name, infos)
            for key, value in key_value_pairs:
                meta["information"][key] = value

        # get statistic information of rock.
        if len(region_path_and_bbox["main_map"]) > 0:
            main_map_path, main_map_bndbox = region_path_and_bbox["main_map"][0]
            vision.rock_region_seg(main_map_path, list(meta["legend"].values()))

        # output digitalization result of geologic map.
        common.create_folder_by_file_path(meta_path)
        with open(meta_path, "w", encoding="utf-8") as f:
            f.write(json.dumps(meta, indent=4, ensure_ascii=False))
        return meta


if __name__ == "__main__":
    image_path = "sample.jpg"
    hie = hierarchical_information_extraction()
    meta = hie.digitalize(image_path)
    print(meta)
