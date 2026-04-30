import re
import json
import os

class rock_type_and_age_db:
    def __init__(self, mode):
        self.rock_name2type = {}

        db_path = os.path.join(os.path.dirname(__file__), "..", "dependencies", "knowledge", f"k2_rock_{mode}.json")

        # ✅ 关键修复点：显式指定 UTF-8 编码
        with open(db_path, "r", encoding="utf-8") as f:
            rock_types = json.load(f)

        for item in rock_types:
            rock_name = item["rock_name"].lower()
            rock_value = item["rock_value"].lower()
            self.rock_name2type[rock_name] = rock_value

    def clean_rock_name(self, name):
        black_list1 = ["脉", "?", ")", "member", "."]
        for key in black_list1:
            name = name.replace(key, "").strip()

        black_list2 = ["夹", "（", "。", ":"]
        for key in black_list2:
            if key in name:
                s = name.find(key)
                name = name[:s].strip()

        black_list3 = ["的", "色", "—"]
        for key in black_list3:
            if key in name:
                s = name.find(key)
                name = name[s + 1:].strip()

        return name.strip()

    def rock_split(self, rock_name):
        keywords = [",", "、", "-", " and ", "和", "及", "或", "\n", "/", "("]
        pattern = "|".join(map(re.escape, keywords))

        rock_name = rock_name.lower().strip()
        names = re.split(pattern, rock_name)
        names = [self.clean_rock_name(n.strip().strip(")")) for n in names]
        names = [n for n in names if len(n) > 0]

        return names

    def get_rock_type_or_age(self, rock_name):
        if rock_name is None:
            return "unknown"

        names = self.rock_split(rock_name)
        find_type = None

        for db_name, db_type in self.rock_name2type.items():
            for name in names:
                if name in db_name or db_name in name:
                    find_type = db_type

        return find_type if find_type is not None else "unknown"


if __name__ == "__main__":
    rock_name = "花岗岩"
    db = rock_type_and_age_db(mode="type")
    rock_type = db.get_rock_type_or_age(rock_name)
    print(rock_type)
