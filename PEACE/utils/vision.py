import cv2
import numpy as np
# from paddleocr import PaddleOCR
import matplotlib.pyplot as plt

def image_size(image):
    if isinstance(image, str):
        image = cv2.imread(image)
    h, w, _ = image.shape
    return h, w

def crop_image(image, bndbox):
    x0, y0, x1, y1 = bndbox
    if isinstance(image, str):
        image = cv2.imread(image)
    cropped_image = image[y0:y1, x0:x1]
    return cropped_image

def crop_and_save_image(image, bndbox, cropped_image_path):
    cropped_image = crop_image(image, bndbox)
    cv2.imwrite(cropped_image_path, cropped_image)

def crop_corners_and_save_image(image, cropped_image_path, relative_size=0.1):
    if isinstance(image, str):
        image = cv2.imread(image)

    # Define the relative size of the corner to be cropped (e.g., 10% of height and width)
    height, width, _ = image.shape
    corner_height = int(height * relative_size)
    corner_width = int(width * relative_size)

    # Crop the top-left corner
    top_left = image[0:corner_height, 0:corner_width]
    # Crop the top-right corner
    top_right = image[0:corner_height, width-corner_width:width]
    # Crop the bottom-left corner
    bottom_left = image[height-corner_height:height, 0:corner_width]
    # Crop the bottom-right corner
    bottom_right = image[height-corner_height:height, width-corner_width:width]

    # Combine the corners into a single image
    # First, combine the top corners horizontally
    top_combined = np.hstack((top_left, top_right))
    # Then, combine the bottom corners horizontally
    bottom_combined = np.hstack((bottom_left, bottom_right))
    # Finally, stack the top and bottom combined images vertically
    cropped_image = np.vstack((top_combined, bottom_combined))
    
    cv2.imwrite(cropped_image_path, cropped_image)

def calc_image_rgb(image):
    # Return color in RBG order.
    pixel_list = np.array(list(filter(lambda c: not (c[0] < 16 and c[1] < 16 and c[2] < 16), list(image.reshape(-1, 3)))), dtype=image.dtype)
    #color, _ = Counter(map(tuple, pixel_list)).most_common(1)[0]
    #color = np.mean(pixel_list, axis=0)
    color = np.median(pixel_list, axis=0)
    return color[::-1]

"""
ocr = PaddleOCR(use_angle_cls=True, lang="ch")
#ocr = PaddleOCR(use_angle_cls=True, lang="en")
def image_ocr(img, need_det=False):
    result = ocr.ocr(img, det=need_det, rec=True, cls=True)
    texts = list()
    for idx in range(len(result)):
        res = result[idx]
        try:
            texts.append(res[0])
        except:
            texts.append('Unrecognized')
    return texts[0][0]
"""

def annotate_image_with_directions(image, output_path, font_size=24, offset=50):
    if isinstance(image, str):
        image = cv2.imread(image)

    img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(img_rgb)

    height, width, _ = img_rgb.shape

    ax.set_xlim([-width * 0.2, width * 1.2])
    ax.set_ylim([height * 1.2, -height * 0.2])

    directions = {
        "N": (width // 2, -offset),
        "S": (width // 2, height + offset),
        "W": (-offset, height // 2),
        "E": (width + offset, height // 2),
        "NE": (width + offset, -offset),
        "NW": (-offset, -offset),
        "SE": (width + offset, height + offset),
        "SW": (-offset, height + offset)
    }

    for direction, (x, y) in directions.items():
        ax.text(x, y, direction, fontsize=font_size, fontweight="bold", ha="center", va="center", color="black")

    ax.axis("off")
    plt.savefig(output_path, bbox_inches="tight", pad_inches=0)
    # plt.show()

def rgb_to_color_name(rgb):
    color_map = {
        (255, 0, 0): "Red",
        (0, 255, 0): "Green",
        (0, 0, 255): "Blue",
        (255, 255, 0): "Yellow",
        (0, 255, 255): "Cyan",
        (255, 0, 255): "Magenta",
        (255, 255, 255): "White",
        (0, 0, 0): "Black",
        (128, 128, 128): "Gray",
        (128, 0, 0): "Maroon",
        (128, 128, 0): "Olive",
        (0, 128, 0): "Dark Green",
        (128, 0, 128): "Purple",
        (0, 128, 128): "Teal",
        (0, 0, 128): "Navy",
        (255, 192, 203): "Pink",
        (255, 165, 0): "Orange",
        (0, 255, 127): "Spring Green",
        (255, 105, 180): "Hot Pink",
        (255, 69, 0): "Red-Orange",
        (102, 205, 170): "Medium Aquamarine",
        (173, 216, 230): "Light Blue",
        (240, 230, 140): "Khaki",
        (255, 20, 147): "Deep Pink",
        (255, 99, 71): "Tomato"
    }
 
    min_distance = float('inf')
    closest_color_name = "Unknown"
    for key, value in color_map.items():
        distance = sum((a - b) ** 2 for a, b in zip(rgb, key))
        if distance < min_distance:
            min_distance = distance
            closest_color_name = value

    return closest_color_name

def fault_line_det(image):
    length_threshold = 10

    if isinstance(image, str):
        image = cv2.imread(image)

    color_ranges = ((np.array([0, 150, 150]), np.array([15, 255, 255])), (np.array([165, 150, 150]), np.array([180, 255, 255])))
    
    height, width = image.shape[:2]
    total_pixels = height * width
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    
    color_mask = None
    for i, (lower_color, upper_color) in enumerate(color_ranges):
        if i == 0:
            color_mask = cv2.inRange(hsv, lower_color, upper_color)
        else:
            color_mask = color_mask + cv2.inRange(hsv, lower_color, upper_color)
    
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray_image, 100, 150)
    combined_mask = cv2.bitwise_and(color_mask, edges)
    
    kernel = np.ones((9, 9), np.uint8)
    combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel)
    
    contours, _ = cv2.findContours(combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filtered_contours = [cnt for cnt in contours if cv2.arcLength(cnt, True) >= length_threshold]
    
    total_length = sum(cv2.arcLength(cnt, True) for cnt in filtered_contours)
    total_contour_area = sum(cv2.contourArea(cnt) for cnt in filtered_contours)
    contour_ratio = total_contour_area / total_pixels
    
    return filtered_contours, total_length, contour_ratio

def rgb_to_hex(rgb_color):
    return "#{:02X}{:02X}{:02X}".format(rgb_color[0], rgb_color[1], rgb_color[2])

def hex_to_rgb(hex_color):
    hex_color = hex_color.lstrip("#")
    return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))

def color_key(color):
    return f"{color[0]}_{color[1]}_{color[2]}"

def cal_color_thred(colors):
    color2thred = dict()
    for color1 in colors:
        color2thred[color_key(color1)] = 10
        min_d = 256 *3
        for color2 in colors:
            if color_key(color1) == color_key(color2):
                continue
            d = abs(color1[0] - color2[0]) + abs(color1[1] - color2[1]) + abs(color1[2] - color2[2])
            if d < min_d:
                min_d = d
                color2thred[color_key(color1)] = min_d / 2
    return color2thred

def rock_region_seg(image, legends):
    if isinstance(image, str):
        image = cv2.imread(image)

    _s = 4
    height, width, _ = image.shape
    new_dimensions = (width//_s, height//_s)
    resized_image = cv2.resize(image, new_dimensions, interpolation=cv2.INTER_NEAREST)
    resized_image_rgb = cv2.cvtColor(resized_image, cv2.COLOR_BGR2RGB)
    image_area = 1.0 * resized_image_rgb.shape[0] * resized_image_rgb.shape[1]
    
    colors = np.array([legend["color"] for legend in legends])
    color2thred = cal_color_thred(colors)
    for legend in legends:
        color = legend["color"]
        legend["color_hex"] = rgb_to_hex(color)
        if color != [255, 255, 255]:
            thred = color2thred[color_key(color)]
            distances = np.sum(np.abs(resized_image_rgb - np.array(color)), axis=-1)
            legend["area"] = round(int(np.sum(distances <= thred)) / image_area, 6)
        else:
            legend["area"] = 0
