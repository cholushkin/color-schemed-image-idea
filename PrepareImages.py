import os
import json
import base64
import requests
from PIL import Image
from io import BytesIO
import numpy as np
from sklearn.neighbors import KDTree
from sklearn.cluster import KMeans

# ===== CONFIGURATION =====
API_URL = "http://127.0.0.1:7860/sdapi/v1/img2img"
SOURCE_FOLDER = "ImgIdeas"
RECOLORED_FOLDER = "Recolored"
OUTPUT_FOLDER = "Images"

FINAL_WIDTH = 1024
FINAL_HEIGHT = 1024
FINAL_STEPS = 28
FINAL_CFG_SCALE = 7.0
FINAL_DENOISING = 0.65

COLOR_SCHEMES = {
    "CrimsonTwilight": ['#C56D70', '#241B28', '#566B7B', '#4DB5BF', '#76353F', '#324557', '#EFDBC2', '#4B92A5', '#BFB5BF', '#5C6464'],
    "Sunset": ['#FF4500', '#FFA500', '#FFD700', '#FF6347', '#800000', '#DC143C', '#FF69B4', '#FFB6C1', '#FFE4B5', '#FFDAB9'],
    "Ocean": ['#000080', '#0000CD', '#4169E1', '#00BFFF', '#87CEFA', '#4682B4', '#5F9EA0', '#00CED1', '#20B2AA', '#40E0D0']
}


NUM_REDUCED_COLORS = 10  # KMeans cluster count
# ==========================


def hex_to_rgb(hex_color):
    hex_color = hex_color.lstrip('#')
    return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))


def recolor_image(input_path, output_path, palette_rgb):
    img = Image.open(input_path).convert("RGBA")
    data = np.array(img)
    alpha = data[..., 3]
    rgb = data[..., :3]

    pixels = rgb.reshape(-1, 3)

    kmeans = KMeans(n_clusters=NUM_REDUCED_COLORS, random_state=42, n_init='auto')
    labels = kmeans.fit_predict(pixels)
    reduced_pixels = kmeans.cluster_centers_.astype(np.uint8)[labels]
    reduced_rgb = reduced_pixels.reshape(rgb.shape)

    tree = KDTree(palette_rgb)
    _, indices = tree.query(reduced_rgb.reshape(-1, 3), k=1)
    mapped = np.array([palette_rgb[i[0]] for i in indices])
    recolored_rgb = mapped.reshape(rgb.shape)

    output = np.dstack((recolored_rgb, alpha)).astype(np.uint8)
    Image.fromarray(output, mode="RGBA").save(output_path)


def image_to_base64(image_path):
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode('utf-8')


def find_json_for_image(image_path):
    base_name = os.path.splitext(image_path)[0]
    json_path = base_name + ".json"
    return json_path if os.path.exists(json_path) else None


def build_prompt(json_path, scheme_name, colors):
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    base_prompt = data.get("prompt", "")
    palette_prompt = "color scheme: " + ", ".join(colors)
    return f"{base_prompt}, {palette_prompt}"


def generate_img2img(image_b64, prompt, params):
    # Reuse some parameters from idea json
    payload = {
        "init_images": [image_b64],
        "prompt": prompt,
        "negative_prompt": params.get("negative_prompt", ""),
        "width": FINAL_WIDTH,
        "height": FINAL_HEIGHT,
        "steps": FINAL_STEPS, 
        "seed": params.get("seed", -1),
        "sampler_name": params.get("sampler_name", "Euler a"),
        "scheduler": params.get("scheduler", "Karras"),
        "cfg_scale": FINAL_CFG_SCALE,
        "denoising_strength": FINAL_DENOISING,
        "resize_mode": 1
    }
    response = requests.post(API_URL, json=payload)
    if response.status_code != 200:
        print(f"Failed generation: {response.status_code} - {response.text}")
        return None
    return response.json().get("images", [None])[0]


def save_base64_image(b64_str, output_path):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    image_data = base64.b64decode(b64_str)
    image = Image.open(BytesIO(image_data))
    image.save(output_path)


def process_images():
    for root, dirs, files in os.walk(SOURCE_FOLDER):
        if "ignore" in files:
            print(f"Skipping folder (ignored): {root}")
            dirs.clear()
            continue

        for file in files:
            if file.lower().endswith(".png"):
                image_path = os.path.join(root, file)
                json_path = find_json_for_image(image_path)
                if not json_path:
                    print(f"JSON not found for: {image_path}, skipping.")
                    continue

                with open(json_path, "r", encoding="utf-8") as f:
                    params = json.load(f)

                relative_path = os.path.relpath(image_path, SOURCE_FOLDER)

                for scheme_name, colors in COLOR_SCHEMES.items():
                    palette_rgb = [hex_to_rgb(c) for c in colors]

                    # Prepare paths
                    recolored_path = os.path.join(RECOLORED_FOLDER, scheme_name, relative_path)
                    os.makedirs(os.path.dirname(recolored_path), exist_ok=True)

                    recolor_image(image_path, recolored_path, palette_rgb)

                    image_b64 = image_to_base64(recolored_path)
                    prompt = build_prompt(json_path, scheme_name, colors)
                    print(f"\nGenerating for {relative_path} with scheme {scheme_name}...")

                    image_result_b64 = generate_img2img(image_b64, prompt, params)

                    if image_result_b64:
                        output_path = os.path.join(OUTPUT_FOLDER, scheme_name, relative_path)
                        save_base64_image(image_result_b64, output_path)
                        print(f"Saved to {output_path}")
                    else:
                        print(f"Failed to generate image for {relative_path} with scheme {scheme_name}.")



if __name__ == "__main__":
    process_images()
