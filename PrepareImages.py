import os
import json
import base64
import requests
from PIL import Image
from PIL import ImageFilter
from io import BytesIO
import numpy as np
from sklearn.neighbors import KDTree
from sklearn.cluster import KMeans

# ===== CONFIGURATION =====
API_URL = "http://127.0.0.1:7860/sdapi/v1/img2img"
SOURCE_FOLDER = "ImgIdeas"
RECOLORED_FOLDER = "RecoloredTemp"
OUTPUT_FOLDER = "Images"
LORA_BICHU = True

USE_DIRECT_PALETTE_MAPPING = False  # Set to True to skip KMeans and map every pixel directly to palette (has slight dithering effect)
BLUR_RECOLORED_IMAGE = True  # Set to True to apply blur before img2img
BLUR_RADIUS = 1.3            # Adjust for more or less smoothing


FINAL_WIDTH = 1024
FINAL_HEIGHT = 1024
FINAL_STEPS = 32
FINAL_CFG_SCALE = 7.0
FINAL_DENOISING = 0.6

COLOR_SCHEMES = {
    "CrimsonTwilight": ['#C56D70', '#241B28', '#566B7B', '#4DB5BF', '#76353F', '#324557', '#EFDBC2', '#4B92A5', '#BFB5BF', '#5C6464'],
    "EnchantedGrove":  ['#3B5A3A', '#CFD8B7', '#97BC90', '#192111', '#879F6D', '#659B7E', '#69915A', '#B7BF4D', '#89AAA5', '#B4741C']
}

NUM_REDUCED_COLORS = 10
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

    tree = KDTree(palette_rgb)

    if USE_DIRECT_PALETTE_MAPPING:
        _, indices = tree.query(pixels, k=1)
        mapped = np.array([palette_rgb[i[0]] for i in indices])
        recolored_rgb = mapped.reshape(rgb.shape)
    else:
        kmeans = KMeans(n_clusters=NUM_REDUCED_COLORS, random_state=42, n_init='auto')
        labels = kmeans.fit_predict(pixels)
        reduced_pixels = kmeans.cluster_centers_.astype(np.uint8)[labels]
        reduced_rgb = reduced_pixels.reshape(rgb.shape)

        _, indices = tree.query(reduced_rgb.reshape(-1, 3), k=1)
        mapped = np.array([palette_rgb[i[0]] for i in indices])
        recolored_rgb = mapped.reshape(rgb.shape)

    output = np.dstack((recolored_rgb, alpha)).astype(np.uint8)
    Image.fromarray(output, mode="RGBA").save(output_path)

def blur_image_in_place(image_path, radius=1.2):
    img = Image.open(image_path).convert("RGBA")
    blurred = img.filter(ImageFilter.GaussianBlur(radius=radius))
    blurred.save(image_path)

def image_to_base64(image_path):
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode('utf-8')


def find_json_for_image(image_path):
    base_name = os.path.splitext(image_path)[0]
    json_path = base_name + ".json"
    return json_path if os.path.exists(json_path) else None


def find_prompt_part(start_dir, filename):
    """Look upward for a txt file (prefix.txt or suffix.txt). Return its contents if found."""
    current_dir = start_dir
    while True:
        path = os.path.join(current_dir, filename)
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                return f.read().strip()
        if current_dir == SOURCE_FOLDER or os.path.dirname(current_dir) == current_dir:
            break
        current_dir = os.path.dirname(current_dir)
    return ""

def clean_prompt(*parts):
    joined = ", ".join(part.strip().strip(',') for part in parts if part.strip())
    return ", ".join([p.strip() for p in joined.split(",") if p.strip()])

def build_prompt(json_path):
    base_dir = os.path.dirname(json_path)
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if "prompt-img2img" in data:
        return data["prompt-img2img"]

    prefix = find_prompt_part(base_dir, "prefix.txt")
    suffix = find_prompt_part(base_dir, "suffix.txt")
    core = data.get("prompt", "")

    return clean_prompt(prefix, core, suffix)



def generate_img2img(image_b64, prompt, params):
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


def generate_img2img_bichu(image_b64, prompt, params):
    bichu_prompt = "<lora:bichu-v0612>, " + prompt
    payload = {
        "init_images": [image_b64],
        "prompt": bichu_prompt,
        "negative_prompt": params.get("negative_prompt", ""),
        "width": FINAL_WIDTH,
        "height": FINAL_HEIGHT,
        "steps": 42,
        "seed": params.get("seed", -1),
        "sampler_name": params.get("sampler_name", "Euler a"),
        "scheduler": params.get("scheduler", "Karras"),
        "cfg_scale": FINAL_CFG_SCALE,
        "denoising_strength": 0.65,
        "resize_mode": 1,
        "override_settings": {
            "sd_model_checkpoint": "dreamshaper_8.safetensors [879db523c3]"
        }
    }
    response = requests.post(API_URL, json=payload)
    if response.status_code != 200:
        print(f"Failed Bichu generation: {response.status_code} - {response.text}")
        return None
    return response.json().get("images", [None])[0]


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
                    output_path = os.path.join(OUTPUT_FOLDER, scheme_name, relative_path)
                    bichu_output_path = output_path.replace(".png", "_bichu.png")

                    palette_rgb = [hex_to_rgb(c) for c in colors]
                    recolored_path = os.path.join(RECOLORED_FOLDER, scheme_name, relative_path)
                    os.makedirs(os.path.dirname(recolored_path), exist_ok=True)

                    # Generate main image if missing
                    if not os.path.exists(output_path):
                        recolor_image(image_path, recolored_path, palette_rgb)
                        if BLUR_RECOLORED_IMAGE:
                            blur_image_in_place(recolored_path, BLUR_RADIUS)
                        image_b64 = image_to_base64(recolored_path)
                        prompt = build_prompt(json_path)

                        print(f"\nGenerating for {relative_path} with scheme {scheme_name}...")
                        image_result_b64 = generate_img2img(image_b64, prompt, params)
                        if image_result_b64:
                            save_base64_image(image_result_b64, output_path)
                            print(f"Saved to {output_path}")
                        else:
                            print(f"Failed to generate image for {relative_path} with scheme {scheme_name}")
                            continue  # skip Bichu if main image generation failed

                    # Generate bichu image if requested and missing
                    if LORA_BICHU and not os.path.exists(bichu_output_path):
                        print(f"Generating Bichu variant for {relative_path}...")
                        base64_for_bichu = image_to_base64(output_path)
                        prompt = build_prompt(json_path)
                        bichu_result_b64 = generate_img2img_bichu(base64_for_bichu, prompt, params)
                        if bichu_result_b64:
                            save_base64_image(bichu_result_b64, bichu_output_path)
                            print(f"Saved Bichu image to {bichu_output_path}")
                        else:
                            print(f"Failed Bichu generation for {relative_path}")



if __name__ == "__main__":
    process_images()
