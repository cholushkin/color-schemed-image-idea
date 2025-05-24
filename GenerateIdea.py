import os
import json
import base64
import requests
from PIL import Image
from io import BytesIO

# === CONFIGURATION ===
api_url = "http://127.0.0.1:7860/sdapi/v1/txt2img"
target_directory = "ImgIdeas"

# === FUNCTIONS ===
def save_base64_image(b64_str, output_path):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    image_data = base64.b64decode(b64_str)
    image = Image.open(BytesIO(image_data))
    image.save(output_path)

def generate_images(payload):
    response = requests.post(api_url, json=payload)
    if response.status_code != 200:
        print(f"Error: {response.status_code} - {response.text}")
        return []
    result = response.json()
    return result.get("images", [])

def process_json_file(json_path, output_image_path):
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if os.path.exists(output_image_path):
        print(f"Image already exists: {output_image_path}, skipping.")
        return

    payload = {
        "prompt": data.get("prompt", ""),
        "negative_prompt": data.get("negative_prompt", ""),
        "width": data.get("width", 768),
        "height": data.get("height", 768),
        "steps": data.get("steps", 26),
        "seed": data.get("seed", -1),
        "sampler_name": data.get("sampler_name", "Euler a"),
        "scheduler": data.get("scheduler", "Karras"),
        "cfg_scale": data.get("cfg_scale", 7.0)
    }

    print(f"Generating image for: {json_path}")
    images_b64 = generate_images(payload)

    if images_b64:
        save_base64_image(images_b64[0], output_image_path)
        print(f"Saved image to {output_image_path}")
    else:
        print(f"No image returned for {json_path}")

def main():
    if not os.path.exists(target_directory):
        print(f"Target directory '{target_directory}' does not exist.")
        return

    for root, _, files in os.walk(target_directory):
        for file in files:
            if file.endswith(".json"):
                json_path = os.path.join(root, file)
                output_image_name = os.path.splitext(file)[0] + ".png"
                output_image_path = os.path.join(root, output_image_name)
                process_json_file(json_path, output_image_path)

if __name__ == "__main__":
    main()
