import cv2
import os
import requests
from PIL import Image
import torch
from diffusers import StableDiffusionInpaintPipeline
from sam import SAMDetector
from count_objects import count_objects_in_image
import numpy as np

image_path = './annotated_frames/frame_000001.jpg'
model_path = './yolov9e.pt'
output_json = 'single_image_object_count.json'

def download_file(url, local_path):
    os.makedirs(os.path.dirname(local_path), exist_ok=True)
    if not os.path.exists(local_path):
        print(f"File not found locally. Downloading from {url}...")
        response = requests.get(url)
        if response.status_code == 200:
            with open(local_path, 'wb') as file:
                file.write(response.content)
            print(f"File downloaded and saved to {local_path}")
        else:
            print(f"Failed to download file. Status code: {response.status_code}")
    else:
        print(f"File already exists at {local_path}")

def draw_bounding_boxes(image, bounding_boxes):
    image = cv2.imread(image_path)
    for box in bounding_boxes:
        x1, y1, x2, y2 = map(int, box)
        cv2.rectangle(image, (x1, y1), (x2, y2), (255, 0, 0), 2)
    return image

url = "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth"
local_path = "checkpoints/sam_vit_h_4b8939.pth"
download_file(url, local_path)
os.makedirs("assets", exist_ok=True)

image_data = count_objects_in_image(image_path, model_path, output_json)
bounding_boxes = image_data['bounding_boxes']

image_with_boxes = draw_bounding_boxes(image_path, bounding_boxes)
cv2.imwrite('assets/image_with_boxes.png', image_with_boxes)

detector = SAMDetector(model_type="vit_h", checkpoint_path="checkpoints/sam_vit_h_4b8939.pth")

image = detector.load_image(image_path)
original_image = Image.open(image_path)

for box in bounding_boxes:
    x1, y1, x2, y2 = map(int, box)
    boxToFindMask = [x1, y1, x2, y2]
    best_mask, scores, logits = detector.predict_mask(boxToFindMask)

    mask_image = Image.fromarray(best_mask)
    mask_image.save("./assets/mask.png")

    inpaint_box = [int(box[0]) - 100, int(box[1]) - 100, int(box[2]) + 100, int(box[3]) + 100]
    cropped_image = original_image.crop(inpaint_box)
    cropped_mask = mask_image.crop(inpaint_box)
    resized_image = cropped_image.resize((512, 512), Image.LANCZOS)
    resized_mask = cropped_mask.resize((512, 512), Image.LANCZOS)

    torch.cuda.empty_cache()  # Czyszczenie pamięci GPU przed inpaintingiem

    # Użycie CPU zamiast GPU
    pipe = StableDiffusionInpaintPipeline.from_pretrained(
        "stabilityai/stable-diffusion-2-inpainting",
        torch_dtype=torch.float32,
    )
    pipe.to("cpu")

    prompt = "background"
    negative_prompt = "car, person, man, girl, boy, woman"

    inpainted_resized_image = pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        image=resized_image,
        mask_image=resized_mask
    ).images[0]

    inpainted_image = inpainted_resized_image.resize(cropped_image.size, Image.LANCZOS)
    original_image.paste(inpainted_image, inpaint_box)

original_image.save("./assets/out.png")
