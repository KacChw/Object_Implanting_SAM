import cv2
from segment_anything import SamPredictor, sam_model_registry
import numpy as np


class SAMDetector:
    def __init__(self, model_type="vit_h", checkpoint_path="checkpoints/sam_vit_h_4b8939.pth"):
        self.sam = sam_model_registry[model_type](checkpoint=checkpoint_path)
        self.mask_predictor = SamPredictor(self.sam)

    def load_image(self, image_path):
        image_bgr = cv2.imread(image_path)
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        self.mask_predictor.set_image(image_rgb)
        return image_bgr

    def predict_mask(self, box):
        box_array = np.array(box).reshape(1, 4)
        masks, scores, logits = self.mask_predictor.predict(box=box_array, multimask_output=True)
        best_mask = masks[np.argmax(scores)]
        return best_mask, scores, logits

    def apply_mask(self, image_bgr, mask):
        result_image = image_bgr.copy()
        result_image[mask] = [0, 0, 255]
        return result_image

    def save_image(self, image, output_path):
        cv2.imwrite(output_path, image)
