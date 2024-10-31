import cv2
import json
from ultralytics import YOLO

def count_objects_in_image(image_path, model_path, output_json=None):
    # Load the YOLO model
    model = YOLO(model_path)  # Replace with your YOLO model path

    # Read the image
    image = cv2.imread(image_path)

    # Detect objects in the image
    results = model(image)

    # Extract detection results
    detections = results[0].boxes.xyxy.cpu().numpy()  # Bounding box coordinates
    classes = results[0].boxes.cls.cpu().numpy()  # Object classes

    # Filter only humans (class 0) and vehicles (classes 2, 3, 5, 7)
    filtered_boxes = []
    for xyxy, cls in zip(detections, classes):
        if int(cls) in [0, 2, 3, 5, 7]:  # Filter for people and vehicles
            filtered_boxes.append(xyxy.tolist())

    # Count the number of humans and vehicles detected
    object_count = len(filtered_boxes)

    # Create the result dictionary
    image_data = {
        'object_count': object_count,
        'bounding_boxes': filtered_boxes  # Only bounding boxes for people and vehicles
    }

    # Save to JSON if output_json path is provided
    if output_json:
        with open(output_json, 'w') as file:
            json.dump(image_data, file, indent=4)
        print(f"Object counts and bounding boxes saved to {output_json}")

    return image_data


