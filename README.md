# Object-Inpainting-SAM

## Project Description

This project implements a system for detecting, masking, and inpainting objects in images using **Segment Anything Model (SAM)**, **YOLOv9** for object detection, and **Stable Diffusion Inpainting** for realistic background filling. It’s designed for seamless object removal and background restoration across several stages:

1. **Object Detection**: YOLOv9 identifies and localizes objects within an image, generating bounding boxes.
  
2. **Masking with SAM**: **Segment Anything Model (SAM)** generates precise masks for detected objects to define areas for inpainting.

3. **Inpainting with Stable Diffusion**: **Stable Diffusion Inpainting** realistically fills masked regions with background, restoring visual consistency.

4. **Result Visualization**: The final output image, with objects removed and background restored, is saved in the `assets` directory.

## Project Features

- **Automatic Object Detection and Masking**: YOLOv9 and SAM integration allows precise identification and masking of objects for removal.
- **High-Quality Inpainting**: Stable Diffusion Inpainting fills the background smoothly and realistically.
- **Adaptable Workflow**: Easily adjust detection classes and inpainting parameters for different effects.

## Example Workflow

### 1. Initial Image with Detected Objects
YOLOv9 detects objects and creates bounding boxes:

![Initial Detection](img/detection_example.png)

### 2. Generated Mask with SAM
SAM produces masks for each detected object:

![Object Masking](img/mask_example.png)

### 3. Background Inpainting
Stable Diffusion fills masked regions with realistic background:

![Inpainting Process](img/inpainting_example.png)

### 4. Final Image
Objects removed and background restored seamlessly:

![Final Image](img/final_result.png)

## Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/yourusername/Object-Inpainting-SAM.git
   cd Object-Inpainting-SAM
