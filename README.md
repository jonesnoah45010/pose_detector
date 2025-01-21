# Pose Detection with MediaPipe

This project demonstrates the use of the [MediaPipe](https://mediapipe.dev/) Pose Landmarker API for pose detection and visualization. The code processes an input image to detect human poses, create segmentation masks, and overlay pose landmarks on the original image or a black background.

## Features

- **Pose Detection**: Detects pose landmarks in an image using MediaPipe's Pose Landmarker.
- **Overlay Landmarks on Original Image**: Adds detected pose landmarks on top of the input image.
- **Pose-only Visualization**: Generates an image with pose landmarks drawn on a black background.
- **Segmentation Mask**: Creates a segmentation mask for detected poses.

## Prerequisites

- Python 3.8+
- Required Python libraries:
  - `mediapipe`
  - `opencv-python`
  - `numpy`

## Installation

1. Clone this repository:
   ```bash
   git clone https://github.com/jonesnoah45010/pose_detector.git
   cd pose_detector
   ```

2. Install the required libraries:
   ```bash
   pip install -r requirements.txt
   ```

3. Download the MediaPipe Pose Landmarker model:
   - [pose_landmarker_heavy.task](https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_heavy/float16/1/pose_landmarker_heavy.task)
   - Save this file in the same directory as `main.py`.

## Usage

1. Place an input image (`image.jpg`) in the same directory as `main.py` or use the included sample (`image.jpg`)

2. Run the script:
   ```bash
   python main.py
   ```

3. The script generates the following output images:
   - `image_with_pose.jpg`: Original image with pose landmarks overlaid.
   - `image_pose.jpg`: Pose landmarks on a black background.
   - `image_mask.jpg`: Segmentation mask highlighting the detected pose.

## Functions

### `get_pose(image_file)`
Detects poses in the provided image file.

- **Parameters**: `image_file` (str) – Path to the input image.
- **Returns**: MediaPipe image object and detection results.

### `add_pose_on_top_of_image(image_file, output_file)`
Overlays detected pose landmarks on the original image.

- **Parameters**:
  - `image_file` (str) – Path to the input image.
  - `output_file` (str) – Path to save the output image.
- **Returns**: Annotated image.

### `convert_image_to_pose(image_file, output_file)`
Generates a visualization of pose landmarks on a black background.

- **Parameters**:
  - `image_file` (str) – Path to the input image.
  - `output_file` (str) – Path to save the output image.
- **Returns**: Annotated image.

### `convert_image_to_mask(image_file, output_file)`
Creates a segmentation mask for the detected pose.

- **Parameters**:
  - `image_file` (str) – Path to the input image.
  - `output_file` (str) – Path to save the segmentation mask.
- **Returns**: Visualized segmentation mask.



