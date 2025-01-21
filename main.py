from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np
import cv2


# download this file into the same directory as main.py before you run the code below.
# https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_heavy/float16/1/pose_landmarker_heavy.task



base_options = python.BaseOptions(model_asset_path='pose_landmarker_heavy.task')
options = vision.PoseLandmarkerOptions(
    base_options=base_options,
    output_segmentation_masks=True)
detector = vision.PoseLandmarker.create_from_options(options)






def draw_landmarks_on_image(rgb_image, detection_result):
    # Convert the RGB image to BGR format
    bgr_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)
    
    pose_landmarks_list = detection_result.pose_landmarks
    annotated_image = np.copy(bgr_image)

    # Loop through the detected poses to visualize.
    for idx in range(len(pose_landmarks_list)):
        pose_landmarks = pose_landmarks_list[idx]

        # Draw the pose landmarks.
        pose_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
        pose_landmarks_proto.landmark.extend([
            landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in pose_landmarks
        ])
        solutions.drawing_utils.draw_landmarks(
            annotated_image,
            pose_landmarks_proto,
            solutions.pose.POSE_CONNECTIONS,
            solutions.drawing_styles.get_default_pose_landmarks_style()
        )
    return annotated_image






def draw_landmarks(rgb_image, detection_result):
    # Create a black background with the same size as the input image
    height, width, _ = rgb_image.shape
    black_background = np.zeros((height, width, 3), dtype=np.uint8)

    pose_landmarks_list = detection_result.pose_landmarks

    # Loop through the detected poses to visualize.
    for idx in range(len(pose_landmarks_list)):
        pose_landmarks = pose_landmarks_list[idx]

        # Draw the pose landmarks.
        pose_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
        pose_landmarks_proto.landmark.extend([
            landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in pose_landmarks
        ])
        solutions.drawing_utils.draw_landmarks(
            black_background,
            pose_landmarks_proto,
            solutions.pose.POSE_CONNECTIONS,
            solutions.drawing_styles.get_default_pose_landmarks_style()
        )
    return black_background








def get_pose(image_file="image.jpg"):
    image = mp.Image.create_from_file(image_file)
    detection_result = detector.detect(image)
    return image, detection_result








def add_pose_on_top_of_image(image_file="image.jpg", output_file="image_with_pose.jpg"):
    original_image = cv2.imread(image_file)  # This loads the image as BGR
    original_image_rgb = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)  # Convert to RGB for processing
    image, detection_result = get_pose(image_file)
    annotated_image = draw_landmarks_on_image(image.numpy_view(), detection_result)
    cv2.imwrite(output_file, annotated_image)
    return annotated_image








def convert_image_to_pose(image_file="image.jpg", output_file="image_pose.jpg"):
    original_image = cv2.imread(image_file)  # This loads the image as BGR
    original_image_rgb = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)  # Convert to RGB for processing
    image, detection_result = get_pose(image_file)
    annotated_image = draw_landmarks(image.numpy_view(), detection_result)
    cv2.imwrite(output_file, annotated_image)
    return annotated_image








def convert_image_to_mask(image_file="image.jpg", output_file="image_mask.jpg"):
    image, detection_result = get_pose(image_file)
    annotated_image = draw_landmarks_on_image(image.numpy_view(), detection_result)
    segmentation_mask = detection_result.segmentation_masks[0].numpy_view()
    visualized_mask = np.repeat(segmentation_mask[:, :, np.newaxis], 3, axis=2) * 255
    cv2.imwrite(output_file, visualized_mask.astype(np.uint8))
    return visualized_mask








if __name__ == "__main__":

    mask = convert_image_to_mask("image.jpg","image_mask.jpg")
    pose = convert_image_to_pose("image.jpg","image_pose.jpg")
    image_with_pose = add_pose_on_top_of_image("image.jpg","image_with_pose.jpg")






