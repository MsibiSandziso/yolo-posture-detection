import os
import json
import cv2
from tqdm import tqdm

# IMPORTANT: Adjust this path to where you placed your downloaded COCO files
DATA_DIR = "/Users/msibisandzisothando/Desktop/coco_dataset" 

# Input file paths
ANNOTATIONS_FILE = os.path.join(DATA_DIR, "annotations", "person_keypoints_train2017.json")
OUTPUT_DIR = os.path.join(DATA_DIR, "yolo_keypoints_format")

# COCO Keypoint Metadata
# COCO uses 17 keypoints. YOLOv8 Keypoint class ID is typically 0 (person).
KEYPOINT_CLASSES = 1 # Only one class: person
KEYPOINT_LABELS = [
    'nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear', 
    'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow', 
    'left_wrist', 'right_wrist', 'left_hip', 'right_hip', 
    'left_knee', 'right_knee', 'left_ankle', 'right_ankle'
]
NUM_KEYPOINTS = len(KEYPOINT_LABELS)

def convert_coco_to_yolo(split_name, annotations_path, images_dir):
    """Converts COCO keypoint annotations to YOLO keypoint format."""
    print(f"Starting conversion for {split_name} split...")

    # Create output directories
    output_labels_dir = os.path.join(OUTPUT_DIR, split_name, "labels")
    os.makedirs(output_labels_dir, exist_ok=True)
    
    # Load COCO annotations
    with open(annotations_path, 'r') as f:
        coco_data = json.load(f)

    # Map image IDs to file names
    image_map = {img['id']: img for img in coco_data['images']}
    
    # Group annotations by image ID
    annotations_map = {}
    for anno in coco_data['annotations']:
        if anno['category_id'] == 1 and anno['iscrowd'] == 0 and sum(anno['keypoints']) > 0: # Ensure valid person annotation with keypoints
            img_id = anno['image_id']
            if img_id not in annotations_map:
                annotations_map[img_id] = []
            annotations_map[img_id].append(anno)

    # Process each image
    for img_id, annotations in tqdm(annotations_map.items(), desc=f"Processing {split_name} images"):
        img_info = image_map[img_id]
        img_filename = img_info['file_name']
        
        # Load image to get width and height (or use info from JSON, which is safer)
        img_width = img_info['width']
        img_height = img_info['height']
        
        # The output YOLO annotation file path
        label_filename = img_filename.replace(".jpg", ".txt")
        output_path = os.path.join(output_labels_dir, label_filename)
        
        yolo_lines = []

        for anno in annotations:
            # 1. Bounding Box (x_min, y_min, width, height)
            bbox = anno['bbox']
            
            # Normalize Bounding Box: [x_center, y_center, w, h] (normalized 0-1)
            x_center = (bbox[0] + bbox[2] / 2) / img_width
            y_center = (bbox[1] + bbox[3] / 2) / img_height
            w_norm = bbox[2] / img_width
            h_norm = bbox[3] / img_height
            
            # Start the YOLO line with class ID (0 for person) and normalized bbox
            yolo_line = [0, x_center, y_center, w_norm, h_norm]
            
            # 2. Keypoints (x, y, visibility)
            # COCO Keypoints are a flat list: [x1, y1, v1, x2, y2, v2, ...] (17 joints * 3)
            keypoints = anno['keypoints']
            
            for k in range(NUM_KEYPOINTS):
                x_k = keypoints[k * 3]
                y_k = keypoints[k * 3 + 1]
                v_k = keypoints[k * 3 + 2] # 0: not labeled, 1: unlabeled/occluded, 2: visible
                
                # Normalize keypoint coordinates
                x_norm = x_k / img_width
                y_norm = y_k / img_height
                
                # Append normalized coordinates and visibility to the line
                yolo_line.extend([x_norm, y_norm, v_k])
                
            yolo_lines.append(" ".join(map(str, yolo_line)))

        # Write all annotations for this image to the .txt file
        if yolo_lines:
            with open(output_path, 'w') as f:
                f.write("\n".join(yolo_lines))
    
    print(f"✅ Conversion for {split_name} complete! Files saved to: {output_labels_dir}")


if __name__ == "__main__":
    # --- Training Data ---
    train_annotations = os.path.join(DATA_DIR, "annotations", "person_keypoints_train2017.json")
    train_images = os.path.join(DATA_DIR, "train2017")
    convert_coco_to_yolo("train", train_annotations, train_images)

    # --- Validation Data ---
    val_annotations = os.path.join(DATA_DIR, "annotations", "person_keypoints_val2017.json")
    val_images = os.path.join(DATA_DIR, "val2017")
    convert_coco_to_yolo("val", val_annotations, val_images)

    # --- Final Next Steps ---
    print("\n------------------------------------------------------------")
    print("YOLOv8 Keypoints preparation is complete!")
    print(f"The YOLO labels are in: {os.path.join(OUTPUT_DIR, 'train/labels')} and {os.path.join(OUTPUT_DIR, 'val/labels')}")
    
    print("\nNEXT STEP: Create the dataset configuration YAML file.")