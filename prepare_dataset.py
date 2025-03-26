import os
import argparse
import shutil
import yaml
import glob
import random
from tqdm import tqdm

def parse_args():
    parser = argparse.ArgumentParser(description='Prepare Dataset for Football Detector Training')
    parser.add_argument('--roboflow-dir', type=str, help='Path to Roboflow dataset directory')
    parser.add_argument('--kaggle-dir', type=str, help='Path to Kaggle dataset directory')
    parser.add_argument('--output-dir', type=str, required=True, help='Output directory for prepared dataset')
    parser.add_argument('--split', type=float, default=0.8, help='Train/val split ratio')
    return parser.parse_args()

def prepare_roboflow_dataset(roboflow_dir, output_dir, train_split=0.8):
    """
    Prepare Roboflow football dataset for training
    
    Args:
        roboflow_dir: Path to Roboflow dataset (downloaded from Roboflow)
        output_dir: Output directory to save prepared dataset
        train_split: Train/validation split ratio
    """
    print("Preparing Roboflow dataset...")
    
    # Create output directories
    os.makedirs(os.path.join(output_dir, 'images', 'train'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'images', 'val'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'labels', 'train'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'labels', 'val'), exist_ok=True)
    
    # Get list of images and labels
    images = sorted(glob.glob(os.path.join(roboflow_dir, 'train', 'images', '*.jpg')))
    labels = sorted(glob.glob(os.path.join(roboflow_dir, 'train', 'labels', '*.txt')))
    
    # Shuffle and split
    combined = list(zip(images, labels))
    random.shuffle(combined)
    split_idx = int(len(combined) * train_split)
    train_data = combined[:split_idx]
    val_data = combined[split_idx:]
    
    # Copy files to respective directories
    for i, (img_path, label_path) in enumerate(tqdm(train_data, desc="Processing train data")):
        img_name = os.path.basename(img_path)
        label_name = os.path.basename(label_path)
        
        shutil.copy(img_path, os.path.join(output_dir, 'images', 'train', img_name))
        shutil.copy(label_path, os.path.join(output_dir, 'labels', 'train', label_name))
    
    for i, (img_path, label_path) in enumerate(tqdm(val_data, desc="Processing validation data")):
        img_name = os.path.basename(img_path)
        label_name = os.path.basename(label_path)
        
        shutil.copy(img_path, os.path.join(output_dir, 'images', 'val', img_name))
        shutil.copy(label_path, os.path.join(output_dir, 'labels', 'val', label_name))
    
    # Create data.yaml
    data_yaml = {
        'path': os.path.abspath(output_dir),
        'train': 'images/train',
        'val': 'images/val',
        'nc': 3,  # Update with actual number of classes
        'names': ['player', 'ball', 'referee']  # Update with actual class names
    }
    
    with open(os.path.join(output_dir, 'data.yaml'), 'w') as f:
        yaml.dump(data_yaml, f, default_flow_style=False)
    
    print(f"Roboflow dataset prepared. Train images: {len(train_data)}, Val images: {len(val_data)}")

def prepare_kaggle_dataset(kaggle_dir, output_dir, train_split=0.8):
    """
    Prepare Kaggle football dataset for training
    
    Args:
        kaggle_dir: Path to Kaggle dataset
        output_dir: Output directory to save prepared dataset
        train_split: Train/validation split ratio
    """
    print("Preparing Kaggle dataset...")
    
    # Create output directories
    os.makedirs(os.path.join(output_dir, 'images', 'train'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'images', 'val'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'labels', 'train'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'labels', 'val'), exist_ok=True)
    
    # Get list of images and labels (adjust paths according to Kaggle dataset structure)
    images = sorted(glob.glob(os.path.join(kaggle_dir, '**', '*.jpg'), recursive=True))
    
    # Shuffle and split images
    random.shuffle(images)
    split_idx = int(len(images) * train_split)
    train_images = images[:split_idx]
    val_images = images[split_idx:]
    
    # Copy images to respective directories
    for img_path in tqdm(train_images, desc="Processing train images"):
        img_name = os.path.basename(img_path)
        shutil.copy(img_path, os.path.join(output_dir, 'images', 'train', img_name))
    
    for img_path in tqdm(val_images, desc="Processing validation images"):
        img_name = os.path.basename(img_path)
        shutil.copy(img_path, os.path.join(output_dir, 'images', 'val', img_name))
    
    print(f"Kaggle dataset images copied. You'll need to label these using a tool like labelImg or CVAT.")
    print(f"Train images: {len(train_images)}, Val images: {len(val_images)}")
    
def main():
    args = parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    if args.roboflow_dir:
        prepare_roboflow_dataset(args.roboflow_dir, args.output_dir, args.split)
    
    if args.kaggle_dir:
        prepare_kaggle_dataset(args.kaggle_dir, args.output_dir, args.split)
    
    print(f"Dataset preparation completed. Output saved to {args.output_dir}")

if __name__ == "__main__":
    main() 