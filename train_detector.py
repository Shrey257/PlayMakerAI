import argparse
from ultralytics import YOLO

def parse_args():
    parser = argparse.ArgumentParser(description='Train Custom Football Detector')
    parser.add_argument('--data', type=str, required=True, help='Path to data.yaml file')
    parser.add_argument('--model', type=str, default='yolov8n.pt', help='Base YOLOv8 model to start training from')
    parser.add_argument('--epochs', type=int, default=50, help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=16, help='Batch size for training')
    parser.add_argument('--img-size', type=int, default=640, help='Image size for training')
    parser.add_argument('--name', type=str, default='football_detector', help='Name for the training run')
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Load a pretrained YOLOv8 model
    model = YOLO(args.model)
    
    # Train the model
    results = model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.img_size,
        batch=args.batch_size,
        name=args.name
    )
    
    # Validate the model
    results = model.val()
    
    print(f"Training completed. Model saved to {model.ckpt_path}")

if __name__ == "__main__":
    main() 