from utils import get_predictions, live_object_detection,detect_from_image
import cv2
import argparse

def main():
    parser = argparse.ArgumentParser(description="Object Detection")
    parser.add_argument('--mode', choices=['live', 'image'], required=True, help="Mode of operation: 'live' for live detection, 'image' for single image detection.")
    parser.add_argument('--image_path', type=str, help="Path to the image file for detection. Required if mode is 'image'.")
    
    args = parser.parse_args()
    
    if args.mode == 'live':
        live_object_detection()
    elif args.mode == 'image':
        if not args.image_path:
            print("Error: --image_path is required when mode is 'image'.")
            return
        detect_from_image(args.image_path)

if __name__ == "__main__":
    main()