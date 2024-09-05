import cv2
import torch
import torchvision.transforms as T
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from config import COLOR_DICT, COCO_INSTANCE_CATEGORY_NAMES

# Load the pre-trained model
model = fasterrcnn_resnet50_fpn(weights="FasterRCNN_ResNet50_FPN_Weights.DEFAULT")
model.eval()

transform = T.Compose([T.ToTensor()])

def get_predictions(img, threshold=0.7):
    img_resized = cv2.resize(img, (640, 480))
    img_tensor = transform(img_resized)
    img_tensor = img_tensor.unsqueeze(0)
    
    with torch.no_grad():
        outputs = model(img_tensor)
    
    pred_classes = [COCO_INSTANCE_CATEGORY_NAMES[i] for i in outputs[0]['labels'].numpy()]
    pred_scores = outputs[0]['scores'].detach().numpy()
    pred_boxes = outputs[0]['boxes'].detach().numpy()
    pred_boxes = pred_boxes[pred_scores >= threshold].astype(int)
    pred_classes = [pred_classes[i] for i in range(len(pred_scores)) if pred_scores[i] >= threshold]
    pred_scores = pred_scores[pred_scores >= threshold]

    pred_boxes = (pred_boxes * [img.shape[1] / 640, img.shape[0] / 480, img.shape[1] / 640, img.shape[0] / 480]).astype(int)

    return pred_boxes, pred_classes, pred_scores

def live_object_detection():
    cap = cv2.VideoCapture(0)
    
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 2024)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1000)
    cap.set(cv2.CAP_PROP_FPS, 8)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        boxes, classes, scores = get_predictions(frame, threshold=0.7)

        for box, label, score in zip(boxes, classes, scores):
            color = COLOR_DICT.get(label, COLOR_DICT['__default__'])
            cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), color, 2)
            cv2.putText(frame, f"{label}: {score:.2f}", (box[0], box[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        cv2.imshow('Live Object Detection', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

def detect_from_image(image_path):
    # Read and process the image
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Could not read the image file {image_path}")
        return
    
    boxes, classes, scores = get_predictions(img, threshold=0.7)
    
    for box, label, score in zip(boxes, classes, scores):
        color = COLOR_DICT.get(label, COLOR_DICT['__default__'])
        cv2.rectangle(img, (box[0], box[1]), (box[2], box[3]), color, 2)
        cv2.putText(img, f"{label}: {score:.2f}", (box[0], box[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    
    cv2.imshow('Image Object Detection', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()