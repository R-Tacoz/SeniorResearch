import cv2
import torch
from ultralytics import YOLO

def detect_camera():
    """Automatically detect an available camera."""
    for i in range(5):  # Check first 5 camera indexes
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            cap.release()
            return i
    return None

def main():
    model_path = "D:/Research/SeniorResearch/runs/detect/train/weights/best.pt"  # Change to your trained model's path
    model = YOLO(model_path)  # Load YOLO model
    
    cam_index = detect_camera()
    if cam_index is None:
        print("No camera detected.")
        return
    
    cap = cv2.VideoCapture(cam_index)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        results = model(frame)  # Run detection
        
        for r in results:
            for box in r.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])  # Bounding box
                conf = box.conf[0].item()  # Confidence score
                cls = int(box.cls[0].item())  # Class index
                label = f"{model.names[cls]} {conf:.2f}"  # Label
                
                # Draw bounding box
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        cv2.imshow("YOLOv8 Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()