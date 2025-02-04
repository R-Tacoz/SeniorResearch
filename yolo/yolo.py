from ultralytics import YOLO

# Load a COCO-pretrained YOLO11n model
print("loading")
# model = YOLO("yolo11s.pt")
model = YOLO("./runs/detect/train/weights/best.pt")

print("training")
# Train the model on the COCO8 example dataset for 100 epochs
# results = model.train(data="coco8.yaml", epochs=100, imgsz=640)

print("inferring")
# Run inference with the YOLO11n model on the 'bus.jpg' image
results = model("./yolo/datasets/bus.jpg")
print(results)

#  python "C:\Users\1595624\AppData\Roaming\Python\Python311\site-packages\labelImg\labelImg.py"
