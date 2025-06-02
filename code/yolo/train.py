from ultralytics import YOLO
from roboflow import Roboflow

ROOT = "D:/Research/SeniorResearch/"

def main():
    # Load a COCO-pretrained YOLO11n model
    print("loading")
    model = YOLO(ROOT+"yolo/yolo11n.pt")
    # model = YOLO("./runs/detect/train/weights/best.pt")
    
    model.info()

    print("training")
    # Download data from my Roboflow project
    rf = Roboflow(api_key="YmuCOgzvMbkcyzSLemjB")
    project = rf.workspace("rocco-z-cnxls").project("sr-yolo")
    version = project.version(3)
    dataset = version.download("yolov11", location=ROOT+"yolo/datasets")
    
    # Train the model on the COCO8 example dataset for 100 epochs
    results = model.train(
        data=ROOT+"yolo/datasets/SR-YOLO-3/data.yaml", 
        epochs=50, 
        imgsz=640, 
        device='cuda',
        project=ROOT+"yolo/runs")

    # print("inferring")
    
    # Run inference with the YOLO11n model on the 'bus.jpg' image
    # results = model("./yolo/datasets/bus.jpg")
    # print(results)

    #  python "C:\Users\1595624\AppData\Roaming\Python\Python311\site-packages\labelImg\labelImg.py"

if __name__=="__main__":
    main()