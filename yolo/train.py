from ultralytics import YOLO

if __name__=="__main__":
    # Load a COCO-pretrained YOLO11n model
    print("loading")
    model = YOLO("D:/Research/SeniorResearch/yolo/yolo11s.pt")
    # model = YOLO("./runs/detect/train/weights/best.pt")
    
    model.info()

    print("training")
    # Train the model on the COCO8 example dataset for 100 epochs
    results = model.train(data="D:/Research/SeniorResearch/yolo/datasets/panda/data.yaml", epochs=50, imgsz=640, device='cuda')

    # print("inferring")
    
    
    
    # Run inference with the YOLO11n model on the 'bus.jpg' image
    # results = model("./yolo/datasets/bus.jpg")
    # print(results)

    #  python "C:\Users\1595624\AppData\Roaming\Python\Python311\site-packages\labelImg\labelImg.py"
