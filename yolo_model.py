#import
from ultralytics import YOLO

#model save "runs/detect/train/weights/last.pt"
#if best, model save "runs/detect/train/weights/best.pt"
def train_yolo():
    # Load a model
    # load a pretrained model (recommended for training)
    model = YOLO("yolov8n.pt")
    #train
    model.train(data="coco128.yaml", epochs=3)
    #evaluate
    metrics = model.val()
    print(metrics)

#load YOLO model
def yolo_model():
    return YOLO("runs/detect/train2/weights/best.pt")