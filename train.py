# -*- coding: utf-8 -*-
from ultralytics import YOLO

model = YOLO('C:/Users/ThinkBook/Desktop/yolov11_proj/runs/detect/train3/weights/last.pt')

model.train(
    data='D:/install/ANACONDA/envs/yolov11/my_dataset/data.yaml',  # 改成你的data.yaml实际路径
    epochs=50,
    imgsz=320,
    batch=4,
    workers=2,
    fraction=0.3,
    device='cpu',
    cache=False,
    resume=True, 
)
