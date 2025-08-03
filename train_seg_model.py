from ultralytics import YOLO
import os

model = YOLO('yolov8n-seg.pt')

print("Starting YOLOv8 Segmentation model training...")
results = model.train(data='Faucet2_dataset_v3/data.yaml', # <--- 指向你的新 data.yaml
                      epochs=30,
                      imgsz=640,
                      batch=2,
                      device='cpu',
                      project='runs/segment', # <--- 將結果儲存到 runs/segment/ 下
                      name='train_seg') # <--- 這次訓練的名稱，例如 'train_seg'
print("YOLOv8 Segmentation model training complete!")

# Step 3: 評估模型 (可選，但強烈建議)
print("\nEvaluating the trained segmentation model...")
# 直接使用訓練完成後的 model 物件來進行評估
eval_results = model.val()
print("\nModel evaluation complete!")
print(eval_results)


# ------------------------------------------------------
# from ultralytics import YOLO
# import os
#
# model = YOLO('yolov8n-seg.pt')
#
# print("Starting YOLOv8 Segmentation model training...")
# results = model.train(data='dataset_coco/Faucet1_dataset/data.yaml', # <--- 指向你的新 data.yaml
#                       epochs=100,
#                       imgsz=640,
#                       batch=16,
#                       device='mps',
#                       project='runs/segment', # <--- 將結果儲存到 runs/segment/ 下
#                       name='train_seg') # <--- 這次訓練的名稱，例如 'train_seg'
# print("YOLOv8 Segmentation model training complete!")
#
# # Step 3: 評估模型 (可選，但強烈建議)
# print("\nEvaluating the trained segmentation model...")
# # 直接使用訓練完成後的 model 物件來進行評估
# eval_results = model.val()
# print("\nModel evaluation complete!")
# print(eval_results)