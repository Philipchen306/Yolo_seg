# demo_video.py

from ultralytics import YOLO

# --- 設定你的路徑 (請務必根據你的實際檔案位置進行修改) ---
TRAINED_MODEL_PATH = 'runs/faucet2_50epoch/train_seg11/weights/best.pt' # 你的 best.pt 路徑
VIDEO_PATH = 'demo_videos/faucet_2.1_out.MOV' # 你的影片檔案路徑

# --- 載入模型 ---
print(f"Loading trained segmentation model from: {TRAINED_MODEL_PATH}")
model = YOLO(TRAINED_MODEL_PATH)

# --- 影片預測設定 ---
# conf: 調整置信度閾值 (例如 0.5 到 0.7)，控制顯示的預測數量，讓 Demo 看起來更乾淨
# iou: 調整 IoU 閾值 (例如 0.6 或 0.7)，控制重疊框的數量
# device: 在 Windows 上，請務必設為 'cuda'
# save: 必須是 True，才能保存輸出影片
print(f"Starting prediction on video: {VIDEO_PATH}")
results = model.predict(source=VIDEO_PATH,
                        conf=0.6,
                        iou=0.7,
                        show=False,
                        save=True,
                        device='cpu' # <--- 在 Windows 上使用 'cuda'
                       )

print("\nVideo prediction complete!")
print(f"Output video saved to: {results[0].save_dir}")