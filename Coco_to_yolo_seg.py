import json
from pathlib import Path
from PIL import Image
import os

# coco_dataset_root = Path('dataset_coco/Faucet1_dataset/')
coco_dataset_root = Path('Faucet2_dataset_v3/')


CLASS_NAMES = [
    "null", "10", "10o", "11", "11o", "12", "12o", "13", "13o", "14", "8", "9", "9c", "9o"
]

def convert_coco_segmentation_to_yolo_seg(coco_json_path, images_dir, output_labels_dir):
    """
    將 COCO Segmentation JSON 轉換為 YOLOv8 Segmentation TXT 格式。
    """
    output_labels_dir.mkdir(parents=True, exist_ok=True)

    with open(coco_json_path, 'r') as f:
        coco_data = json.load(f)

    # 建立 category_id 到 class_name 的映射 (COCO 的 category_id 從 1 開始)
    # 並轉換為你的 0-based class_id
    id_to_name = {cat['id']: cat['name'] for cat in coco_data['categories']}
    name_to_yolo_id = {name: i for i, name in enumerate(CLASS_NAMES)}

    # 檢查 COCO 的 category ID 是否匹配你的 class_names 順序
    # 如果 COCO 的 category ID 不從 0 開始，或與你的類別順序不符，這裡可能需要調整
    # Roboflow 匯出的 COCO 通常會將你的類別 ID 從 0 或 1 開始映射好

    print(f"Processing {len(coco_data['images'])} images from {coco_json_path.name}...")

    for image_info in coco_data['images']:
        image_id = image_info['id']
        file_name = image_info['file_name']
        img_width = image_info['width']
        img_height = image_info['height']

        output_txt_path = output_labels_dir / f"{Path(file_name).stem}.txt"

        annotations_for_image = [
            ann for ann in coco_data['annotations'] if ann['image_id'] == image_id
        ]

        # 寫入 YOLO .txt 檔案
        with open(output_txt_path, 'w') as f_out:
            for ann in annotations_for_image:
                category_id_coco = ann['category_id']

                # 將 COCO category_id 映射到你的 YOLO class_id (0-based)
                # Roboflow 通常會直接將你的類別順序映射為 COCO 的 category_id (從1開始)
                # 所以 category_id_coco - 1 應該會對應到你的 0-based index
                # 但更安全的是透過名稱映射:
                class_name = id_to_name.get(category_id_coco)
                if class_name not in name_to_yolo_id:
                    print(
                        f"Warning: Class '{class_name}' from COCO not found in your CLASS_NAMES. Skipping annotation for image {file_name}.")
                    continue
                yolo_class_id = name_to_yolo_id[class_name]

                # segmentation 數據是一個列表，包含多邊形點 [x1, y1, x2, y2, ...]
                segmentation_points = ann['segmentation'][0]  # Roboflow 通常是單個多邊形，取第一個

                # 將像素座標正規化到 0-1 範圍
                normalized_points = []
                for i in range(0, len(segmentation_points), 2):
                    x_norm = segmentation_points[i] / img_width
                    y_norm = segmentation_points[i + 1] / img_height

                    x_norm = max(0.0, min(1.0, x_norm))
                    y_norm = max(0.0, min(1.0, y_norm))

                    normalized_points.extend([f"{x_norm:.6f}", f"{y_norm:.6f}"])  # 格式化為6位小數

                # 寫入一行：class_id x1_norm y1_norm x2_norm y2_norm ...
                f_out.write(f"{yolo_class_id} {' '.join(normalized_points)}\n")

        # 如果檔案是空的 (表示圖片沒有任何標註)，則刪除它
        if os.path.getsize(output_txt_path) == 0:
            os.remove(output_txt_path)


print("Starting COCO to YOLOv8 Segmentation Conversion...")

# 處理訓練集
convert_coco_segmentation_to_yolo_seg(
    coco_dataset_root / 'train' / '_annotations.coco.json',
    coco_dataset_root / 'train' / 'images',
    coco_dataset_root / 'train' / 'labels'  # 輸出到新的 labels 資料夾
)

# 處理驗證集
convert_coco_segmentation_to_yolo_seg(
    coco_dataset_root / 'valid' / '_annotations.coco.json',
    coco_dataset_root / 'valid' / 'images',
    coco_dataset_root / 'valid' / 'labels'  # 輸出到新的 labels 資料夾
)

# 處理測試集 (如果你想評估最終模型)
convert_coco_segmentation_to_yolo_seg(
    coco_dataset_root / 'test' / '_annotations.coco.json',
    coco_dataset_root / 'test' / 'images',
    coco_dataset_root / 'test' / 'labels'  # 輸出到新的 labels 資料夾
)

print("Conversion complete! YOLOv8 Segmentation labels are in the 'labels' subfolders.")