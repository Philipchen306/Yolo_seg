import os
from pathlib import Path
from PIL import Image  # 仍然需要 PIL 來打開圖片文件
import logging
import torch  # 假設你的模型是基於 PyTorch 的
from torchvision import transforms  # 常用於圖像預處理

# --- 配置日誌 ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- 核心配置 (請務必根據你的專案修改這裡) ---
# 你的數據集根目錄
# 範例：DATASET_ROOT_PATH = Path('/Users/你的用戶名/Yolo_seg/Faucet2_dataset_v3/')
DATASET_ROOT_PATH = Path('Faucet2_dataset_v3/')  # 假設 debug.py 和 Faucet2_dataset_v3 在同一個目錄

# 存放處理失敗圖片的隔離資料夾 (這個腳本會自動創建，但請確保它與數據集目錄不重疊)
CORRUPTED_IMAGES_DIR = Path('./problematic_images_for_model_input/')

# --- 你的模型期望的圖片尺寸 (imgsz) ---
# 這個尺寸必須與你的訓練腳本 (train_seg_model.py) 中使用的圖片輸入尺寸一致
IMG_SIZE = (640, 640)  # 替換為你的模型實際使用的尺寸，例如 (640, 640) 或 (416, 416)

# --- 檢查配置路徑的有效性 ---
if not DATASET_ROOT_PATH.is_dir():
    logger.error(f"Error: DATASET_ROOT_PATH '{DATASET_ROOT_PATH.resolve()}' is not a valid directory.")
    exit(1)  # 如果路徑不對，直接退出程式

if CORRUPTED_IMAGES_DIR and not CORRUPTED_IMAGES_DIR.exists():
    CORRUPTED_IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    logger.info(f"Created quarantine directory: {CORRUPTED_IMAGES_DIR.resolve()}")


# --- 關鍵部分：模擬模型輸入的圖片載入和預處理管道 ---
# 你 **必須** 根據你的 `train_seg_model.py` 中實際的數據載入和轉換邏輯來修改這個函式。
# 這裡提供一個常見的 PyTorch/torchvision 範例。
def load_and_preprocess_image_for_model(image_path: Path):
    """
    模擬載入、預處理圖片的過程，使其符合模型輸入的格式。
    這個函式必須與你的模型實際的數據載入管道完全一致。
    """
    # 步驟 1: 打開圖片 (Pillow 是最常用的 Python 圖片處理庫)
    # .convert('RGB') 確保圖片有 3 個通道，對於許多模型是必須的。
    with Image.open(image_path).convert('RGB') as img:
        # 步驟 2: 定義並應用圖像轉換 (Transforms)
        # 這裡的轉換必須與你 train_seg_model.py 中用於訓練的 `transforms` 或 `augmentations` 保持一致。
        # 確保 `Resize` 的尺寸與 `IMG_SIZE` 匹配。
        # `ToTensor()` 會將 PIL Image 轉換為 PyTorch Tensor，並將像素值從 [0, 255] 縮放到 [0.0, 1.0]。
        transform = transforms.Compose([
            transforms.Resize(IMG_SIZE),  # 將圖片縮放到模型期望的尺寸
            transforms.ToTensor(),  # 將 PIL Image 轉換為 PyTorch Tensor
            # 許多模型還會進行歸一化 (Normalization)。如果你的模型有，請在這裡添加：
            # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            # 這些 mean 和 std 值應與你模型訓練時使用的值相同。
        ])

        img_tensor = transform(img)

        # (可選) 在這裡添加對 Tensor 尺寸的檢查，確保它符合模型輸入的 [C, H, W]
        # if img_tensor.shape != (3, IMG_SIZE[0], IMG_SIZE[1]):
        #     raise ValueError(f"Processed image tensor has incorrect shape: {img_tensor.shape}")

        return img_tensor  # 返回預處理後的 Tensor


# --- 主要 Debugging 腳本 ---
def debug_model_input_pipeline(root_dir: Path, quarantine_dir: Path = None):
    """
    遍歷數據集中的所有圖片，嘗試將它們載入並預處理，
    模擬餵給模型，並在出錯時報告問題圖片。
    """
    error_count = 0
    total_count = 0
    problematic_images = []  # 儲存出錯圖片的路徑列表

    logger.info(f"Starting model input pipeline debugging in: {root_dir.resolve()}")

    # 定義要檢查的圖片文件擴展名
    image_extensions = ('*.png', '*.jpg', '*.jpeg', '*.gif', '*.bmp', '*.tiff')
    all_image_paths = []
    # 遞歸查找根目錄下所有符合擴展名的圖片文件
    for ext in image_extensions:
        all_image_paths.extend(list(root_dir.rglob(ext)))

    total_count = len(all_image_paths)
    if total_count == 0:
        logger.warning(f"No image files found in '{root_dir.resolve()}'. Please check DATASET_ROOT_PATH.")
        return

    logger.info(f"Found {total_count} image files to check for model input.")

    for i, image_path_obj in enumerate(all_image_paths):
        image_path_str = str(image_path_obj)  # 轉換為字串用於日誌和錯誤報告

        # 每處理一定數量或快完成時，打印進度
        if (i + 1) % 100 == 0 or (i + 1) == total_count:
            logger.info(f"Processed {i + 1}/{total_count} images. Found {error_count} issues so far.")

        try:
            # 呼叫你自定義的圖片載入和預處理函式
            processed_data = load_and_preprocess_image_for_model(image_path_obj)

            # (可選) 如果你想在這裡進一步模擬模型的「最小前向傳播」，可以這麼做：
            # 這需要載入你的模型或模型的最小部分。
            # 例如：
            # if your_model_instance is not None:
            #     # 假設模型期望的是批次輸入，需要添加一個批次維度
            #     dummy_model_input = processed_data.unsqueeze(0)
            #     # dummy_output = your_model_instance(dummy_model_input)
            #     # 你可以檢查 dummy_output 的基本屬性，例如形狀。

        except Exception as e:
            # 捕獲任何在載入或預處理過程中發生的錯誤
            error_count += 1
            problematic_images.append(image_path_str)  # 記錄出錯圖片的路徑
            logger.error(f"Error processing image for model input: {image_path_str} - Error: {e}",
                         exc_info=True)  # exc_info=True 打印詳細追溯

            if quarantine_dir:  # 如果設定了隔離資料夾，就移動圖片
                try:
                    relative_path = image_path_obj.relative_to(root_dir)
                    target_path = quarantine_dir / relative_path
                    target_path.parent.mkdir(parents=True, exist_ok=True)  # 確保目標目錄存在

                    os.rename(image_path_str, str(target_path))  # 移動文件
                    logger.info(f"Moved problematic image to quarantine: {target_path.resolve()}")
                except Exception as move_e:
                    logger.error(f"Failed to move problematic image {image_path_str}: {move_e}")

    logger.info(f"Debugging complete. Total images checked: {total_count}, Errors found: {error_count}")
    if error_count > 0:
        logger.warning(f"Please review the {error_count} images that caused errors in the model input pipeline:")
        for path in problematic_images:
            logger.warning(f" - {path}")
    else:
        logger.info("All images processed successfully through the model input pipeline. Good to go!")


if __name__ == "__main__":
    # --- 運行數據集驗證 ---
    # 確保 DATASET_ROOT_PATH 和 CORRUPTED_IMAGES_DIR 的設定與你的檔案結構相符
    # 例如，如果 debug.py 在 Yolo_seg/，數據集在 Yolo_seg/Faucet2_dataset_v3/
    # DATASET_ROOT_PATH = Path('./Faucet2_dataset_v3/') # 這會是相對於運行腳本的位置
    # CORRUPTED_IMAGES_DIR = Path('./problematic_images_for_model_input_quarantine/') # 隔離資料夾

    debug_model_input_pipeline(DATASET_ROOT_PATH, CORRUPTED_IMAGES_DIR)