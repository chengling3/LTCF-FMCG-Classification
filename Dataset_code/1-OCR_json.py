import os
import cv2
from paddleocr import PaddleOCR
from pathlib import Path
import time
from tqdm import tqdm


def batch_ocr_process(dataset_folder, supported_formats=None):
    """
    批量处理数据集文件夹中的所有图片，进行OCR识别并保存JSON结果

    Args:
        dataset_folder: 数据集文件夹路径
        supported_formats: 支持的图片格式列表，默认为常见图片格式
    """
    if supported_formats is None:
        supported_formats = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif']

    # 初始化PaddleOCR实例
    print("正在初始化PaddleOCR...")
    ocr = PaddleOCR(
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
        use_textline_orientation=True
    )
    print("PaddleOCR初始化完成\n")

    # 统计信息
    processed_images = 0
    skipped_images = 0
    error_images = 0

    # 遍历数据集文件夹及其子文件夹
    dataset_path = Path(dataset_folder)

    if not dataset_path.exists():
        print(f"错误: 数据集文件夹 {dataset_folder} 不存在")
        return

    # 收集所有图片文件
    print("正在扫描图片文件...")
    image_files = []
    for image_path in dataset_path.rglob('*'):
        if image_path.is_file() and image_path.suffix.lower() in supported_formats:
            image_files.append(image_path)

    total_images = len(image_files)
    if total_images == 0:
        print("未找到任何图片文件")
        return

    print(f"找到 {total_images} 张图片\n")

    # 使用tqdm创建进度条
    with tqdm(total=total_images, desc="OCR处理进度", unit="张") as pbar:
        for image_path in image_files:
            # 生成对应的JSON文件路径
            json_path = image_path.with_suffix('.json')

            # 检查JSON文件是否已存在
            if json_path.exists():
                skipped_images += 1
                pbar.set_postfix({
                    '已处理': processed_images,
                    '跳过': skipped_images,
                    '错误': error_images
                })
                pbar.update(1)
                continue

            try:
                # 更新进度条描述显示当前处理的文件
                current_file = image_path.name
                if len(current_file) > 30:
                    current_file = current_file[:27] + "..."
                pbar.set_description(f"处理: {current_file}")

                # 进行OCR识别
                result = ocr.predict(input=str(image_path))

                # 保存JSON结果
                for res in result:
                    res.save_to_json(str(json_path))

                processed_images += 1

            except Exception as e:
                error_images += 1
                # 只在出错时打印错误信息
                tqdm.write(f"错误: {image_path.relative_to(dataset_path)} - {str(e)}")

            # 更新进度条后缀信息
            pbar.set_postfix({
                '已处理': processed_images,
                '跳过': skipped_images,
                '错误': error_images
            })
            pbar.update(1)

    # 最终统计结果
    print("\n" + "=" * 50)
    print("批量处理完成!")
    print(f"总图片数: {total_images}")
    print(f"成功处理: {processed_images}")
    print(f"跳过 (已存在): {skipped_images}")
    print(f"处理失败: {error_images}")
    if total_images > 0:
        success_rate = (processed_images + skipped_images) / total_images * 100
        print(f"成功率: {success_rate:.1f}%")
    print("=" * 50)


def batch_ocr_process_simple(dataset_folder, supported_formats=None):
    """
    简化版本 - 最少输出的批量处理
    """
    if supported_formats is None:
        supported_formats = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif']

    # 静默初始化OCR
    ocr = PaddleOCR(
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
        use_textline_orientation=True
    )

    dataset_path = Path(dataset_folder)
    if not dataset_path.exists():
        print(f"错误: 目录 {dataset_folder} 不存在")
        return

    # 收集图片文件
    image_files = [p for p in dataset_path.rglob('*')
                   if p.is_file() and p.suffix.lower() in supported_formats]

    if not image_files:
        print("未找到图片文件")
        return

    # 简洁的进度条
    processed = skipped = errors = 0

    for image_path in tqdm(image_files, desc="OCR识别", unit="张"):
        json_path = image_path.with_suffix('.json')

        if json_path.exists():
            skipped += 1
            continue

        try:
            result = ocr.predict(input=str(image_path))
            for res in result:
                res.save_to_json(str(json_path))
            processed += 1
        except Exception:
            errors += 1

    print(f"\n完成: {processed}张处理, {skipped}张跳过, {errors}张失败")


def main():
    # 默认数据集文件夹路径
    folder_path = "/...../RP_203/image/train/"
    print(f"使用默认数据集路径: {folder_path}")

    # 使用默认模式（详细进度条模式）
    print("使用默认模式: 详细进度条")
    batch_ocr_process(folder_path)


if __name__ == "__main__":
    main()
