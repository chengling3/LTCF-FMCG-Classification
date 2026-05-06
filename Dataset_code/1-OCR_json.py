import os
import cv2
from paddleocr import PaddleOCR
from pathlib import Path
import time
from tqdm import tqdm


def batch_ocr_process(dataset_folder, supported_formats=None):
    if supported_formats is None:
        supported_formats = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif']

    print("Initializing PaddleOCR...")
    ocr = PaddleOCR(
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
        use_textline_orientation=True
    )
    print("PaddleOCR initialization completed\n")

    processed_images = 0
    skipped_images = 0
    error_images = 0

    dataset_path = Path(dataset_folder)

    if not dataset_path.exists():
        print(f"Error: Dataset folder {dataset_folder} does not exist")
        return

    print("Scanning image files...")
    image_files = []
    for image_path in dataset_path.rglob('*'):
        if image_path.is_file() and image_path.suffix.lower() in supported_formats:
            image_files.append(image_path)

    total_images = len(image_files)
    if total_images == 0:
        print("No image files found")
        return

    print(f"Found {total_images} images\n")

    with tqdm(total=total_images, desc="OCR Processing", unit="img") as pbar:
        for image_path in image_files:
            json_path = image_path.with_suffix('.json')

            if json_path.exists():
                skipped_images += 1
                pbar.set_postfix({
                    'Processed': processed_images,
                    'Skipped': skipped_images,
                    'Errors': error_images
                })
                pbar.update(1)
                continue

            try:
                current_file = image_path.name
                if len(current_file) > 30:
                    current_file = current_file[:27] + "..."
                pbar.set_description(f"Processing: {current_file}")

                result = ocr.predict(input=str(image_path))

                for res in result:
                    res.save_to_json(str(json_path))

                processed_images += 1

            except Exception as e:
                error_images += 1
                tqdm.write(f"Error: {image_path.relative_to(dataset_path)} - {str(e)}")

            pbar.set_postfix({
                'Processed': processed_images,
                'Skipped': skipped_images,
                'Errors': error_images
            })
            pbar.update(1)

    print("\n" + "=" * 50)
    print("Batch processing completed!")
    print(f"Total images: {total_images}")
    print(f"Successfully processed: {processed_images}")
    print(f"Skipped (existing): {skipped_images}")
    print(f"Failed: {error_images}")
    if total_images > 0:
        success_rate = (processed_images + skipped_images) / total_images * 100
        print(f"Success rate: {success_rate:.1f}%")
    print("=" * 50)


def batch_ocr_process_simple(dataset_folder, supported_formats=None):
    if supported_formats is None:
        supported_formats = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif']

    ocr = PaddleOCR(
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
        use_textline_orientation=True
    )

    dataset_path = Path(dataset_folder)
    if not dataset_path.exists():
        print(f"Error: Directory {dataset_folder} does not exist")
        return

    image_files = [p for p in dataset_path.rglob('*')
                   if p.is_file() and p.suffix.lower() in supported_formats]

    if not image_files:
        print("No image files found")
        return

    processed = skipped = errors = 0

    for image_path in tqdm(image_files, desc="OCR Recognition", unit="img"):
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

    print(f"\nCompleted: {processed} processed, {skipped} skipped, {errors} failed")


def main():
    folder_path = "/...../RP_203/image/train/"
    print(f"Using default dataset path: {folder_path}")

    print("Using default mode: detailed progress bar")
    batch_ocr_process(folder_path)


if __name__ == "__main__":
    main()