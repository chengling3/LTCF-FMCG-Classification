import os
import json
import cv2
import numpy as np
from glob import glob


def calculate_polygon_area(polygon):
    x = [p[0] for p in polygon]
    y = [p[1] for p in polygon]
    return 0.5 * abs(sum(x[i] * y[i + 1] - x[i + 1] * y[i] for i in range(-1, len(polygon) - 1)))


def calculate_center(polygon):
    x = [p[0] for p in polygon]
    y = [p[1] for p in polygon]
    return (sum(x) / len(polygon), sum(y) / len(polygon))


def calculate_importance(areas, positions, img_width, img_height,
                         area_weight=0.25, position_weight=0.75,
                         vertical_exponent=2.0, edge_threshold=0.05):
    print(f"Calculating importance scores - regions: {len(areas)}")
    areas = np.array(areas)
    if len(areas) == 0:
        return []

    min_area = np.min(areas)
    max_area = np.max(areas)
    normalized_areas = (areas - min_area) / (max_area - min_area + 1e-10)
    print(f"Normalized area range: {np.min(normalized_areas):.4f} ~ {np.max(normalized_areas):.4f}")

    def get_zone_boost(x, y):
        col = int(x / (img_width / 3))
        row = int(y / (img_height / 3))
        zone = row * 3 + col + 1
        return 1.2 if zone == 2 else 1.5 if zone == 5 else 0.3

    vertical_scores = []
    zone_boosts = []
    edge_penalties = []

    for x, y in positions:
        normalized_y = y / img_height
        vertical_score = np.exp(-vertical_exponent * normalized_y)
        vertical_scores.append(vertical_score)

        zone_boost = get_zone_boost(x, y)
        zone_boosts.append(zone_boost)

        left_dist = x / img_width
        right_dist = (img_width - x) / img_width
        top_dist = y / img_height
        bottom_dist = (img_height - y) / img_height
        min_dist = min(left_dist, right_dist, top_dist, bottom_dist)
        edge_penalty = max(min_dist / edge_threshold, 0.3) if min_dist < edge_threshold else 1.0
        edge_penalties.append(edge_penalty)

    position_scores = np.array(vertical_scores) * np.array(zone_boosts) * np.array(edge_penalties)
    print(f"Position score range: {np.min(position_scores):.4f} ~ {np.max(position_scores):.4f}")

    importance_scores = (area_weight * normalized_areas) + (position_weight * position_scores)
    print(f"Importance score range: {np.min(importance_scores):.4f} ~ {np.max(importance_scores):.4f}")
    return importance_scores.tolist()


def process_image(image_path):
    print(f"\nProcessing image: {image_path}")

    file_base = os.path.basename(image_path)
    file_name, _ = os.path.splitext(file_base)
    print(f"Image filename: {file_name}")

    json_file = os.path.join(os.path.dirname(image_path), f"{file_name}.json")
    if not os.path.exists(json_file):
        print(f"Warning: Skip {image_path}, JSON not found: {json_file}")
        return None
    print(f"Using JSON file: {json_file}")

    try:
        img = cv2.imread(image_path)
        if img is None:
            print(f"Warning: Skip {image_path}, cannot read image")
            return None
        img_height, img_width = img.shape[:2]
        print(f"Image size: {img_width}×{img_height}")
    except Exception as e:
        print(f"Warning: Skip {image_path}, error reading image: {str(e)}")
        return None

    try:
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        text_count = len(data.get('rec_texts', []))
        print(f"JSON loaded, contains {text_count} text regions")
    except Exception as e:
        print(f"Warning: Skip {image_path}, error reading JSON: {str(e)}")
        return None

    try:
        polygons = data["rec_polys"]
        rec_texts = data["rec_texts"]
        rec_scores = data["rec_scores"]
        print(f"Extracted {len(polygons)} polygons, {len(rec_texts)} texts, {len(rec_scores)} scores")
    except KeyError as e:
        print(f"Warning: Skip {image_path}, missing field: {str(e)}")
        return None

    if len(polygons) == 0 or len(rec_texts) == 0:
        print(f"Warning: Skip {image_path}, no text detected")
        return None

    areas = [calculate_polygon_area(poly) for poly in polygons]
    positions = [calculate_center(poly) for poly in polygons]
    print(f"Area range: {min(areas):.1f} ~ {max(areas):.1f}")

    importance_scores = calculate_importance(
        areas, positions,
        img_width=img_width,
        img_height=img_height
    )

    if not importance_scores:
        print(f"Warning: Skip {image_path}, cannot calculate importance scores")
        return None

    results = []
    for i, (text, area, pos, score, orig_score) in enumerate(zip(
            rec_texts, areas, positions, importance_scores, rec_scores)):
        if not text.strip():
            print(f"Skipping empty text: index={i}, raw='{text}'")
            continue
        results.append({
            'text': text,
            'importance_score': score
        })

    if not results:
        print(f"Warning: Skip {image_path}, no valid text after filtering")
        return None

    results.sort(key=lambda x: x['importance_score'], reverse=True)

    if len(results) == 1:
        print(f"Sorted: 1 valid text, score: {results[0]['importance_score']:.4f}")
    elif len(results) >= 2:
        print(f"Sorted: top 2 scores: {results[0]['importance_score']:.4f}, {results[1]['importance_score']:.4f}")
    else:
        print("Warning: Empty results after sorting")
        return None

    return {
        'image_name': file_base,
        'results': results,
        'json_file': json_file,
    }


def save_top_texts(image_data):
    if not image_data or not image_data['results']:
        print("Error: No results to save")
        return False

    results = image_data['results']
    json_file = image_data['json_file']
    print(f"JSON path: {json_file}")

    image_base = os.path.splitext(os.path.basename(json_file))[0]
    image_folder = os.path.join(os.path.dirname(json_file), image_base)
    os.makedirs(image_folder, exist_ok=True)
    txt_file = os.path.join(image_folder, image_base + '.txt')
    print(f"Saving TXT to: {txt_file}")

    try:
        with open(txt_file, 'w', encoding='utf-8') as f:
            for item in results:
                text = item['text'].strip()
                score = item['importance_score']
                if text:
                    f.write(f"{text}\t{score:.4f}\n")
                else:
                    print(f"Skipping empty text: score={score:.4f}")
        print(f"Saved successfully: {txt_file}")
        return True
    except Exception as e:
        print(f"Error: Cannot save {txt_file}: {str(e)}")
        return False


if __name__ == "__main__":
    root_folder = "/...../RP_203/image/train/"

    image_extensions = ('*.jpg', '*.jpeg', '*.png')
    image_files = []
    for ext in image_extensions:
        image_files.extend(glob(os.path.join(root_folder, '**', ext), recursive=True))

    if not image_files:
        print(f"Error: No images found in {root_folder}")
        exit()

    print(f"Starting processing root folder: {root_folder}")
    print(f"Found {len(image_files)} images")

    processed_count = 0
    saved_count = 0
    blank_text_count = 0
    for img_file in image_files:
        image_data = process_image(img_file)
        if image_data:
            processed_count += 1
            blank_texts = sum(1 for item in image_data.get('results', []) if not item['text'].strip())
            if blank_texts > 0:
                blank_text_count += blank_texts
                print(f"Filtered {blank_texts} empty texts in this image")
            if save_top_texts(image_data):
                saved_count += 1

    print(f"\n{'=' * 60}")
    print("Processing Summary:")
    print(f"- Total images: {len(image_files)}")
    print(f"- Successfully processed: {processed_count}")
    print(f"- TXT files saved: {saved_count}")
    print(f"- Empty texts filtered: {blank_text_count}")