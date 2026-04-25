import os
import json
import cv2
import numpy as np
from glob import glob


def calculate_polygon_area(polygon):
    """计算多边形的面积（鞋带公式）"""
    x = [p[0] for p in polygon]
    y = [p[1] for p in polygon]
    return 0.5 * abs(sum(x[i] * y[i + 1] - x[i + 1] * y[i] for i in range(-1, len(polygon) - 1)))


def calculate_center(polygon):
    """计算多边形的中心点"""
    x = [p[0] for p in polygon]
    y = [p[1] for p in polygon]
    return (sum(x) / len(polygon), sum(y) / len(polygon))


def calculate_importance(areas, positions, img_width, img_height,
                         area_weight=0.25, position_weight=0.75,
                         vertical_exponent=2.0, edge_threshold=0.05):
    """计算重要性分数"""
    print(f"计算重要性分数 - 区域数量: {len(areas)}")
    areas = np.array(areas)
    if len(areas) == 0:
        return []

    # 归一化面积
    min_area = np.min(areas)
    max_area = np.max(areas)
    normalized_areas = (areas - min_area) / (max_area - min_area + 1e-10)
    print(f"归一化面积范围: {np.min(normalized_areas):.4f} ~ {np.max(normalized_areas):.4f}")

    # 计算九宫格区域和对应的加成
    def get_zone_boost(x, y):
        col = int(x / (img_width / 3))
        row = int(y / (img_height / 3))
        zone = row * 3 + col + 1  # 1-9
        return 1.2 if zone == 2 else 1.5 if zone == 5 else 0.3

    vertical_scores = []
    zone_boosts = []
    edge_penalties = []

    for x, y in positions:
        # 基础垂直得分
        normalized_y = y / img_height
        vertical_score = np.exp(-vertical_exponent * normalized_y)
        vertical_scores.append(vertical_score)

        # 区域加成
        zone_boost = get_zone_boost(x, y)
        zone_boosts.append(zone_boost)

        # 边缘惩罚
        left_dist = x / img_width
        right_dist = (img_width - x) / img_width
        top_dist = y / img_height
        bottom_dist = (img_height - y) / img_height
        min_dist = min(left_dist, right_dist, top_dist, bottom_dist)
        edge_penalty = max(min_dist / edge_threshold, 0.3) if min_dist < edge_threshold else 1.0
        edge_penalties.append(edge_penalty)

    # 应用区域加成和边缘惩罚
    position_scores = np.array(vertical_scores) * np.array(zone_boosts) * np.array(edge_penalties)
    print(f"位置分数范围: {np.min(position_scores):.4f} ~ {np.max(position_scores):.4f}")

    # 综合得分
    importance_scores = (area_weight * normalized_areas) + (position_weight * position_scores)
    print(f"重要性分数范围: {np.min(importance_scores):.4f} ~ {np.max(importance_scores):.4f}")
    return importance_scores.tolist()


def process_image(image_path):
    """处理单张图片及其对应的JSON文件"""
    print(f"\n处理图片: {image_path}")

    # 提取图片文件名（去除扩展名）
    file_base = os.path.basename(image_path)
    file_name, _ = os.path.splitext(file_base)
    print(f"图片文件名: {file_name}")

    # 构建JSON文件路径（与图片同级、同名）
    json_file = os.path.join(os.path.dirname(image_path), f"{file_name}.json")
    if not os.path.exists(json_file):
        print(f"警告：跳过 {image_path}，JSON文件不存在: {json_file}")
        return None
    print(f"使用JSON文件: {json_file}")


    # 读取图片获取尺寸
    try:
        img = cv2.imread(image_path)
        if img is None:
            print(f"警告：跳过 {image_path}，无法读取图片")
            return None
        img_height, img_width = img.shape[:2]
        print(f"图片尺寸: {img_width}×{img_height}")
    except Exception as e:
        print(f"警告：跳过 {image_path}，读取图片出错: {str(e)}")
        return None

    # 读取JSON数据
    try:
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        text_count = len(data.get('rec_texts', []))
        print(f"成功读取JSON文件，包含 {text_count} 个文本区域")
    except Exception as e:
        print(f"警告：跳过 {image_path}，读取JSON文件出错: {str(e)}")
        return None

    # 提取数据
    try:
        polygons = data["rec_polys"]
        rec_texts = data["rec_texts"]
        rec_scores = data["rec_scores"]
        print(f"提取到 {len(polygons)} 个多边形，{len(rec_texts)} 个文本，{len(rec_scores)} 个分数")
    except KeyError as e:
        print(f"警告：跳过 {image_path}，JSON文件缺少必要字段: {str(e)}")
        return None

    # 检查是否有识别结果
    if len(polygons) == 0 or len(rec_texts) == 0:
        print(f"警告：跳过 {image_path}，无文本识别结果")
        return None

    # 计算区域数据
    areas = [calculate_polygon_area(poly) for poly in polygons]
    positions = [calculate_center(poly) for poly in polygons]
    print(f"计算区域数据 - 面积范围: {min(areas):.1f} ~ {max(areas):.1f}")

    # 计算重要性分数
    importance_scores = calculate_importance(
        areas, positions,
        img_width=img_width,
        img_height=img_height
    )

    # 检查重要性分数是否为空
    if not importance_scores:
        print(f"警告：跳过 {image_path}，无法计算重要性分数")
        return None

    # 组装结果并按重要性排序，同时过滤空白文本
    results = []
    for i, (text, area, pos, score, orig_score) in enumerate(zip(
            rec_texts, areas, positions, importance_scores, rec_scores)):
        # 过滤空白文本（包括纯空格、换行符等）
        if not text.strip():
            print(f"跳过空白文本: 索引={i}, 原始文本='{text}'")
            continue
        results.append({
            'text': text,
            'importance_score': score
        })

    # 再次检查结果是否为空（过滤后）
    if not results:
        print(f"警告：跳过 {image_path}，过滤后无有效文本")
        return None

    # 按重要性排序
    results.sort(key=lambda x: x['importance_score'], reverse=True)

    # 处理文本区域数量少于2的情况
    if len(results) == 1:
        print(f"排序后结果 - 仅有1个有效文本，重要性分数: {results[0]['importance_score']:.4f}")
    elif len(results) >= 2:
        print(
            f"排序后结果 - 前2名重要性分数: {results[0]['importance_score']:.4f}，{results[1]['importance_score']:.4f}")
    else:
        print("警告：排序后结果为空")
        return None

    return {
        'image_name': file_base,
        'results': results,
        'json_file': json_file,
    }


def save_top_texts(image_data):
    """保存每个识别文本及其重要性分数到txt文件"""
    if not image_data or not image_data['results']:
        print("错误：无结果数据，无法保存TXT文件")
        return False

    results = image_data['results']
    json_file = image_data['json_file']
    print(f"JSON文件路径: {json_file}")

    # 构建保存路径：放入与图片同名的子文件夹中
    image_base = os.path.splitext(os.path.basename(json_file))[0]
    image_folder = os.path.join(os.path.dirname(json_file), image_base)
    os.makedirs(image_folder, exist_ok=True)
    txt_file = os.path.join(image_folder, image_base + '.txt')
    print(f"TXT文件将保存至: {txt_file}")

    try:
        with open(txt_file, 'w', encoding='utf-8') as f:
            for item in results:
                text = item['text'].strip()
                score = item['importance_score']
                if text:
                    f.write(f"{text}\t{score:.4f}\n")
                else:
                    print(f"跳过保存空白文本: 重要性分数={score:.4f}")
        print(f"成功保存: {txt_file}")
        return True
    except Exception as e:
        print(f"错误：无法保存文件 {txt_file}: {str(e)}")
        return False



# 主程序
if __name__ == "__main__":
    # 指定要处理的根文件夹路径
    root_folder = "/...../RP_203/image/train/"

    # 查找所有图片文件（递归查找）
    image_extensions = ('*.jpg', '*.jpeg', '*.png')
    image_files = []
    for ext in image_extensions:
        image_files.extend(glob(os.path.join(root_folder, '**', ext), recursive=True))

    if not image_files:
        print(f"错误：在 {root_folder} 及其子文件夹中没有找到图片文件")
        exit()

    print(f"开始处理根文件夹: {root_folder}")
    print(f"找到 {len(image_files)} 张图片")

    # 处理每张图片
    processed_count = 0
    saved_count = 0
    blank_text_count = 0  # 新增：统计过滤的空白文本数量
    for img_file in image_files:
        image_data = process_image(img_file)
        if image_data:
            processed_count += 1
            # 统计该图片中过滤的空白文本数量
            blank_texts = sum(1 for item in image_data.get('results', []) if not item['text'].strip())
            if blank_texts > 0:
                blank_text_count += blank_texts
                print(f"该图片过滤了 {blank_texts} 个空白文本")
            if save_top_texts(image_data):
                saved_count += 1

    # 打印统计信息
    print(f"\n{'=' * 60}")
    print("处理统计信息:")
    print(f"- 总图片数: {len(image_files)}")
    print(f"- 成功处理图片数: {processed_count}")
    print(f"- 成功保存txt文件数: {saved_count}")
    print(f"- 总共过滤空白文本数: {blank_text_count}")