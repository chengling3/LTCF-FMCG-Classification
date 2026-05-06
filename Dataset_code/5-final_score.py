import re
import json
import os
import torch
from typing import List, Tuple, Dict

def char_type(c):
    if '\u4e00' <= c <= '\u9fff':
        return 'chinese'
    elif c.isascii() and c.isalpha():
        return 'english'
    elif c.isdigit():
        return 'digit'
    else:
        return 'other'


char_type_weights = {
    'chinese': 2.0,
    'english': 1.5,
    'digit': 1.0
}

coefficients = torch.tensor([0.25, 0.25, 0.25, 0.25], dtype=torch.float32)

def calculate_similarity(input_raw: str, candidate: str) -> Tuple[float, Dict[str, float]]:
    input_clean = input_raw.strip()
    candidate_base = re.sub(r'[（\(].*?[）\)]', '', candidate).strip()

    input_types = {char: char_type(char) for char in input_clean}
    candidate_types = {char: char_type(char) for char in candidate_base}
    common_chars = set(input_clean) & set(candidate_base)

    type_counts = {'chinese': 0, 'english': 0, 'digit': 0}
    for c in common_chars:
        t1 = input_types.get(c)
        t2 = candidate_types.get(c)
        if t1 == t2 and t1 in type_counts:
            type_counts[t1] += 1

    char_match_score = sum(type_counts[t] * char_type_weights[t] for t in type_counts)

    max_consecutive = 0
    for i in range(len(input_clean)):
        for j in range(len(candidate_base)):
            k = 0
            while (i + k < len(input_clean) and j + k < len(candidate_base) and input_clean[i + k] == candidate_base[
                j + k]):
                k += 1
            if k > max_consecutive:
                max_consecutive = k
    consecutive_score = max_consecutive ** 2 if max_consecutive > 1 else 0

    position_score = 0
    min_len = min(len(input_clean), len(candidate_base))
    for i in range(min_len):
        if input_clean[i] == candidate_base[i]:
            position_score += 3

    seq_score = 0
    m, n = len(input_clean), len(candidate_base)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if input_clean[i - 1] == candidate_base[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
    lcs_length = dp[m][n]
    seq_score = lcs_length ** 2 if lcs_length > 1 else 0

    scores = torch.tensor([char_match_score, consecutive_score, position_score, seq_score], dtype=torch.float32)
    adjusted_scores = scores * coefficients
    total_score = adjusted_scores.sum().item()

    detail_scores = {
        "char": round(char_match_score, 2),
        "consecutive": consecutive_score,
        "position": position_score,
        "sequence": seq_score
    }
    return total_score, detail_scores

def calculate_all_scores(input_txt: str, candidates: List[str],
                         importance_list: List[Tuple[str, float]] = None) -> List[Tuple[str, float, Dict[str, float]]]:
    input_raw = input_txt.strip()
    scores = [(candidate, *calculate_similarity(input_raw, candidate)) for candidate in candidates]

    if importance_list:
        for i, (candidate, total, detail) in enumerate(scores):
            bonus = 0.0
            for recog_text, imp_score in importance_list:
                if set(candidate) & set(recog_text):
                    bonus += imp_score
            if bonus > 0:
                print(f"[Bonus] Label: {candidate} Original: {total:.2f} Bonus: {bonus:.2f} → New total: {total + bonus:.2f}")
                detail['char'] += bonus
            scores[i] = (
            candidate, detail['char'] + detail['consecutive'] + detail['position'] + detail['sequence'], detail)

    scores.sort(key=lambda x: x[1], reverse=True)
    return scores


def process_file(file_path: str, candidates: List[str], output_dir: str, save_results: bool = True) -> None:
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            text = f.read().strip()
        if not text or text == "None":
            print(f"File {file_path} is empty or 'None', skipping")
            return
        input_raw = text.strip()

        importance_list = [(input_raw, 1.0)]
        try:
            base_file = os.path.basename(file_path)
            base_name, _ = os.path.splitext(base_file)
            image_dir = os.path.join(os.path.dirname(file_path), base_name)
            if os.path.exists(image_dir) and os.path.isdir(image_dir):
                for fname in os.listdir(image_dir):
                    if fname.endswith('_res.txt'):
                        imp_path = os.path.join(image_dir, fname)
                        with open(imp_path, 'r', encoding='utf-8') as imp_file:
                            for line in imp_file:
                                parts = line.strip().split('\t')
                                if len(parts) == 2:
                                    recog_text, score = parts
                                    try:
                                        importance_list.append((recog_text.strip(), float(score)))
                                    except ValueError:
                                        continue
        except Exception as e:
            print(f"Error reading bonus file: {str(e)}")

        print(f"\n{'=' * 60}")
        print(f"File: {os.path.basename(file_path)}")
        print(f"Content: '{text}'")
        print("[Debug] importance_list:")
        for rtext, weight in importance_list:
            print(f"  - '{rtext}' (weight={weight})")

        scores = calculate_all_scores(input_raw, candidates, importance_list)
        print("Top 5 matches:")
        for candidate, total, detail in scores[:5]:
            print(
                f"  - {candidate}: total={total:.2f}, char={detail['char']:.2f}, consecutive={detail['consecutive']}, position={detail['position']}, sequence={detail['sequence']}")

        if save_results:
            file_name = os.path.basename(file_path)
            base_name, _ = os.path.splitext(file_name)
            output_path = os.path.join(output_dir, f"{base_name}.txt")
            os.makedirs(output_dir, exist_ok=True)
            score_dict = {candidate: (total, detail) for candidate, total, detail in scores}
            with open(output_path, 'w', encoding='utf-8') as f:
                for candidate in candidates:
                    total, detail = score_dict.get(candidate,
                                                   (0, {"char": 0, "consecutive": 0, "position": 0, "sequence": 0}))
                    f.write(
                        f"{candidate}: {detail['char']:.2f} {detail['consecutive']} {detail['position']} {detail['sequence']}\n")
            print(f"Similarity scores saved to: {output_path}")
    except Exception as e:
        print(f"\nError processing file {file_path}: {str(e)}")


def process_folder(input_dir: str, json_path: str, output_dir: str, save_results: bool = True) -> None:
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            class_indices = json.load(f)
            sorted_items = sorted(class_indices.items(), key=lambda x: int(x[0]))
            candidates = [item[1] for item in sorted_items]
        print(f"Loaded {len(candidates)} candidate labels")
    except Exception as e:
        print(f"Failed to load candidate set: {str(e)}")
        return

    txt_files = []
    for root, _, files in os.walk(input_dir):
        for file in files:
            if file.endswith('.txt') and not file.endswith('_res.txt'):
                txt_files.append(os.path.join(root, file))

    if not txt_files:
        print(f"No main OCR TXT files found in {input_dir}")
        return

    print(f"Processing {len(txt_files)} main text files...")
    for candidate in candidates:
        os.makedirs(os.path.join(output_dir, candidate), exist_ok=True)

    for file_path in txt_files:
        rel_path = os.path.relpath(file_path, input_dir)
        parts = rel_path.split(os.sep)
        if len(parts) >= 2:
            category = parts[0]
        else:
            print(f"Warning: Cannot extract category from path {rel_path}, skipping")
            continue
        category_dir = os.path.join(output_dir, category)
        os.makedirs(category_dir, exist_ok=True)
        process_file(file_path, candidates, category_dir, save_results)


def main():
    input_dir = "/...../RP_203/image/train/"
    json_file = "/...../RP_203/image/class_indices.json"
    output_dir = "/...../RP_203/image/similarity_scores/train"
    print("=" * 60)
    print("Starting text matching score calculation (4 rules + char weights + OCR importance bonus)")
    print(f"Input dir: {input_dir}")
    print(f"Candidate file: {json_file}")
    print(f"Output dir: {output_dir}")
    print("=" * 60)
    process_folder(input_dir, json_file, output_dir, save_results=True)


if __name__ == "__main__":
    main()