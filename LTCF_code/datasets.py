import os
import torch
import json
from torch.utils.data import Dataset
from PIL import Image
import numpy as np


class BatchDataset(Dataset):
    def __init__(self, root_dir, label_file, similarity_dir=None, transform=None, is_json=True):
        """
        Args:
            root_dir (string): 图像文件根目录
            label_file (string): 标签文件路径
            similarity_dir (string): 相似度分数文件目录（默认为None，与图像在同一目录）
            transform (callable, optional): 图像转换操作
            is_json (bool): 标签文件是否为JSON格式
        """
        self.root_dir = root_dir
        self.similarity_dir = similarity_dir or root_dir  # 默认与图像在同一目录
        self.transform = transform
        self.is_json = is_json

        # 加载标签列表
        self.labels = self._load_labels(label_file)

        # 生成数据集样本列表
        self.samples = self._make_dataset()

    def _load_labels(self, label_file):
        """根据文件格式加载标签"""
        if self.is_json:
            with open(label_file, 'r', encoding='utf-8') as f:
                class_indices = json.load(f)
            sorted_items = sorted(class_indices.items(), key=lambda x: int(x[0]))
            return [item[1] for item in sorted_items]
        else:
            with open(label_file, 'r', encoding='utf-8') as f:
                return [line.strip() for line in f.readlines()]

    def _make_dataset(self):
        samples = []
        for label_idx, label in enumerate(self.labels):
            label_dir = os.path.join(self.root_dir, label)
            if not os.path.isdir(label_dir):
                continue

            for img_name in os.listdir(label_dir):
                if img_name.endswith(('.jpg', '.jpeg', '.png')):
                    img_path = os.path.join(label_dir, img_name)

                    # 相似度文件路径（如果指定了单独的目录）
                    base_name = os.path.splitext(img_name)[0]
                    txt_name = f"{base_name}.txt"

                    if self.similarity_dir:
                        txt_path = os.path.join(self.similarity_dir, label, txt_name)
                    else:
                        txt_path = os.path.join(label_dir, txt_name)

                    if os.path.exists(txt_path):
                        samples.append((img_path, label_idx, txt_path))

        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label, txt_path = self.samples[idx]

        # 加载图像
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)

        # 加载相似度得分
        sim_scores = self._load_similarity_scores(txt_path)

        # 转换为张量
        sim_tensor = torch.tensor(sim_scores, dtype=torch.float32)

        return image, label, sim_tensor

    def _load_similarity_scores(self, txt_path):
        """加载四种规则的相似度得分"""
        scores = []
        with open(txt_path, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split(':')
                if len(parts) == 2:
                    try:
                        # 假设每行格式为 "类别: 分数1 分数2 分数3 分数4"
                        class_scores = list(map(float, parts[1].strip().split()))
                        scores.append(class_scores)
                    except ValueError:
                        scores.append([0.0, 0.0, 0.0, 0.0])

        if not scores:
            # 如果没有分数，使用均匀分布
            default_score = 1.0 / len(self.labels)
            scores = [[default_score] * 4 for _ in range(len(self.labels))]

        scores = np.array(scores)
        scores = np.clip(scores, 0, None)

        # 如果所有分数都是0，设置为均匀分布
        zero_mask = np.all(scores == 0, axis=1)
        if np.any(zero_mask):
            default_score = 1.0 / len(self.labels)
            scores[zero_mask] = [[default_score] * 4 for _ in range(zero_mask.sum())]

        return scores