import os
import torch
import json
from torch.utils.data import Dataset
from PIL import Image
import numpy as np


class BatchDataset(Dataset):
    def __init__(self, root_dir, label_file, similarity_dir=None, transform=None, is_json=True):
        self.root_dir = root_dir
        self.similarity_dir = similarity_dir or root_dir
        self.transform = transform
        self.is_json = is_json

        self.labels = self._load_labels(label_file)
        self.samples = self._make_dataset()

    def _load_labels(self, label_file):
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

        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)

        sim_scores = self._load_similarity_scores(txt_path)
        sim_tensor = torch.tensor(sim_scores, dtype=torch.float32)

        return image, label, sim_tensor

    def _load_similarity_scores(self, txt_path):
        scores = []
        with open(txt_path, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split(':')
                if len(parts) == 2:
                    try:
                        class_scores = list(map(float, parts[1].strip().split()))
                        scores.append(class_scores)
                    except ValueError:
                        scores.append([0.0, 0.0, 0.0, 0.0])

        if not scores:
            default_score = 1.0 / len(self.labels)
            scores = [[default_score] * 4 for _ in range(len(self.labels))]

        scores = np.array(scores)
        scores = np.clip(scores, 0, None)

        zero_mask = np.all(scores == 0, axis=1)
        if np.any(zero_mask):
            default_score = 1.0 / len(self.labels)
            scores[zero_mask] = [[default_score] * 4 for _ in range(zero_mask.sum())]

        return scores