import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import googlenet


class SimilarityFusionNet(nn.Module):
    def __init__(self, num_classes, pretrained=True, alpha_init=0.2, max_alpha=0.4):
        super().__init__()
        # 加载GoogleNet结构并禁用辅助分类器
        base_model = googlenet(weights=None, aux_logits=False)

        # 加载预训练权重，移除分类层
        if pretrained:
            state_dict = torch.load("./yuxunlian.pth", map_location="cpu")
            state_dict = {k: v for k, v in state_dict.items() if not k.startswith('fc.')}
            base_model.load_state_dict(state_dict, strict=False)

        # 提取特征部分并Flatten
        self.img_backbone = nn.Sequential(
            *list(base_model.children())[:-1],  # 去掉fc层
            nn.Flatten()
        )
        self.num_classes = num_classes

        # 将图像特征映射到类别空间
        self.img_proj = nn.Linear(1024, num_classes)

        # 相似度调整网络 - 输出调整系数
        self.sim_adjust = nn.Sequential(
            nn.Linear(num_classes, num_classes),
            nn.Tanh()  # 输出范围[-1, 1]
        )

        # 可学习的融合权重，使用sigmoid约束在[0, max_alpha]
        self.alpha_raw = nn.Parameter(torch.tensor(alpha_init))
        self.max_alpha = max_alpha

        # 添加批量归一化稳定训练
        self.img_bn = nn.BatchNorm1d(num_classes)
        self.sim_bn = nn.BatchNorm1d(num_classes)

        # 修改：四种规则的权重系数，初始化为1，无总和约束，下限为0
        self.rule_weights = nn.Parameter(torch.ones(4))

    @property
    def alpha(self):
        # 使用sigmoid确保alpha在合理范围内
        return torch.sigmoid(self.alpha_raw) * self.max_alpha

    @property
    def normalized_rule_weights(self):
        # 修改：直接返回ReLU处理后的权重，确保非负，无总和约束
        return F.relu(self.rule_weights)

    def forward(self, img, sim_scores):
        # 提取图像特征并映射到类别空间
        img_features = self.img_backbone(img)
        img_logits = self.img_proj(img_features)

        # 应用批量归一化
        img_logits = self.img_bn(img_logits)

        # 图像预测概率
        img_probs = F.softmax(img_logits, dim=-1)

        # 处理相似度分数（假设sim_scores形状为[batch_size, num_classes, 4]）
        # 修改：应用未归一化的规则权重
        weights = self.normalized_rule_weights
        weighted_sim_scores = torch.sum(sim_scores * weights.view(1, 1, -1), dim=2)

        # 相似度分数归一化
        sim_scores = self.sim_bn(weighted_sim_scores)

        # 相似度调整系数
        adjust_factors = self.sim_adjust(sim_scores)

        # 应用调整：高相似度提高，低相似度降低
        # 添加小常数确保数值稳定性
        eps = 1e-6
        adjusted_probs = img_probs * (1 + self.alpha * adjust_factors + eps)

        # 重新归一化，保持概率和为1
        final_probs = adjusted_probs / torch.sum(adjusted_probs, dim=-1, keepdim=True)

        # 检查输出是否有NaN
        if torch.isnan(final_probs).any():
            print("NaN detected in final_probs!")
            # 返回原始图像预测作为回退
            return img_probs

        return final_probs