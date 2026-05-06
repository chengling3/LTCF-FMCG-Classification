import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import googlenet


class SimilarityFusionNet(nn.Module):
    def __init__(self, num_classes, pretrained=True, alpha_init=0.2, max_alpha=0.4):
        super().__init__()

        if pretrained:
            weights = GoogLeNet_Weights.IMAGENET1K_V1
        else:
            weights = None

        base_model = googlenet(
            weights=weights,
            aux_logits=True
        )

        base_model.aux_logits = False
        base_model.aux1 = None
        base_model.aux2 = None

        self.img_backbone = nn.Sequential(
            *list(base_model.children())[:-1],
            nn.Flatten()
        )

        self.num_classes = num_classes        
        self.img_proj = nn.Linear(1024, num_classes)
        
        self.sim_adjust = nn.Sequential(
            nn.Linear(num_classes, num_classes),
            nn.Tanh()  
        )

        self.alpha_raw = nn.Parameter(torch.tensor(alpha_init))
        self.max_alpha = max_alpha
       
        self.img_bn = nn.BatchNorm1d(num_classes)
        self.sim_bn = nn.BatchNorm1d(num_classes)
        self.rule_weights = nn.Parameter(torch.ones(4))

    @property
    def alpha(self):
        return torch.sigmoid(self.alpha_raw) * self.max_alpha

    @property
    def normalized_rule_weights(self):
        return F.relu(self.rule_weights)

    def forward(self, img, sim_scores):
        
        img_features = self.img_backbone(img)
        img_logits = self.img_proj(img_features)
        img_logits = self.img_bn(img_logits)
        img_probs = F.softmax(img_logits, dim=-1)
 
        weights = self.normalized_rule_weights
        weighted_sim_scores = torch.sum(sim_scores * weights.view(1, 1, -1), dim=2)

        sim_scores = self.sim_bn(weighted_sim_scores)
      
        adjust_factors = self.sim_adjust(sim_scores)
       
        eps = 1e-6
        adjusted_probs = img_probs * (1 + self.alpha * adjust_factors + eps)

       
        final_probs = adjusted_probs / torch.sum(adjusted_probs, dim=-1, keepdim=True)
       
        if torch.isnan(final_probs).any():
            print("NaN detected in final_probs!")
        
            return img_probs

        return final_probs