# model_save.py
import torch
import torchvision.models as models
import torchvision.transforms as transforms

# 事前学習済みモデルをロード（分類: 1000クラス）
model = models.resnet18(pretrained=True)
model.eval()

# 保存
torch.save(model, "model.pt")
