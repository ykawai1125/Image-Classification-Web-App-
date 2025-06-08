# model_save.py
import torch
import torchvision.models as models
import torchvision.transforms as transforms

# Load the pre-trained model (Classification: 1000 classes)
model = models.resnet18(pretrained=True)
model.eval()

# save
torch.save(model, "model.pt")
