from flask import Flask, request, jsonify, render_template
import torch
import torchvision.transforms as transforms
import torchvision.models as models
from PIL import Image
import urllib.request

app = Flask(__name__)

# PyTorch 2.6+ 対応: 安全なグローバル定義として ResNet を許可
import torch.serialization
torch.serialization.add_safe_globals([models.resnet.ResNet])

# モデル読み込み
model = torch.load("model.pt", map_location=torch.device('cpu'), weights_only=False)
model.eval()

# ImageNetのラベル（クラス番号→名前）を読み込む
LABELS_URL = "https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt"
with urllib.request.urlopen(LABELS_URL) as f:
    imagenet_classes = [line.strip().decode("utf-8") for line in f]

# 入力画像の前処理
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor()
])

# トップページ
@app.route("/")
def index():
    return render_template("index.html")

# 画像アップロードして分類
@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    file = request.files["image"]
    img = Image.open(file).convert("RGB")
    img_tensor = transform(img).unsqueeze(0)

    # 推論実行
    with torch.no_grad():
        outputs = model(img_tensor)
        _, predicted = torch.max(outputs, 1)
        class_idx = predicted.item()
        class_name = imagenet_classes[class_idx]

    return jsonify({"result": f"Predicted class: {class_name} (index: {class_idx})"})

if __name__ == "__main__":
    app.run(debug=True)
