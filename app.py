from flask import Flask, request, jsonify, render_template
import torch
import torchvision.transforms as transforms
import torchvision.models as models
from PIL import Image
import urllib.request

app = Flask(__name__)

# For PyTorch 2.6+ compatibility: allow ResNet to be safely deserialized
import torch.serialization
torch.serialization.add_safe_globals([models.resnet.ResNet])

# Load the pretrained model from file
model = torch.load("model.pt", map_location=torch.device('cpu'), weights_only=False)
model.eval()  # Set the model to evaluation mode

# Load ImageNet class labels (class index to human-readable label)
LABELS_URL = "https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt"
with urllib.request.urlopen(LABELS_URL) as f:
    imagenet_classes = [line.strip().decode("utf-8") for line in f]

# Preprocessing steps for input image
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor()
])

# Route for the homepage
@app.route("/")
def index():
    return render_template("index.html")

# Route for handling image upload and classification
@app.route("/predict", methods=["POST"])
def predict():
    # Check if an image was uploaded
    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    # Load and convert the image to RGB
    file = request.files["image"]
    img = Image.open(file).convert("RGB")
    
    # Apply transformations and add batch dimension
    img_tensor = transform(img).unsqueeze(0)

    # Run inference
    with torch.no_grad():
        outputs = model(img_tensor)
        _, predicted = torch.max(outputs, 1)
        class_idx = predicted.item()
        class_name = imagenet_classes[class_idx]

    # Return the result as JSON
    return jsonify({"result": f"Predicted class: {class_name} (index: {class_idx})"})

# Run the app (in debug mode)
if __name__ == "__main__":
    app.run(debug=True)
