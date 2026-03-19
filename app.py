from transformers import AutoImageProcessor, AutoModelForImageClassification
from PIL import Image
import torch
import torch.nn.functional as F

# Load model from local folder
model_path = "./dog-breeds-multiclass-image-classification-with-vit"

processor = AutoImageProcessor.from_pretrained(model_path)
model = AutoModelForImageClassification.from_pretrained(model_path)

model.eval()

def predict_top3(image_path):
    # Load image
    image = Image.open(image_path).convert("RGB")

    # Preprocess
    inputs = processor(images=image, return_tensors="pt")

    # Inference
    with torch.no_grad():
        outputs = model(**inputs)

    logits = outputs.logits

    # Convert to probabilities
    probs = F.softmax(logits, dim=-1)

    # Get top 3
    top3 = torch.topk(probs, 3)

    results = []
    for i in range(3):
        idx = top3.indices[0][i].item()
        score = top3.values[0][i].item()
        label = model.config.id2label[idx]

        results.append({
            "breed": label,
            "confidence": round(score * 100, 2)
        })

    return results


# 🔥 Test locally
if __name__ == "__main__":
    image_path = "dog.jpg"  # replace with your image
    predictions = predict_top3(image_path)

    print("\nTop 3 Predictions:\n")
    for i, pred in enumerate(predictions, 1):
        print(f"{i}. {pred['breed']} - {pred['confidence']}%")