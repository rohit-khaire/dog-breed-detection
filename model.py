from transformers import AutoImageProcessor, AutoModelForImageClassification
from ultralytics import YOLO
from PIL import Image
import torch
import torch.nn.functional as F
import cv2
import tempfile

# -----------------------------
# Load Models (ONLY ONCE)
# -----------------------------
# model_path = "./dog-breeds-multiclass-image-classification-with-vit"
model_path = "wesleyacheng/dog-breeds-multiclass-image-classification-with-vit"

processor = AutoImageProcessor.from_pretrained(model_path)
classifier = AutoModelForImageClassification.from_pretrained(model_path)
classifier.eval()

# YOLO detector
detector = YOLO("yolov8m.pt")   # or yolov8l.pt
# detector = YOLO("yolov8n.pt")

# -----------------------------
# Classifier Function (Top 3)
# -----------------------------
def predict_top3_pil(image):
    inputs = processor(images=image, return_tensors="pt")

    with torch.no_grad():
        outputs = classifier(**inputs)

    probs = F.softmax(outputs.logits, dim=-1)
    top3 = torch.topk(probs, 3)

    results = []
    for i in range(3):
        idx = top3.indices[0][i].item()
        score = top3.values[0][i].item()
        label = classifier.config.id2label[idx]

        results.append({
            "breed": label.replace("_", " ").title(),
            "confidence": round(score * 100, 2)
        })

    return results


# -----------------------------
# Detect + Classify Multiple Dogs
# -----------------------------
def predict_multiple_dogs(image_path):
    image = cv2.imread(image_path)
    results = detector(image_path, conf=0.25)

    final_results = []

    # -----------------------------
    # Detect Dogs
    # -----------------------------
    for r in results:
        for box in r.boxes:
            cls = int(box.cls[0])
            label = detector.names[cls]

            if label == "dog":
                x1, y1, x2, y2 = map(int, box.xyxy[0])

                h, w, _ = image.shape

                # Dynamic padding
                box_width = x2 - x1
                box_height = y2 - y1

                pad_x = int(0.15 * box_width)
                pad_y = int(0.15 * box_height)

                x1 = max(0, x1 - pad_x)
                y1 = max(0, y1 - pad_y)
                x2 = min(w, x2 + pad_x)
                y2 = min(h, y2 + pad_y)

                crop = image[y1:y2, x1:x2]
                crop_pil = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))

                preds = predict_top3_pil(crop_pil)

                final_results.append({
                    "box": (x1, y1, x2, y2),
                    "predictions": preds,
                    "type": "detected"
                })

    # -----------------------------
    # 🔥 FALLBACK (if no dogs detected)
    # -----------------------------
    if len(final_results) == 0:
        full_image = Image.open(image_path).convert("RGB")
        preds = predict_top3_pil(full_image)

        final_results.append({
            "box": None,
            "predictions": preds,
            "type": "fallback"
        })

    return final_results