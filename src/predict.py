import argparse
import os

import torch
import torchvision
import torchvision.transforms as T

from PIL import Image, ImageDraw, ImageFont
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor


# ============================================================
# Configuration
# ============================================================

CLASSES = {
    1: "with_mask",
    2: "without_mask",
    3: "mask_weared_incorrect",
}

NUM_CLASSES = 4

MODEL_PATH = "outputs/models/model_final.pt"

DEFAULT_CONFIDENCE_THRESHOLD = 0.5


# Couleurs utilisées pour les différentes classes
CLASS_COLORS = {
    1: (46, 204, 113),    # green
    2: (231, 76, 60),     # red
    3: (243, 156, 18),    # orange
}


# ============================================================
# Chargement du modèle
# ============================================================

def load_model(model_path, device):

    print("Loading model...")

    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(
        weights=None
    )

    in_features = model.roi_heads.box_predictor.cls_score.in_features

    model.roi_heads.box_predictor = FastRCNNPredictor(
        in_features,
        NUM_CLASSES
    )

    checkpoint = torch.load(
        model_path,
        map_location=device,
        weights_only=False
    )

    if "model_state_dict" in checkpoint:
        model.load_state_dict(
            checkpoint["model_state_dict"]
        )
    else:
        model.load_state_dict(checkpoint)

    model.to(device)
    model.eval()

    print(f"Model loaded on {device}")

    return model


# ============================================================
# Prédiction
# ============================================================

def predict(model, image_path, device):

    image = Image.open(image_path).convert("RGB")

    transform = T.ToTensor()

    image_tensor = transform(image).to(device)

    with torch.no_grad():
        prediction = model([image_tensor])[0]

    return image, prediction


# ============================================================
# Police
# ============================================================

def get_font():

    # Police système macOS
    mac_font = "/System/Library/Fonts/Supplemental/Arial.ttf"

    if os.path.exists(mac_font):
        return ImageFont.truetype(mac_font, 16)

    # Fallback si la police n'existe pas
    return ImageFont.load_default()


# ============================================================
# Visualisation des prédictions
# ============================================================

def draw_predictions(
    image,
    prediction,
    threshold=DEFAULT_CONFIDENCE_THRESHOLD  
    ):

    draw = ImageDraw.Draw(image)

    font = get_font()

    boxes = prediction["boxes"].cpu()
    labels = prediction["labels"].cpu()
    scores = prediction["scores"].cpu()

    detections = []

    # Positions déjà utilisées par les labels
    label_positions = []

    for box, label, score in zip(
        boxes,
        labels,
        scores
    ):

        score = float(score)
        label = int(label)

        if score < threshold:
            continue

        class_name = CLASSES.get(
            label,
            "unknown"
        )

        color = CLASS_COLORS.get(
            label,
            (255, 255, 255)
        )

        x1, y1, x2, y2 = map(
            int,
            box.tolist()
        )

        # ====================================================
        # Bounding box
        # ====================================================

        draw.rectangle(
            [x1, y1, x2, y2],
            outline=color,
            width=4
        )

        # ====================================================
        # Label
        # ====================================================

        text = f"{class_name} {score:.2f}"

        text_bbox = draw.textbbox(
            (0, 0),
            text,
            font=font
        )

        text_width = text_bbox[2] - text_bbox[0]
        text_height = text_bbox[3] - text_bbox[1]

        padding = 4

        label_width = text_width + 2 * padding
        label_height = text_height + 2 * padding

        label_x = x1

        # Position initiale : au-dessus de la bounding box
        label_y = y1 - label_height - 2

        # Si le label sort de l'image, on le place dans la box
        if label_y < 0:
            label_y = y1 + 2

        # ====================================================
        # Éviter les labels trop proches
        # ====================================================

        while True:

            collision = False

            current_box = (
                label_x,
                label_y,
                label_x + label_width,
                label_y + label_height
            )

            for previous_box in label_positions:

                px1, py1, px2, py2 = previous_box

                overlap = not (
                    current_box[2] < px1
                    or current_box[0] > px2
                    or current_box[3] < py1
                    or current_box[1] > py2
                )

                if overlap:
                    collision = True
                    break

            if not collision:
                break

            # Décaler verticalement le label
            label_y -= label_height + 3

            # Si on atteint le haut, revenir sous la box
            if label_y < 0:
                label_y = y2 + 3
                break

        label_box = (
            label_x,
            label_y,
            label_x + label_width,
            label_y + label_height
        )

        label_positions.append(label_box)

        # ====================================================
        # Fond du label
        # ====================================================

        draw.rounded_rectangle(
            label_box,
            radius=4,
            fill=color
        )

        # ====================================================
        # Texte
        # ====================================================

        draw.text(
            (
                label_x + padding,
                label_y + padding
            ),
            text,
            fill=(255, 255, 255),
            font=font
        )

        # ====================================================
        # Sauvegarder la détection
        # ====================================================

        detections.append({
            "class": class_name,
            "label": label,
            "confidence": score,
            "box": [x1, y1, x2, y2]
        })

    return image, detections


# ============================================================
# Affichage des résultats
# ============================================================

def print_results(detections):

    print("\n" + "=" * 60)
    print("DETECTION RESULTS")
    print("=" * 60)

    if not detections:
        print("No detections above the confidence threshold.")

    else:

        for i, detection in enumerate(
            detections,
            start=1
        ):

            print(
                f"{i}. "
                f"{detection['class']:25s} "
                f"confidence={detection['confidence']:.3f} "
                f"box={detection['box']}"
            )

    print("-" * 60)

    print(
        f"Number of detections: {len(detections)}"
    )

    print("=" * 60)


# ============================================================
# Main
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description="Face Mask Detection using Faster R-CNN"
    )

    parser.add_argument(
        "--image",
        required=True,
        help="Path to the input image"
    )

    parser.add_argument(
        "--output",
        default="outputs/predictions/prediction.png",
        help="Path to save the prediction image"
    )

    parser.add_argument(
        "--threshold",
        type=float,
        default=DEFAULT_CONFIDENCE_THRESHOLD,
        help="Minimum confidence score"
    )

    args = parser.parse_args()

    # ========================================================
    # Vérification de l'image
    # ========================================================

    if not os.path.exists(args.image):

        raise FileNotFoundError(
            f"Image not found: {args.image}"
        )

    # ========================================================
    # Device
    # ========================================================

    if torch.backends.mps.is_available():

        device = torch.device("mps")

    elif torch.cuda.is_available():

        device = torch.device("cuda")

    else:

        device = torch.device("cpu")

    print(f"Device: {device}")

    # ========================================================
    # Charger le modèle
    # ========================================================

    model = load_model(
        MODEL_PATH,
        device
    )

    # ========================================================
    # Faire la prédiction
    # ========================================================

    image, prediction = predict(
        model,
        args.image,
        device
    )

    # ========================================================
    # Dessiner les prédictions
    # ========================================================

    result, detections = draw_predictions(
        image,
        prediction,
        threshold=args.threshold
    )

    # ========================================================
    # Afficher les résultats
    # ========================================================

    print_results(detections)

    # ========================================================
    # Sauvegarder
    # ========================================================

    output_dir = os.path.dirname(args.output)

    if output_dir:
        os.makedirs(
            output_dir,
            exist_ok=True
        )

    result.save(args.output)

    print(
        f"\nPrediction saved to: {args.output}"
    )


# ============================================================
# Entry point
# ============================================================

if __name__ == "__main__":
    main()