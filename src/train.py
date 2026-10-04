import os
import csv
import random

import torch
import torchvision.transforms as T
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from src.dataset import FaceMaskDataset
from src.model import FastRCNN

from configs.config import (
    DATA_FOLDER,
    BATCH_SIZE,
    NUMBER_EPOCHS,
    TRAIN_SPLIT_PERCENTAGE,
    VAL_SPLIT_PERCENTAGE,
    TEST_SPLIT_PERCENTAGE,
    NUM_CLASSES,
    LEARNING_RATE,
    MOMENTUM,
    SEED,
    CONFIDENCE_THRESHOLD,
    IOU_THRESHOLD
)


# ============================================================
# Configuration
# ============================================================

SEED = SEED
CONFIDENCE_THRESHOLD = CONFIDENCE_THRESHOLD
IOU_THRESHOLD = IOU_THRESHOLD

CLASS_NAMES = {
    1: "with_mask",
    2: "without_mask",
    3: "mask_weared_incorrect"
}

os.makedirs("outputs/models", exist_ok=True)
os.makedirs("outputs/figures", exist_ok=True)


# ============================================================
# Reproducibility
# ============================================================

random.seed(SEED)
torch.manual_seed(SEED)


# ============================================================
# Dataset split
# ============================================================

train_split_percentage = TRAIN_SPLIT_PERCENTAGE
val_split_percentage = VAL_SPLIT_PERCENTAGE
test_split_percentage = TEST_SPLIT_PERCENTAGE

size_of_the_dataset = 853

indexes = list(range(size_of_the_dataset))
random.shuffle(indexes)

train_indexes = indexes[
    :int(train_split_percentage * len(indexes))
]

val_indexes = indexes[
    int(train_split_percentage * len(indexes)):
    int((train_split_percentage + val_split_percentage) * len(indexes))
]

test_indexes = indexes[
    int((train_split_percentage + val_split_percentage) * len(indexes)):
]


print(
    f"Effective train split = "
    f"{len(train_indexes) / len(indexes) * 100:.4f}%"
)

print(
    f"Effective val split = "
    f"{len(val_indexes) / len(indexes) * 100:.4f}%"
)

print(
    f"Effective test split = "
    f"{len(test_indexes) / len(indexes) * 100:.4f}%"
)


# ============================================================
# DataLoader utilities
# ============================================================

def collate_fn(batch):
    return tuple(zip(*batch))


def get_transform(train):
    transforms = []

    transforms.append(T.ToTensor())

    # TODO: potentially add data augmentation

    return T.Compose(transforms)


# ============================================================
# Dataset
# ============================================================

batch_size = BATCH_SIZE

print("Loading training set")

train_dataset = FaceMaskDataset(
    DATA_FOLDER,
    DATA_FOLDER,
    train_indexes,
    conversion=get_transform(True)
)

print("Loading validation set")

val_dataset = FaceMaskDataset(
    DATA_FOLDER,
    DATA_FOLDER,
    val_indexes,
    conversion=get_transform(False)
)

print("Loading test set")

test_dataset = FaceMaskDataset(
    DATA_FOLDER,
    DATA_FOLDER,
    test_indexes,
    conversion=get_transform(False)
)

print(f"Training samples: {len(train_dataset)}")
print(f"Validation samples: {len(val_dataset)}")
print(f"Test samples: {len(test_dataset)}")


# ============================================================
# DataLoaders
# ============================================================

train_loader = torch.utils.data.DataLoader(
    dataset=train_dataset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=0,
    collate_fn=collate_fn
)

val_loader = torch.utils.data.DataLoader(
    dataset=val_dataset,
    batch_size=batch_size,
    shuffle=False,
    num_workers=0,
    collate_fn=collate_fn
)

test_loader = torch.utils.data.DataLoader(
    dataset=test_dataset,
    batch_size=batch_size,
    shuffle=False,
    num_workers=0,
    collate_fn=collate_fn
)


# ============================================================
# Device
# ============================================================

device = torch.device(
    "mps" if torch.backends.mps.is_available() else "cpu"
)

print(f"Device: {device}")


# ============================================================
# Model
# ============================================================

model = FastRCNN(num_classes=NUM_CLASSES)
model.to(device)


# ============================================================
# Optimizer
# ============================================================

number_epochs = NUMBER_EPOCHS

params = [
    p for p in model.parameters()
    if p.requires_grad
]

optimizer = torch.optim.SGD(
    params,
    lr=LEARNING_RATE,
    momentum=MOMENTUM
)


# ============================================================
# Training
# ============================================================

train_loss = []
val_loss = []

for epoch in range(number_epochs):

    print("\n" + "=" * 60)
    print(f"Starting epoch {epoch + 1}/{number_epochs}")
    print("=" * 60)

    # --------------------------------------------------------
    # Training
    # --------------------------------------------------------

    model.train()

    epoch_train_loss = 0.0

    for images, labels in train_loader:

        images = [
            image.to(device)
            for image in images
        ]

        labels = [
            {k: v.to(device) for k, v in t.items()}
            for t in labels
        ]

        loss_dict = model(images, labels)

        losses = sum(
            loss for loss in loss_dict.values()
        )

        optimizer.zero_grad()

        losses.backward()

        optimizer.step()

        epoch_train_loss += losses.detach().cpu().item()

    epoch_train_loss /= len(train_dataset)

    train_loss.append(epoch_train_loss)

    print(
        f"Training loss: {epoch_train_loss:.4f}"
    )


    # --------------------------------------------------------
    # Validation
    # --------------------------------------------------------

    epoch_val_loss = 0.0

    with torch.no_grad():

        # Faster R-CNN returns losses in train mode
        model.train()

        for images, labels in val_loader:

            images = [
                image.to(device)
                for image in images
            ]

            labels = [
                {k: v.to(device) for k, v in t.items()}
                for t in labels
            ]

            loss_dict = model(images, labels)

            losses = sum(
                loss for loss in loss_dict.values()
            )

            epoch_val_loss += losses.detach().cpu().item()

    epoch_val_loss /= len(val_dataset)

    val_loss.append(epoch_val_loss)

    print(
        f"Validation loss: {epoch_val_loss:.4f}"
    )


# ============================================================
# Save loss curve
# ============================================================

plt.figure(figsize=(8, 5))

epochs = range(1, number_epochs + 1)

plt.plot(
    epochs,
    train_loss,
    marker="o",
    label="Train"
)

plt.plot(
    epochs,
    val_loss,
    marker="o",
    label="Validation"
)

plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training and Validation Loss")
plt.legend()
plt.grid(True)

plt.tight_layout()

loss_curve_path = "outputs/figures/loss_curve.png"

plt.savefig(
    loss_curve_path,
    dpi=300
)

plt.close()

print(
    f"\nLoss curve saved to: {loss_curve_path}"
)


# ============================================================
# Save final model
# ============================================================

model_path = "outputs/models/model_final.pt"

torch.save(
    model.state_dict(),
    model_path
)

print(
    f"Model saved to: {model_path}"
)


# ============================================================
# Test loss
# ============================================================

print("\n" + "=" * 60)
print("Evaluating test loss")
print("=" * 60)

test_loss = 0.0

with torch.no_grad():

    # Faster R-CNN returns losses in train mode
    model.train()

    for images, labels in test_loader:

        images = [
            image.to(device)
            for image in images
        ]

        labels = [
            {k: v.to(device) for k, v in t.items()}
            for t in labels
        ]

        loss_dict = model(images, labels)

        losses = sum(
            loss for loss in loss_dict.values()
        )

        test_loss += losses.detach().cpu().item()

test_loss /= len(test_dataset)

print(
    f"Test loss: {test_loss:.4f}"
)


# ============================================================
# IoU function
# ============================================================

def calculate_iou(box1, box2):

    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])

    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    intersection_width = max(0, x2 - x1)
    intersection_height = max(0, y2 - y1)

    intersection = (
        intersection_width *
        intersection_height
    )

    area1 = (
        max(0, box1[2] - box1[0]) *
        max(0, box1[3] - box1[1])
    )

    area2 = (
        max(0, box2[2] - box2[0]) *
        max(0, box2[3] - box2[1])
    )

    union = area1 + area2 - intersection

    if union == 0:
        return 0.0

    return intersection / union


# ============================================================
# Detection metrics
# ============================================================

metrics = {
    class_id: {
        "TP": 0,
        "FP": 0,
        "FN": 0
    }
    for class_id in CLASS_NAMES
}


# ------------------------------------------------------------
# Store predictions for visualization
# ------------------------------------------------------------

visualization_results = []


# ============================================================
# Test predictions
# ============================================================

print("\n" + "=" * 60)
print("Evaluating detection metrics")
print("=" * 60)

model.eval()

with torch.no_grad():

    for images, targets in test_loader:

        images_device = [
            image.to(device)
            for image in images
        ]

        outputs = model(images_device)

        for image, target, output in zip(
            images,
            targets,
            outputs
        ):

            pred_boxes = output["boxes"].cpu()
            pred_labels = output["labels"].cpu()
            pred_scores = output["scores"].cpu()

            # Confidence filtering
            keep = pred_scores >= CONFIDENCE_THRESHOLD

            pred_boxes = pred_boxes[keep]
            pred_labels = pred_labels[keep]
            pred_scores = pred_scores[keep]

            true_boxes = target["boxes"].cpu()
            true_labels = target["labels"].cpu()

            matched_gt = set()

            # ------------------------------------------------
            # Match predictions with ground truth
            # ------------------------------------------------

            for pred_idx in range(len(pred_boxes)):

                pred_box = pred_boxes[pred_idx].tolist()
                pred_label = pred_labels[pred_idx].item()

                best_iou = 0.0
                best_gt_idx = None

                for gt_idx in range(len(true_boxes)):

                    if gt_idx in matched_gt:
                        continue

                    if true_labels[gt_idx].item() != pred_label:
                        continue

                    gt_box = true_boxes[gt_idx].tolist()

                    iou = calculate_iou(
                        pred_box,
                        gt_box
                    )

                    if iou > best_iou:
                        best_iou = iou
                        best_gt_idx = gt_idx

                if (
                    best_gt_idx is not None
                    and best_iou >= IOU_THRESHOLD
                ):

                    metrics[pred_label]["TP"] += 1

                    matched_gt.add(best_gt_idx)

                else:

                    if pred_label in metrics:
                        metrics[pred_label]["FP"] += 1

            # ------------------------------------------------
            # False negatives
            # ------------------------------------------------

            for gt_idx in range(len(true_boxes)):

                if gt_idx not in matched_gt:

                    gt_label = true_labels[gt_idx].item()

                    if gt_label in metrics:
                        metrics[gt_label]["FN"] += 1

            # ------------------------------------------------
            # Store first few images for visualization
            # ------------------------------------------------

            if len(visualization_results) < 5:

                visualization_results.append(
                    (
                        image.cpu(),
                        true_boxes,
                        true_labels,
                        pred_boxes,
                        pred_labels,
                        pred_scores
                    )
                )


# ============================================================
# Calculate final metrics
# ============================================================

total_tp = 0
total_fp = 0
total_fn = 0

for class_id in CLASS_NAMES:

    tp = metrics[class_id]["TP"]
    fp = metrics[class_id]["FP"]
    fn = metrics[class_id]["FN"]

    total_tp += tp
    total_fp += fp
    total_fn += fn

    precision = (
        tp / (tp + fp)
        if (tp + fp) > 0
        else 0.0
    )

    recall = (
        tp / (tp + fn)
        if (tp + fn) > 0
        else 0.0
    )

    f1 = (
        2 * precision * recall /
        (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )

    metrics[class_id]["Precision"] = precision
    metrics[class_id]["Recall"] = recall
    metrics[class_id]["F1"] = f1


# ============================================================
# Overall metrics
# ============================================================

overall_precision = (
    total_tp / (total_tp + total_fp)
    if (total_tp + total_fp) > 0
    else 0.0
)

overall_recall = (
    total_tp / (total_tp + total_fn)
    if (total_tp + total_fn) > 0
    else 0.0
)

overall_f1 = (
    2 * overall_precision * overall_recall /
    (overall_precision + overall_recall)
    if (overall_precision + overall_recall) > 0
    else 0.0
)


# ============================================================
# Print metrics
# ============================================================

print("\n" + "=" * 60)
print("FINAL TEST METRICS")
print("=" * 60)

print(
    f"Test loss: {test_loss:.4f}"
)

print(
    f"Overall Precision: {overall_precision:.4f}"
)

print(
    f"Overall Recall:    {overall_recall:.4f}"
)

print(
    f"Overall F1-score:  {overall_f1:.4f}"
)

print("\nPer-class metrics:")

for class_id, class_name in CLASS_NAMES.items():

    print(f"\n{class_name}")

    print(
        f"  TP: {metrics[class_id]['TP']}"
    )

    print(
        f"  FP: {metrics[class_id]['FP']}"
    )

    print(
        f"  FN: {metrics[class_id]['FN']}"
    )

    print(
        f"  Precision: "
        f"{metrics[class_id]['Precision']:.4f}"
    )

    print(
        f"  Recall: "
        f"{metrics[class_id]['Recall']:.4f}"
    )

    print(
        f"  F1: "
        f"{metrics[class_id]['F1']:.4f}"
    )


# ============================================================
# Save metrics CSV
# ============================================================

metrics_path = "outputs/metrics.csv"

with open(
    metrics_path,
    "w",
    newline=""
) as csv_file:

    writer = csv.writer(csv_file)

    writer.writerow([
        "Class",
        "TP",
        "FP",
        "FN",
        "Precision",
        "Recall",
        "F1"
    ])

    for class_id, class_name in CLASS_NAMES.items():

        writer.writerow([
            class_name,
            metrics[class_id]["TP"],
            metrics[class_id]["FP"],
            metrics[class_id]["FN"],
            metrics[class_id]["Precision"],
            metrics[class_id]["Recall"],
            metrics[class_id]["F1"]
        ])

    writer.writerow([
        "Overall",
        total_tp,
        total_fp,
        total_fn,
        overall_precision,
        overall_recall,
        overall_f1
    ])

    writer.writerow([
        "Test Loss",
        "",
        "",
        "",
        test_loss,
        "",
        ""
    ])

print(
    f"\nMetrics saved to: {metrics_path}"
)


# ============================================================
# Visualize predictions
# ============================================================

print("\nSaving prediction visualizations...")

for idx, (
    image,
    true_boxes,
    true_labels,
    pred_boxes,
    pred_labels,
    pred_scores
) in enumerate(visualization_results):

    image_np = image.permute(
        1, 2, 0
    ).numpy()

    fig, ax = plt.subplots(
        figsize=(8, 6)
    )

    ax.imshow(image_np)

    # --------------------------------------------------------
    # Ground truth - solid boxes
    # --------------------------------------------------------

    for box, label in zip(
        true_boxes,
        true_labels
    ):

        xmin, ymin, xmax, ymax = box.tolist()

        rect = patches.Rectangle(
            (xmin, ymin),
            xmax - xmin,
            ymax - ymin,
            linewidth=2,
            edgecolor="green",
            facecolor="none"
        )

        ax.add_patch(rect)

        ax.text(
            xmin,
            ymin,
            f"GT: {CLASS_NAMES.get(label.item(), 'unknown')}",
            fontsize=8,
            backgroundcolor="white"
        )

    # --------------------------------------------------------
    # Predictions - dashed boxes
    # --------------------------------------------------------

    for box, label, score in zip(
        pred_boxes,
        pred_labels,
        pred_scores
    ):

        xmin, ymin, xmax, ymax = box.tolist()

        rect = patches.Rectangle(
            (xmin, ymin),
            xmax - xmin,
            ymax - ymin,
            linewidth=2,
            edgecolor="red",
            facecolor="none",
            linestyle="--"
        )

        ax.add_patch(rect)

        ax.text(
            xmin,
            ymax,
            f"Pred: {CLASS_NAMES.get(label.item(), 'unknown')} "
            f"{score:.2f}",
            fontsize=8,
            backgroundcolor="white"
        )

    ax.set_title(
        "Ground Truth (green) vs Predictions (red)"
    )

    ax.axis("off")

    plt.tight_layout()

    prediction_path = (
        f"outputs/figures/prediction_{idx}.png"
    )

    plt.savefig(
        prediction_path,
        dpi=200,
        bbox_inches="tight"
    )

    plt.close()

    print(
        f"Saved: {prediction_path}"
    )


# ============================================================
# Final summary
# ============================================================

print("\n" + "=" * 60)
print("EXPERIMENT COMPLETED")
print("=" * 60)

print(
    f"Final training loss: {train_loss[-1]:.4f}"
)

print(
    f"Final validation loss: {val_loss[-1]:.4f}"
)

print(
    f"Test loss: {test_loss:.4f}"
)

print(
    f"Precision: {overall_precision:.4f}"
)

print(
    f"Recall: {overall_recall:.4f}"
)

print(
    f"F1-score: {overall_f1:.4f}"
)

print("\nOutputs:")
print("- outputs/models/model_final.pt")
print("- outputs/figures/loss_curve.png")
print("- outputs/figures/prediction_*.png")
print("- outputs/metrics.csv")