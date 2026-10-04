import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

from configs.config import (
    NUM_CLASSES
)


def FastRCNN(num_classes=NUM_CLASSES):
    # Load Faster R-CNN pre-trained on COCO
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(
        pretrained=True
    )

    # Get the number of input features of the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features

    # Replace the classifier with one adapted to our dataset
    model.roi_heads.box_predictor = FastRCNNPredictor(
        in_features,
        num_classes
    )

    return model