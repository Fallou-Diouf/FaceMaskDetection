# FaceMaskDetection

Face mask detection using Faster R-CNN and transfer learning.

A Computer Vision project for detecting and classifying face-mask usage in images using **Faster R-CNN** with a **ResNet-50 + Feature Pyramid Network (FPN)** backbone.

The model detects faces and classifies them into three categories:

- `with_mask`
- `without_mask`
- `mask_weared_incorrect`

The project includes the complete pipeline, from XML annotation analysis and dataset preparation to model training, evaluation, and visual prediction.

---

## Project Overview

The goal of this project is to build an object detection system that can automatically identify people wearing masks and determine whether the mask is correctly worn.

The complete pipeline is:

```text
Images + XML Annotations
          │
          ▼
    Dataset Preparation
          │
          ▼
   Faster R-CNN Training
          │
          ▼
       Validation
          │
          ▼
      Test Evaluation
          │
          ▼
   Prediction & Visualization
```

---

## Classes

The model detects three classes:

| Class | Description |
|---|---|
| `with_mask` | Person correctly wearing a mask |
| `without_mask` | Person not wearing a mask |
| `mask_weared_incorrect` | Person wearing a mask incorrectly |

The model also uses a fourth class internally:

```text
background
```

Therefore, the final detection model predicts four categories:

```text
0 → background
1 → with_mask
2 → without_mask
3 → mask_weared_incorrect
```

---

## Dataset

The dataset contains:

- **853 images**
- **853 XML annotation files**
- **4072 annotated objects**

The annotations use a Pascal VOC-style XML format.

Each object contains:

- class name
- bounding box

The bounding box is represented as:

```text
[xmin, ymin, xmax, ymax]
```

For example:

```text
[79, 105, 109, 142]
```

The XML structure contains the objects present in each image, their class names, and their bounding boxes.

### Class Distribution

| Class | Number of objects | Percentage |
|---|---:|---:|
| `with_mask` | 3232 | 79.4% |
| `without_mask` | 717 | 17.6% |
| `mask_weared_incorrect` | 123 | 3.0% |
| **Total** | **4072** | **100%** |

The dataset is strongly imbalanced.

The `with_mask` class represents most of the annotated objects, while `mask_weared_incorrect` represents only about 3% of the dataset.

This imbalance is an important limitation because the model has much fewer examples from the minority class to learn from.

---

## Dataset Split

The 853 images were divided into three subsets:

| Split | Number of images | Percentage |
|---|---:|---:|
| Training | 511 | ~60% |
| Validation | 171 | ~20% |
| Test | 171 | ~20% |
| **Total** | **853** | **100%** |

A fixed random seed was used to make the experiment reproducible.

---

## Model

The project uses **Faster R-CNN**, a two-stage object detection model.

The architecture uses:

- Faster R-CNN
- ResNet-50 backbone
- Feature Pyramid Network (FPN)
- COCO-pretrained weights
- A custom classification head adapted to the three mask classes

The original classification head was replaced to predict:

```text
background
with_mask
without_mask
mask_weared_incorrect
```

---

## Training

The model was trained using **PyTorch** and **Torchvision**.

### Main Hyperparameters

| Parameter | Value |
|---|---:|
| Model | Faster R-CNN |
| Backbone | ResNet-50 + FPN |
| Pretraining | COCO |
| Optimizer | SGD |
| Learning rate | 0.001 |
| Momentum | 0.01 |
| Batch size | 2 |
| Number of epochs | 5 |
| Device | Apple MPS |

The experiment uses **transfer learning**, starting from a model pretrained on the COCO dataset and adapting its classification head to the face-mask detection task.

---

## Training Results

The training and validation losses decreased during the five epochs.

| Epoch | Training Loss | Validation Loss |
|---:|---:|---:|
| 1 | 0.3199 | 0.3017 |
| 2 | 0.2424 | 0.2485 |
| 3 | 0.2033 | 0.2316 |
| 4 | 0.1873 | 0.2138 |
| 5 | 0.1761 | 0.2072 |

The final training loss was:

```text
0.1761
```

The final validation loss was:

```text
0.2072
```

The loss curve is available at:

```text
outputs/figures/loss_curve.png
```

Both training and validation losses decreased during the five epochs. No increase in validation loss was observed during this experiment.

---

## Test Results

The final model was evaluated on the **171 test images**.

The test loss was:

```text
0.1860
```

Predictions were evaluated using:

- confidence threshold: **0.5**
- IoU threshold: **0.5**

The model obtained the following overall results:

| Metric | Score |
|---|---:|
| Precision | 0.7894 |
| Recall | 0.7317 |
| F1-score | 0.7595 |

### Per-Class Results

| Class | Precision | Recall | F1-score |
|---|---:|---:|---:|
| `with_mask` | 0.8013 | 0.8386 | 0.8195 |
| `without_mask` | 0.6824 | 0.3671 | 0.4774 |
| `mask_weared_incorrect` | 0.0000 | 0.0000 | 0.0000 |

The `with_mask` class has the best performance, with an F1-score of **0.8195**.

The `without_mask` class has a lower recall of **0.3671**, meaning that several objects from this class were missed or classified incorrectly.

The model did not correctly detect any `mask_weared_incorrect` object during this evaluation.

This result is consistent with the strong class imbalance and the small number of examples available for this class.

> **Note:** These results use a custom IoU-based matching procedure with a confidence threshold of 0.5 and an IoU threshold of 0.5. They are not COCO mAP scores.

---

## Prediction

The project includes a prediction script that can be used to run the trained model on an individual image.

Example:

```bash
python -m src.predict \
    --image Data/images/maksssksksss0.png
```

The script:

1. loads the trained Faster R-CNN model;
2. loads an input image;
3. performs object detection;
4. filters predictions using a confidence threshold;
5. draws bounding boxes;
6. displays the predicted class;
7. displays the confidence score;
8. saves the prediction image.

The default output is:

```text
outputs/predictions/prediction.png
```

### Confidence Threshold

The default confidence threshold is:

```text
0.5
```

It can be changed using:

```bash
python -m src.predict \
    --image Data/images/maksssksksss0.png \
    --threshold 0.7
```

### Example Prediction

For one test image, the model produced:

```text
with_mask       confidence = 0.991
with_mask       confidence = 0.846
without_mask    confidence = 0.574
```

The model correctly localized the three faces, but one `without_mask` face was classified as `with_mask`.

This example shows an important aspect of object detection:

```text
Correct localization
        does not always mean
Correct classification
```

---

## Prediction Visualization

The prediction visualization uses different colors for the different classes:

| Color | Class |
|---|---|
| Green | `with_mask` |
| Red | `without_mask` |
| Orange | `mask_weared_incorrect` |

Prediction examples are stored in:

```text
outputs/figures/
```

The main prediction generated by `predict.py` is stored in:

```text
outputs/predictions/prediction.png
```

---

## Project Structure

```text
FaceMaskDetection/
│
├── configs/
│   └── config.py
│
├── notebooks/
│   └── ...
│
├── rapport/
│   └── rapport.tex
│
├── src/
│   ├── dataset.py
│   ├── evaluate.py
│   ├── model.py
│   ├── predict.py
│   └── train.py
│
├── outputs/
│   ├── figures/
│   │   ├── loss_curve.png
│   │   ├── prediction_0.png
│   │   ├── prediction_1.png
│   │   ├── prediction_2.png
│   │   ├── prediction_3.png
│   │   └── prediction_4.png
│   │
│   ├── predictions/
│   │   └── prediction.png
│   │
│   └── metrics.csv
│
├── requirements.txt
├── .gitignore
└── README.md
```

The dataset and trained model weights are not included in the Git repository.

---

## Installation

Clone the repository:

```bash
git clone https://github.com/Fallou-Diouf/FaceMaskDetection.git
cd FaceMaskDetection
```

Create a virtual environment:

```bash
python3 -m venv .venv
```

Activate the virtual environment.

### macOS / Linux

```bash
source .venv/bin/activate
```

### Windows

```bash
.venv\Scripts\activate
```

Install the required packages:

```bash
pip install -r requirements.txt
```

---

## Dataset Setup

The dataset is not included in the repository.

Place the dataset in the project directory with the following structure:

```text
Data/
├── images/
│   ├── maksssksksss0.png
│   ├── maksssksksss1.png
│   └── ...
│
└── annotations/
    ├── maksssksksss0.xml
    ├── maksssksksss1.xml
    └── ...
```

Each image should have a corresponding XML annotation file.

---

## Training the Model

From the project root, run:

```bash
python -m src.train
```

The training script:

1. loads the dataset;
2. creates the training, validation, and test splits;
3. creates the Faster R-CNN model;
4. trains the model;
5. calculates training and validation losses;
6. evaluates the final model;
7. saves the metrics;
8. generates prediction visualizations.

The generated results are stored in:

```text
outputs/
```

---

## Evaluation

The evaluation uses:

```text
Confidence threshold = 0.5
IoU threshold        = 0.5
```

A predicted bounding box is matched with a ground-truth bounding box when their IoU is at least 0.5.

The project reports:

- Precision
- Recall
- F1-score

for the complete test set and for each class.

The results are saved in:

```text
outputs/metrics.csv
```

---

## Report

A technical report is included in:

```text
rapport/rapport.tex
```

The report describes:

- XML annotation analysis;
- dataset preparation;
- dataset split;
- class distribution;
- Faster R-CNN architecture;
- training configuration;
- quantitative evaluation;
- qualitative analysis;
---

## Technologies

The project was developed using:

- **Python**
- **PyTorch**
- **Torchvision**
- **PIL**
- **NumPy**
- **Pandas**
- **Matplotlib**

The model was trained using the **Apple MPS backend** on a Mac.

---

## Key Results

The main results of the experiment are:

```text
Test Loss       : 0.1860
Precision       : 0.7894
Recall          : 0.7317
F1-score        : 0.7595
```


---


## Author

**Fallou Diouf**


Computer Vision · Deep Learning · Machine Learning

---

## Repository

GitHub:

https://github.com/Fallou-Diouf/FaceMaskDetection
