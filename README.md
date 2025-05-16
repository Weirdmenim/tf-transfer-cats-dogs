# 🐱🐶 Cats v Dogs Transfer Learning

An end‑to‑end pipeline that uses MobileNetV2 transfer learning to classify images of cats and dogs. This project includes data preparation, augmentation, two‑phase training (feature‑head training + fine‑tuning), diagnostic tooling to debug training/validation behavior, and a Gradio app deployed on Hugging Face Spaces.

---

## 📊 Business Context

Automated pet image classification has applications in:

* **Veterinary clinics**: Streamline patient intake by automatically identifying species.
* **Animal shelters**: Tag and sort incoming animals.
* **Pet retail & e‑commerce**: Recommend products based on the pet type in user‑uploaded images.

By reducing manual labeling, businesses save time, cut errors, and improve user experience.

---

## 🚀 Features

* **Automated Data Pipeline**
  Downloads and extracts the "Cats vs. Dogs" dataset, verifies folder structure and class balance.

* **Data Augmentation**
  Real‑time image transformations (flip, rotate, zoom, brightness, contrast) to improve generalization.

* **Two‑Phase Training**

  1. **Phase 1**: Train custom head on frozen MobileNetV2 base
  2. **Phase 2**: Unfreeze top layers of MobileNetV2 and fine‑tune with lower learning rate

* **Regularization**
  L2 weight penalties, batch normalization, and dropout to prevent overfitting.

* **Diagnostics & Debugging**

  * Training vs. inference mode comparisons
  * Component‑wise ablations (no augmentation, reduced regularization)
  * Visualization of augmentation effects
  * Dataset statistics and sanity checks

* **Evaluation & Reporting**

  * Accuracy, precision, recall, F1‑score
  * Confusion matrix
  * Classification report
  * Training/validation curves

* **Interactive Demo**
  Gradio web app for live inference, deployed on Hugging Face Spaces.

* **Testing & CI**
  Pytest suite covering data loading, model build, training utilities, and the Gradio predictor.

---

## 📂 Repository Structure

```
cats-vs-dogs-transfer/
├── app.py                        # Gradio inference interface
├── main.py                       # Training pipeline with diagnostics
├── utils.py                      # Plotting and metrics utilities
├── debug_training_validation_gap.py   # Extended diagnostic script
├── models/
│   └── cats_dogs_mobilenet_v2_debug.keras  # SavedModel for deployment
├── plots/
│   ├── training_accuracy.png
│   ├── training_loss.png
│   ├── confusion_matrix.png
│   ├── classification_report.txt
│   └── augmentation_samples.png
├── requirements.txt
├── test_main.py                  # Pytest suite
├── .github/
│   └── workflows/ci.yml          # GitHub Actions CI pipeline
└── README.md                     # This file
```

---

## 🔧 Installation & Local Run

1. **Clone the repo**

   ```bash
   git clone https://github.com/weirdmenim/cats-vs-dogs-transfer.git
   cd cats-vs-dogs-transfer
   ```

2. **Create & activate a virtual environment**

   ```bash
   python3 -m venv venv
   source venv/bin/activate     # macOS/Linux
   venv\Scripts\activate        # Windows
   ```

3. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

4. **Train & Debug**

   ```bash
   python main.py
   # For extended diagnostics:
   python debug_training_validation_gap.py
   ```

5. **Run Tests**

   ```bash
   pytest
   ```

---

## 📊 Performance & Metrics

After two‑phase training (15 epochs):

| Metric         | Value     |
| -------------- | --------- |
| **Accuracy**   | 97.0%     |
| **Precision**  | 97%-98%   |
| **Recall**     | 97%-98%   |
| **F1‑Score**   | 97%-98%   |
| **Error Rate** | 3.0%      |

```text
              precision    recall  f1-score   support
         Cat       0.97      0.98      0.98       500
         Dog       0.98      0.97      0.97       500
    accuracy                           0.97      1000
   macro avg       0.98      0.97      0.97      1000
weighted avg       0.98      0.97      0.97      1000
```

### Confusion Matrix

![Confusion Matrix](plots/confusion_matrix.png)

The confusion matrix shows strong performance with:
- 492 correctly classified cats (true negatives)
- 483 correctly classified dogs (true positives)
- Only 8 cats misclassified as dogs (false positives)
- Only 17 dogs misclassified as cats (false negatives)

### Training Progress

![Training Accuracy](plots/training_accuracy.png)
![Training Loss](plots/training_loss.png)

The training curves show the expected pattern of a well-regularized model:
- Training accuracy remains around 55% during training while validation accuracy quickly rises to over 90%
- Training loss decreases slowly while validation loss drops rapidly
- The final model achieves over 97% validation accuracy

### Data Augmentation

![Augmentation Samples](plots/augmentation_samples.png)

The augmentation visualization demonstrates how the model sees varied versions of the same images during training, helping it generalize better to new data.

---

## 🧠 Understanding Training vs. Validation Accuracy Gap

### Observed Behavior

When running this model, you'll notice that during training, the reported training accuracy (~58%) is significantly lower than the validation accuracy (~98%). However, after training completes, both training and validation accuracies are high (>96%).

### Why This Happens

This behavior is intentional and a sign of a healthy, well-regularized model. It occurs due to several factors:

1. **Dropout Effects**: During training, dropout randomly deactivates neurons, making predictions less accurate. During inference/validation, dropout is disabled, resulting in more accurate predictions.

2. **Data Augmentation**: We apply significant data augmentation (flips, rotations, zooms, brightness/contrast changes) to training data but not to validation data. This intentionally makes the training task more difficult.

3. **L2 Regularization**: The model uses L2 regularization which penalizes large weights during training, affecting the model's ability to fit the training data perfectly.

### This Is Actually Good!

The behavior indicates that:

- The regularization techniques are working as intended, preventing overfitting
- The model generalizes well to unseen data (high validation accuracy)
- Despite the added difficulty during training, the model learns robust features

The final evaluation, which shows high accuracy on both training and validation sets in inference mode, reflects the true performance of the model.

### Evidence from Diagnostics

```
INFO: Training sample 0 (label=0):
INFO: Training mode: 0.4955
INFO: Inference mode: 0.0558
INFO: Difference: 0.4397

INFO: Final Evaluation:
INFO: Training: Loss=0.1211, Accuracy=0.9690
INFO: Validation: Loss=0.1096, Accuracy=0.9740
```

When evaluated after training in inference mode (dropout disabled), training accuracy (96.9%) is very close to validation accuracy (97.4%), confirming that the gap observed during training is due to regularization techniques rather than model issues.

---

## 🛠 Gradio Demo & Deployment

### Run Locally

```bash
python app.py
# Opens http://localhost:7860 in your browser
```

### Hugging Face Spaces

* **Repo name**: `cats-vs-dogs-transfer`
* **SDK**: Gradio
* **Link**: [https://huggingface.co/spaces/Weirdmenim/cats-vs-dogs-transfer](https://huggingface.co/spaces/Weirdmenim/cats-vs-dogs-transfer)

---

## 🔄 Continuous Integration

GitHub Actions CI file (`.github/workflows/ci.yml`):

```yaml
name: CI

on: [push, pull_request]

jobs:
  build-test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: Setup Python
        uses: actions/setup-python@v4
        with: python-version: '3.9'
      - name: Install dependencies
        run: pip install -r requirements.txt
      - name: Run tests
        run: pytest --maxfail=1 --disable-warnings -q
```

---

## 📄 License

MIT License © 2025 Nsikak Menim

---

## 🤛 Author & Contributions

**Nsikak Menim** – Data Scientist | Engineer | AI Strategist

* Conceptualized & implemented the full pipeline
* Designed diagnostic tools to debug training dynamics
* Deployed an interactive demo for stakeholders

Thank you for reading! Feel free to file issues or contribute via pull requests.