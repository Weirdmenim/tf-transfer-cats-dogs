# README.md
```markdown
# Cats vs Dogs Transfer Learning v1.2

This project fine-tunes a pretrained MobileNetV2 on the Cats vs Dogs filtered dataset. It demonstrates:
- Data preparation via automated download and extraction
- Building and freezing a Keras MobileNetV2 feature extractor
- Training, validation, and plotting of accuracy/loss
- Saving and loading models for inference on new images

## Setup
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Usage
```bash
python main.py
```
- Trains for 5 epochs and shows metrics plots.
- Saves model to `cats_dogs_mobilenet_v2.h5`.
- Runs a sample inference and logs the result.

## Testing
```bash
pytest -q
```

## Slide Snippet
```markdown
# Transfer Learning: Cats vs Dogs

**Key Feature:**
– Pretrained MobileNetV2 as frozen feature extractor

**Takeaway:**
Rapidly adapt large vision models with minimal training data.

**Next Steps:**
– Unfreeze top layers for fine-tuning
– Deploy as REST API with FastAPI
```
```

