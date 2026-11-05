# Smart Interactive Road Safety System

A team hackathon project exploring how road images could help identify hazards. This repository preserves the initial **ResNet18 image-classification baseline** recovered from the project files.

## My contribution

I prepared the image dataset, trained and evaluated the model using Python and PyTorch, and presented the project idea.

## What this version includes

- Four image classes: `safe`, `accident`, `fog` and `animal`.
- Image preparation and checks for unreadable files.
- Transfer learning with ImageNet-pretrained ResNet18.
- Validation-based checkpoint selection, a classification report and a confusion matrix.

This is a learning prototype. It classifies individual images; the code here does not implement live camera integration, object detection, emergency alerts or a deployed dashboard. No performance figures are claimed for this recovered version.

## Files

| File | Purpose |
| --- | --- |
| `prepare_dataset.py` | Organise source images into train, validation and test folders |
| `check_dataset.py` | Count images and report unreadable files |
| `train_model.py` | Train ResNet18 and save the best validation checkpoint |
| `test_model.py` | Evaluate the saved checkpoint on the test split |

## Run locally

Use a Python virtual environment and install the dependencies:

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
python -m pip install -r requirements.txt
```

Provide your own appropriately licensed images. Images and trained weights are not included.

The simplest option is to prepare `dataset/train/`, `dataset/val/` and `dataset/test/`, each containing the same four class folders (`accident/`, `animal/`, `fog/`, `safe/`) with images. Keep class names identical across all splits.

Alternatively, `prepare_dataset.py` expects:
- `animals/{train,val,test}/`: animal images, including nested folders.
- `cars/{train,val,test}/`: class folders whose names identify accident or safe/non-accident images.
- `Fog/`: fog images, split 70%/20%/10% by the script.

Update the source paths if your folders use different names, such as `Foggy_Cityscapes`. Use a fresh output directory when preparing a dataset. Multiple car source folders mapped to the same class can overwrite output filenames in this original script; consolidate them before running it.

```bash
# Only if using the source layout described above:
python prepare_dataset.py

python check_dataset.py
python train_model.py
python test_model.py
```

Check that every class and split contains usable images before training. The training script downloads pretrained weights on first use and selects CUDA automatically when available. Its original settings are 5 epochs, batch size 16 and learning rate 0.001. The checkpoint is saved as `smart_road_model.pth` when validation accuracy improves above zero.

## Limitations and next steps

The original scripts are preserved without changes. They were reviewed for this portfolio upload but were not executed or retrained as part of it; dependency versions are not pinned to a verified environment.

Before interpreting evaluation results, check for duplicate images and related scenes across splits, review class balance and verify labels. A random image split can overestimate generalisation when related frames appear in multiple splits. This prototype has not been validated for safety-critical use.

Next steps include reproducible dependency versions, stronger dataset validation, explicit class-mapping checks and evaluation on unseen road scenes.
