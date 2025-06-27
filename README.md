# Retinal Vessel Segmentation using U-Net Variants

In this project, I worked on segmenting blood vessels from retinal fundus images using deep learning. I implemented and compared four U-Net-based architectures: **Vanilla U-Net**, **U-Net++**, **U-Net³⁺**, and **Gated U-Net**, with the aim of identifying which variant performs best for medical image segmentation, particularly for detecting fine vessel structures relevant to diseases like diabetic retinopathy.

---

## Objective

The goal was to build a robust semantic segmentation pipeline and evaluate how different U-Net variants perform in accurately detecting blood vessels—both major vessels and fine capillaries—in retinal images.

---

## What I Did

- Studied the architecture of U-Net and its key variants.
- Developed a complete image segmentation pipeline using **TensorFlow**, **Keras**, **OpenCV**, and **NumPy**.
- Applied preprocessing techniques such as:
  - CLAHE-based contrast enhancement
  - Patch extraction and augmentation
  - Intensity normalization
  - Grayscale mask handling

- Trained four U-Net variants on the **DRIVE dataset**, which consists of 40 high-resolution fundus images (20 for training and 20 for testing), each annotated by medical experts.

---

## Training Details

- Each model was trained for **250 epochs** on an **NVIDIA A100 GPU**.
- Used a combined **Binary Crossentropy + Dice Loss** function.
- Training monitored via validation accuracy and loss curves to evaluate generalization and detect overfitting.

---

## Results

| Model         | Validation Dice | IoU   | Observations                                         |
|---------------|------------------|-------|------------------------------------------------------|
| Gated U-Net   | ~0.92            | ~0.84 | Best overall; clean vessel maps, low noise           |
| U-Net++       | ~0.90            | ~0.83 | Strong on capillaries; stable and consistent         |
| U-Net³⁺       | ~0.88            | ~0.81 | Mid-training instability; good final performance     |
| Vanilla U-Net | ~0.75            | ~0.68 | Overfit early; poor generalization on test images    |

---

## Key Observations

- **Gated U-Net** and **U-Net++** performed the best overall, showing high vessel continuity and low false positives.
- **U-Net++** was particularly effective at capturing thin capillaries with high recall.
- **U-Net³⁺** showed some training instability but ultimately recovered to a good level of accuracy.
- **Vanilla U-Net** had fast convergence but overfit severely, showing weak test-time performance.

---

## Future Work

As a next step, I plan to explore a **Neural Architecture Search (NAS)** framework using **Differential Evolution (DE)** to automate the optimization of U-Net architectures and hyperparameters for medical image segmentation tasks. This could improve adaptability across different datasets and clinical applications.

---

## Tools and Technologies

- Python
- TensorFlow / Keras
- NumPy
- OpenCV
- Matplotlib
- Google Colab (NVIDIA A100 GPU)

---

