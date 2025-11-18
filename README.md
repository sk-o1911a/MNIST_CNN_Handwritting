# Handwritten Digit Sequence Recognizer

## Project Information

**Course:** Artfical Intelligence  
**Author:** Nguyen Quoc Khanh  
**Student ID:** 42200211

---

## Overview

The application uses a Convolutional Neural Network (CNN) built with PyTorch, trained on a heavily augmented MNIST dataset, and includes a real-time GUI (built with Tkinter) that allows users to draw multiple digits and get a sequence prediction.
## Key Features

- **Multi-Digit Recognition**: The GUI uses OpenCV to segment multiple, non-overlapping digits from a single canvas, processes them individually, and predicts them as a sequence.
- **Rich Data Augmentation**:  Uses ElasticTransform, RandomAffine, GaussianBlur, and RandomErasing to create a highly robust model that generalizes well to "messy" real-world drawing.
- **Interactive GUI**: A responsive Tkinter application for real-time inference.

## Project Structure

```
├── app.py                      # Main script to run the Tkinter GUI application
├── main.py                     # Main script to run the training pipeline
├── model.py                    # Definition of the CNN architecture
├── dataset.py                  # Data loading, augmentation, and splitting (Train/Val/Test)
├── ultilities.py               # Helper functions (EarlyStopping, plotting, model save/load)
├── requirements.txt            # Project dependencies
├── README.md                   # This file
├── checkpoints/                
    └── mnist_cnn_best.pt       # Best model checkpoint saved during training
```

## Technical Details

### Convolutional Nerual Network (CNN)

The core of the project is a custom CNN designed to be lightweight but accurate for the MNIST task. It incorporates modern best practices like Batch Normalization (for stable training) and Dropout (for regularization).

### Neural Network Architecture

```
Input (1, 28, 28)
    │
    ├─ Conv Block 1 ──────────
    │  ├─ Conv2d (1 -> 32, 3x3, padding=1)
    │  ├─ BatchNorm2d(32)
    │  ├─ ReLU
    │  └─ MaxPool2d(2, 2)
    │
    └─ Output: (32, 14, 14)
    │
    ├─ Conv Block 2 ──────────
    │  ├─ Conv2d (32 -> 64, 3x3, padding=1)
    │  ├─ BatchNorm2d(64)
    │  ├─ ReLU
    │  └─ MaxPool2d(2, 2)
    │
    └─ Output: (64, 7, 7)
    │
    ├─ Conv Block 3 ──────────
    │  ├─ Conv2d (64 -> 128, 3x3, padding=1)
    │  ├─ BatchNorm2d(128)
    │  └─ ReLU
    │
    └─ Output: (128, 7, 7)
    │
    ├─ Classifier ────────────
    │  ├─ Flatten (features = 128 * 7 * 7 = 6272)
    │  ├─ Linear (6272 -> 256) + ReLU
    │  ├─ Dropout(0.5)
    │  ├─ Linear (256 -> 128) + ReLU
    │  ├─ Dropout(0.5)
    │  └─ Linear (128 -> 10)
    │
    └─ Output: Logits (10 classes)
```

### Training Process

The model is trained using a robust pipeline::

1. **Load Data:** The get_data_loaders function loads the MNIST dataset.
- **Training Set:** Receives heavy data augmentation (ElasticTransform, RandomRotation, etc.).
- **Validation Set:** Split from the training set (10%) but uses no augmentation.
- **Test Set:** The original MNIST test set, used only for final evaluation.
2. **Initialize:** The CNN model, CrossEntropyLoss criterion, and Adam optimizer are initialized.
3. **Helpers:** ReduceLROnPlateau and EarlyStopping are initialized to monitor val_loss.
4. **Training Loop:** For each epoch:
- **Train Phase:** The model trains on all batches from train_loader. Loss and accuracy are logged using tqdm 
- **Validation Phase:** The model runs in eval() mode on the val_loader. 
- **Check:** The val_loss is passed to the scheduler and early stopping. 
- **Save:** If the val_acc is the best seen so far, the model is saved to checkpoints/mnist_cnn_best.pt.
5. **Stop:** The loop breaks if NUM_EPOCHS is reached or EarlyStopping is triggered.
6. **Final Test:** After the loop, the best saved model is loaded and run one time on the test_loader to get the final, unbiased performance report.
7. **Visualize:** Plots for loss/accuracy and sample predictions are saved.

**Hyperparameters:**
- Learning Rate: 0.001 (managed by Adam and ReduceLROnPlateau)
- Batch Size: 64
- Epochs: 20 (or fewer, if EarlyStopping triggers)
- Patience: 5 (for EarlyStopping), 3 (for ReduceLROnPlateau)
- Weight Decay: 1e-5 (L2 Regularization)

## Installation

### Requirements

- Python 3.8+
- A CUDA-enabled GPU (NVIDIA) is highly recommended for training.

### Setup

**Clone rep:**
```bash
# Clone the repository
git clone https://github.com/sk-o1911a/MNIST_CNN_Handwritting.git
cd MNIST_CNN_Handwritting
```

**Set up a Virtual Environment:**

```bash

# Windows
python -m venv .venv
.\.venv\Scripts\activate

# macOS / Linux
python3 -m venv .venv
source .venv/bin/activate
```

**Install dependencies:**
```bash
# Install requirements
pip install -r requirements.txt
```

**Install PyTorch (GPU Specific):**

This is the most important step. You must install the correct PyTorch version for your GPU.

Option A: For NVIDIA RTX 5000 Series (or newer)

(As of late 2025, this requires the nightly build for cu128+ support)
```bash
pip3 install --pre torch torchvision --index-url https://download.pytorch.org/whl/nightly/cu128
```

Option B: For NVIDIA RTX 4000 Series / 3000 Series (or older)

(This uses the stable build with CUDA 12.6, which is widely compatible)
```bash
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu126
```

Option C: For CPU-only (No NVIDIA GPU)

```bash
pip3 install torch torchvision
```

## Usage

### Training

Run the main training script with curriculum learning:

```bash
python main.py
```

This will:
- Run for multiple epochs (up to 20, but EarlyStopping will likely trigger sooner).
- You can monitor the progress in the terminal.
- The final training graphs (training_history.png) and test visualizations (predictions_visualization.png) will be saved.

### Running the Application

Once you have a trained model (either from running main.py or by using a provided .pt file), you can run the app.

```bash
python app.py
```

The script will:
- A Tkinter window will open.
- Draw a sequence of one or more digits (0-9) on the canvas.
- Click "Dự đoán (Predict)" to see the result.
- Click "Xóa (Clear)" to reset the canvas.



## References

- LeCun, Y., Bottou, L., Bengio, Y., & Haffner, P. (1998). Gradient-based learning applied to document recognition. Proceedings of the IEEE. 
- The MNIST Database of Handwritten Digits. http://yann.lecun.com/exdb/mnist/

## License

This project is submitted as coursework for Artificial Intelligence course.

---

**Note:** Training from scratch may take several minutes depending on hardware. A GPU is recommended for faster training. Pre-trained checkpoints can be loaded to skip initial training phases.