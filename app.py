import tkinter as tk
import torch
import torch.nn.functional as F
import os
import cv2
import numpy as np
from tkinter import ttk
from PIL import Image, ImageDraw
from utilities import load_model
from model import CNN
from torchvision import transforms


# -------- Configuration --------
CANVAS_WIDTH = 900
CANVAS_HEIGHT = 280
IMG_SIZE = 28
LINE_WIDTH = 20
MODEL_PATH = 'checkpoints/mnist_cnn_best.pt'
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

class DigitApp:
    def __init__(self, root):
        self.root = root
        root.title("Vẽ chữ số - Dự đoán MNIST")

        # load model
        self.model = CNN().to(DEVICE)
        self.model_loaded = False
        if not os.path.exists(MODEL_PATH):
            print(f"[WARN] Model file not found: {MODEL_PATH}")
        else:
            self.model, _, _ = load_model(self.model, MODEL_PATH, device=DEVICE)
            self.model.eval()
            self.model_loaded = True
            print(f"Model loaded from {MODEL_PATH}")

        self.canvas = tk.Canvas(root, width=CANVAS_WIDTH, height=CANVAS_HEIGHT, bg='white', cursor='cross')
        self.canvas.grid(row=0, column=0, rowspan=4, padx=10, pady=10)

        self.image1 = Image.new("RGB", (CANVAS_WIDTH, CANVAS_HEIGHT), 'white')
        self.draw = ImageDraw.Draw(self.image1)

        self.canvas.bind("<B1-Motion>", self.draw_callback)
        self.canvas.bind("<ButtonPress-1>", self.draw_start)
        self.last_x, self.last_y = None, None

        btn_clear = ttk.Button(root, text="Xóa (Clear)", command=self.clear_canvas)
        btn_clear.grid(row=0, column=1, sticky='ew', padx=6, pady=6)

        btn_predict = ttk.Button(root, text="Dự đoán (Predict)", command=self.predict_canvas)
        btn_predict.grid(row=1, column=1, sticky='ew', padx=6, pady=6)

        self.result_var = tk.StringVar(value="Vẽ chuỗi số (VD: 123)")
        self.result_label = ttk.Label(root, textvariable=self.result_var, font=('Helvetica', 24, 'bold'))
        self.result_label.grid(row=2, column=1, padx=6, pady=20)

        self.conf_var = tk.StringVar(value="Độ tin cậy: --")
        self.conf_label = ttk.Label(root, textvariable=self.conf_var, font=('Helvetica', 12))
        self.conf_label.grid(row=3, column=1, padx=6, pady=0, sticky='n')

        if not self.model_loaded:
            self.result_var.set("Model chưa load.")
            self.conf_var.set(f"Kiểm tra file: {MODEL_PATH}")

    def draw_start(self, event):
        self.last_x, self.last_y = event.x, event.y

    def draw_callback(self, event):
        x, y = event.x, event.y
        if self.last_x is not None and self.last_y is not None:
            self.canvas.create_line(self.last_x, self.last_y, x, y, width=LINE_WIDTH, fill='black', capstyle=tk.ROUND, smooth=True)
            self.draw.line([self.last_x, self.last_y, x, y], fill='black', width=LINE_WIDTH)
        self.last_x, self.last_y = x, y

    def clear_canvas(self):
        self.canvas.delete("all")
        self.draw.rectangle([0, 0, CANVAS_WIDTH, CANVAS_HEIGHT], fill='white')
        self.result_var.set("Vẽ chuỗi số")
        self.conf_var.set("Độ tin cậy: --")

    def preprocess_digit_roi(self, roi):
        h, w = roi.shape
        max_dim = max(w, h)
        pad_top = (max_dim - h) // 2
        pad_bottom = max_dim - h - pad_top
        pad_left = (max_dim - w) // 2
        pad_right = max_dim - w - pad_left

        border = int(max_dim * 0.2)

        padded_img = cv2.copyMakeBorder(
            roi,
            pad_top + border,
            pad_bottom + border,
            pad_left + border,
            pad_right + border,
            cv2.BORDER_CONSTANT,
            value=0  # Nền đen
        )

        resized_img = cv2.resize(padded_img, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)
        pil_img = Image.fromarray(resized_img)
        tensor = transform(pil_img)
        return tensor.unsqueeze(0).to(DEVICE)

    def predict_canvas(self):
        if not self.model_loaded:
            self.result_var.set("Không có model để dự đoán.")
            return

        img = self.image1.convert('L')
        img_np = np.array(img)

        img_inverted = cv2.bitwise_not(img_np)
        _, thresh = cv2.threshold(img_inverted, 30, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if not contours:
            self.result_var.set("Không tìm thấy số nào.")
            self.conf_var.set("Độ tin cậy: --")
            return

        bounding_boxes = []
        for c in contours:
            (x, y, w, h) = cv2.boundingRect(c)
            if w * h > 50:
                bounding_boxes.append((x, y, w, h))

        sorted_boxes = sorted(bounding_boxes, key=lambda b: b[0])
        predictions = []
        confidences = []

        for (x, y, w, h) in sorted_boxes:
            digit_roi = img_inverted[y:y + h, x:x + w]
            tensor = self.preprocess_digit_roi(digit_roi)

            with torch.no_grad():
                outputs = self.model(tensor)
                probs = F.softmax(outputs, dim=1)
                conf, pred = torch.max(probs, 1)

                predictions.append(str(pred.item()))
                confidences.append(conf.item())

        final_prediction_str = " ".join(predictions)
        if predictions:
            avg_confidence = (sum(confidences) / len(confidences)) * 100
            self.result_var.set(f"Dự đoán: {final_prediction_str}")
            self.conf_var.set(f"Độ tin cậy (TB): {avg_confidence:.2f}%")
        else:
            self.result_var.set("Không tìm thấy số nào.")
            self.conf_var.set("Độ tin cậy: --")


if __name__ == '__main__':
    root = tk.Tk()
    app = DigitApp(root)
    root.resizable(False, False)
    root.mainloop()