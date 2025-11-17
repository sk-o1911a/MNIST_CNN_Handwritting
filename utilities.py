import matplotlib.pyplot as plt
import torch
import os
import numpy as np
from PIL import Image
from torchvision import transforms


def plot_history(train_loss_history, val_loss_history, train_acc_history, val_acc_history,
                 save_path='training_history.png'):

    plt.figure(figsize=(14, 5))

    #Loss
    plt.subplot(1, 2, 1)
    plt.plot(train_loss_history, label='Training Loss', marker='o')
    plt.plot(val_loss_history, label='Validation Loss', marker='s')
    plt.title('Training and Validation Loss', fontsize=14, fontweight='bold')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)

    #Accuracy
    plt.subplot(1, 2, 2)
    plt.plot(train_acc_history, label='Training Accuracy', marker='o')
    plt.plot(val_acc_history, label='Validation Accuracy', marker='s')
    plt.title('Training and Validation Accuracy', fontsize=14, fontweight='bold')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"Training history saved to {save_path}")


def save_model(model, path='checkpoints/mnist_cnn.pt', optimizer=None, epoch=None, loss=None):
    if not os.path.exists('checkpoints'):
        os.makedirs('checkpoints')

    checkpoint = {
        'model_state_dict': model.state_dict(),
        'epoch': epoch,
        'loss': loss,
    }

    if optimizer is not None:
        checkpoint['optimizer_state_dict'] = optimizer.state_dict()

    torch.save(checkpoint, path)
    print(f"Model saved to {path}")


def load_model(model, path='checkpoints/mnist_cnn.pt', optimizer=None, device='cpu'):

    if not os.path.exists(path):
        raise FileNotFoundError(f"Model file not found: {path}")

    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])

    epoch = checkpoint.get('epoch', None)
    loss = checkpoint.get('loss', None)

    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        return model, optimizer, epoch, loss

    return model, epoch, loss


def predict_single_image(model, image_path, device='cpu'):

    transform = transforms.Compose([
        transforms.Grayscale(),
        transforms.Resize((28, 28)),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    image = Image.open(image_path)
    image_tensor = transform(image).unsqueeze(0).to(device)

    # Predict
    model.eval()
    with torch.no_grad():
        output = model(image_tensor)
        probabilities = torch.softmax(output, dim=1)
        confidence, predicted = torch.max(probabilities, 1)

    return predicted.item(), confidence.item()


def visualize_predictions(model, test_loader, device='cpu', num_images=10):

    model.eval()

    images, labels = next(iter(test_loader))
    images, labels = images[:num_images].to(device), labels[:num_images]

    with torch.no_grad():
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)

    fig, axes = plt.subplots(2, 5, figsize=(12, 6))
    axes = axes.ravel()

    for i in range(num_images):
        img = images[i].cpu().squeeze()
        axes[i].imshow(img, cmap='gray')

        color = 'green' if predicted[i] == labels[i] else 'red'
        axes[i].set_title(f'Pred: {predicted[i].item()}\nTrue: {labels[i].item()}',
                          color=color, fontweight='bold')
        axes[i].axis('off')

    plt.tight_layout()
    plt.savefig('predictions_visualization.png', dpi=300, bbox_inches='tight')
    plt.show()


class EarlyStopping:

    def __init__(self, patience=5, min_delta=0.001, verbose=True):

        self.patience = patience
        self.min_delta = min_delta
        self.verbose = verbose
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter}/{self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0