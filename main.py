import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm

from model import CNN
from dataset import get_data_loaders
from utilities import plot_history,save_model,visualize_predictions,EarlyStopping



def main():

    LEARNING_RATE = 0.001
    BATCH_SIZE = 64
    NUM_EPOCHS = 20
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    PATIENCE = 5

    train_loader, val_loader, test_loader = get_data_loaders(BATCH_SIZE, val_split=0.1)

    print(f"Train batches: {len(train_loader)}")
    print(f"Validation batches: {len(val_loader)}")
    print(f"Test batches: {len(test_loader)}")


    model = CNN().to(DEVICE)

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-5)

    scheduler = ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=3
    )

    early_stopping = EarlyStopping(patience=PATIENCE, min_delta=0.001)

    train_loss_history = []
    val_loss_history = []
    train_acc_history = []
    val_acc_history = []

    best_val_acc = 0.0

    for epoch in range(NUM_EPOCHS):

        model.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0

        train_loop = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{NUM_EPOCHS} [Train]")

        for images, labels in train_loop:
            images, labels = images.to(DEVICE), labels.to(DEVICE)

            # Forward
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)

            # Backward
            loss.backward()
            optimizer.step()

            # Statistics
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total_train += labels.size(0)
            correct_train += (predicted == labels).sum().item()

            # Update progress bar
            train_loop.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{100 * correct_train / total_train:.2f}%'
            })

        epoch_train_loss = running_loss / len(train_loader)
        epoch_train_acc = 100 * correct_train / total_train
        train_loss_history.append(epoch_train_loss)
        train_acc_history.append(epoch_train_acc)

        # Validation
        model.eval()
        val_loss = 0.0
        correct_val = 0
        total_val = 0

        val_loop = tqdm(val_loader, desc=f"Epoch {epoch + 1}/{NUM_EPOCHS} [Val]")

        with torch.no_grad():
            for images, labels in val_loop:
                images, labels = images.to(DEVICE), labels.to(DEVICE)

                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()

                _, predicted = torch.max(outputs.data, 1)
                total_val += labels.size(0)
                correct_val += (predicted == labels).sum().item()

                val_loop.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'acc': f'{100 * correct_val / total_val:.2f}%'
                })

        epoch_val_loss = val_loss / len(val_loader)
        epoch_val_acc = 100 * correct_val / total_val
        val_loss_history.append(epoch_val_loss)
        val_acc_history.append(epoch_val_acc)

        tqdm.write("")
        tqdm.write(
            f"Epoch {epoch + 1:02d} | "
            f"Train Loss: {epoch_train_loss:.4f}, Train Acc: {epoch_train_acc:.2f}% | "
            f"Val Loss: {epoch_val_loss:.4f}, Val Acc: {epoch_val_acc:.2f}%"
        )

        scheduler.step(epoch_val_loss)

        if epoch_val_acc > best_val_acc:
            best_val_acc = epoch_val_acc
            save_model(
                model,
                'checkpoints/mnist_cnn_best.pt',
                optimizer=optimizer,
                epoch=epoch + 1,
                loss=epoch_val_loss
            )
            tqdm.write(f"New best model saved! Val Acc: {best_val_acc:.2f}%\n")

        # Early stop
        early_stopping(epoch_val_loss)
        if early_stopping.early_stop:
            tqdm.write(f"Early stopping triggered at epoch {epoch + 1}")
            break

    print("\n" + "=" * 60)
    print("Training Finished!")
    print(f"Best Validation Accuracy: {best_val_acc:.2f}%")
    print("=" * 60 + "\n")

    # Save final model
    save_model(
        model,
        'checkpoints/mnist_cnn_final.pt',
        optimizer=optimizer,
        epoch=epoch + 1,
        loss=epoch_val_loss
    )

    plot_history(train_loss_history, val_loss_history, train_acc_history, val_acc_history)

    # ===== Test =====
    print("\n" + "=" * 60)
    print("Testing on Test Set...")
    print("=" * 60)

    model.eval()
    test_loss = 0.0
    correct_test = 0
    total_test = 0

    test_loop = tqdm(test_loader, desc="Testing")

    with torch.no_grad():
        for images, labels in test_loop:
            images, labels = images.to(DEVICE), labels.to(DEVICE)

            outputs = model(images)
            loss = criterion(outputs, labels)
            test_loss += loss.item()

            _, predicted = torch.max(outputs.data, 1)
            total_test += labels.size(0)
            correct_test += (predicted == labels).sum().item()

    test_loss = test_loss / len(test_loader)
    test_acc = 100 * correct_test / total_test

    print(f"\n{'=' * 60}")
    print(f"  Test Loss: {test_loss:.4f}")
    print(f"  Test Acc:  {test_acc:.2f}%")
    print(f"{'=' * 60}\n")

    print("Generating predictions visualization...")
    visualize_predictions(model, test_loader, DEVICE, num_images=10)

    print("\nDONE")


if __name__ == '__main__':
    main()