import multiprocessing as mp
import os

import numpy as np
# Optuna is preferred over GridSearchCV or RandomizedSearchCV
# due to its efficiency and speed for large datasets and complex models
import optuna
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch_directml  # Python 3.10 is a must for this to work
import torchvision.models as models  # For pre-trained models
import torchvision.transforms as transforms
from PIL import Image
from sklearn.metrics import classification_report, confusion_matrix
# For stratified sampling and a representative dataset split
from sklearn.model_selection import StratifiedShuffleSplit
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

##################################################
# Step 1
# Set up reproducibility for the model
#
# Setting seeds ensures consistent results across runs
# This is important for fair comparison between different
# model configurations and for debugging purposes
##################################################
# Set seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

##################################################
# Step 2
# Configure system performance settings
#
# Limiting threads helps prevent system overload
# Enabling CUDNN benchmark can speed up training
# when input sizes are consistent
##################################################
# Enable Intel MKL optimizations if available
torch.set_num_threads(
    min(8, mp.cpu_count())
)  # Limit to 8 threads to avoid crashing VS code - limit to 4 if on low-end CPU
torch.backends.cudnn.benchmark = True  # Enable benchmark mode for faster training

# Ensure saved_models directory exists
if not os.path.exists("saved_models"):
    os.makedirs("saved_models")


##################################################
# Step 3
# Create custom dataset class for ancient font images
#
# This class handles:
# - Loading images from file paths stored in the dataframe
# - Converting labels from strings to numerical indices
# - Applying transformations to images (resize, normalize, etc.)
# - Caching images in memory to speed up training
##################################################
class FontDataset(Dataset):
    def __init__(self, dataframe, transform=None, use_cache=True):
        self.dataframe = dataframe
        self.transform = transform
        self.classes = sorted(self.dataframe["Type"].unique())
        self.class_to_idx = {cls_name: idx for idx, cls_name in enumerate(self.classes)}
        self.use_cache = use_cache
        self.cache = {}

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        img_path = self.dataframe.iloc[idx]["Image"]
        label_str = self.dataframe.iloc[idx]["Type"]
        label = self.class_to_idx[label_str]

        # Use cached image if available
        if self.use_cache and img_path in self.cache:
            image = self.cache[img_path]
        else:
            image = Image.open(img_path).convert("RGB")
            if self.transform:
                image = self.transform(image)  # Apply transform before caching
            else:
                image = transforms.ToTensor()(image)

            if self.use_cache:
                self.cache[img_path] = (
                    image  # Store transformed tensor instead of raw image
                )

        return image, label


##################################################
# Step 4 and 5
# Define the Transfer Learning Model Architecture
#
# Instead of a custom CNN, we use a pre-trained MobileNetV2 model:
# - Pre-trained on ImageNet with 1000 classes
# - Replace the final classification layer for our 11 font classes
# - Freeze early layers to preserve learned features
# - Fine-tune later layers for our specific task
##################################################
def create_model(num_classes=11, freeze_layers=True):
    # Load pre-trained MobileNetV2
    model = models.mobilenet_v2(weights="IMAGENET1K_V1")

    # Freeze early layers to preserve learned features
    if freeze_layers:
        for param in list(model.parameters())[
            :-8
        ]:  # Freeze all but the last few layers
            param.requires_grad = False

    # Replace the final fully connected layer
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, num_classes)

    return model


##################################################
# Step 6
# Training function for a single epoch
#
# This function:
# - Sets the model to training mode
# - Loops through all batches in the training dataloader
# - Computes forward pass, loss, and updates weights
# - Tracks loss and accuracy
# - Returns average epoch loss and accuracy
##################################################
def train_one_epoch(model, optimizer, criterion, dataloader, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.set_grad_enabled(True):
        for images, labels in tqdm(dataloader, desc="Training", leave=False):
            images, labels = images.to(device), labels.to(device)

            # Mixed precision for faster computation
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)

            # Calculate accuracy
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    epoch_loss = running_loss / len(dataloader.dataset)
    accuracy = correct / total
    return epoch_loss, accuracy


##################################################
# Step 7
# Evaluation function for validation and testing
#
# This function:
# - Sets the model to evaluation mode
# - Disables gradient calculation for efficiency
# - Evaluates the model on the provided dataloader
# - Tracks loss, accuracy, predictions and true labels
# - Returns metrics for reporting and analysis
##################################################
def evaluate(model, criterion, dataloader, device):
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_labels = []
    total = 0
    correct = 0

    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc="Evaluating", leave=False):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            running_loss += loss.item() * images.size(0)

            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    avg_loss = running_loss / len(dataloader.dataset)
    accuracy = correct / total
    return avg_loss, accuracy, all_preds, all_labels


##################################################
# Step 8
# Early stopping implementation
#
# This class tracks validation loss and stops training when:
# - Loss doesn't improve for a specified number of epochs (patience)
# - Helps prevent overfitting by stopping training when performance plateaus
# - Saves computational resources by avoiding unnecessary epochs
##################################################
# Early stopping implementation
class EarlyStopping:
    def __init__(self, patience=5, min_delta=0.001):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0


##################################################
# Step 9
# Hyperparameter tuning objective function for Optuna
#
# This function:
# - Takes a trial object from Optuna
# - Suggests hyperparameters to test (learning rate, batch size, optimizer, freeze_layers)
# - Creates and trains a model with these parameters for a few epochs
# - Implements different learning rates for fine-tuning vs. training from scratch
# - Uses a learning rate scheduler and early stopping for efficiency
# - Returns validation loss for Optuna to minimize
##################################################
def objective(trial):
    # Trial parameters with narrower ranges for faster convergence
    lr = trial.suggest_float("lr", 1e-4, 5e-3, log=True)
    batch_size = trial.suggest_categorical(
        "batch_size", [32]
    )  # 64 and 128 are too large for the GPU
    optimizer_name = trial.suggest_categorical("optimizer", ["Adam", "SGD"])
    freeze_layers = trial.suggest_categorical("freeze_layers", [True, False])
    epochs = 3  # Reduced number of epochs for faster trials

    # Use minimal workers to avoid multiprocessing issues
    train_loader = get_dataloader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=2
    )
    val_loader = get_dataloader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=2
    )

    device = torch_directml.device()
    model = create_model(num_classes=11, freeze_layers=freeze_layers)
    model.to(device)

    criterion = nn.CrossEntropyLoss()

    # Different learning rate for fine-tuning vs. training from scratch
    if freeze_layers:
        # Only train the classifier head with higher learning rate
        optimizer_params = [{"params": model.classifier.parameters(), "lr": lr}]
    else:
        # Two different learning rates - slower for pre-trained layers
        optimizer_params = [
            {"params": model.classifier.parameters(), "lr": lr},
            {
                "params": [
                    p for n, p in model.named_parameters() if "classifier" not in n
                ],
                "lr": lr / 10,
            },
        ]

    if optimizer_name == "Adam":
        optimizer = optim.Adam(optimizer_params)
    else:
        optimizer = optim.SGD(optimizer_params, momentum=0.9)

    # Add learning rate scheduler
    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=1)

    # Add early stopping
    early_stopping = EarlyStopping(patience=2)

    best_val_loss = float("inf")
    print(
        f"Starting hyperparameter tuning trial with lr={lr}, batch_size={batch_size}, optimizer={optimizer_name}, freeze_layers={freeze_layers}"
    )

    for epoch in range(epochs):
        train_loss, train_acc = train_one_epoch(
            model, optimizer, criterion, train_loader, device
        )
        val_loss, val_acc, _, _ = evaluate(model, criterion, val_loader, device)

        print(
            f"Trial Epoch {epoch + 1}/{epochs} - Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} - Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}"
        )

        scheduler.step(val_loss)
        early_stopping(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss

        if early_stopping.early_stop:
            print("Early stopping triggered")
            break

    return best_val_loss


##################################################
# Step 10
# DataLoader creation helper function
#
# This function creates a PyTorch DataLoader with specified:
# - Dataset object
# - Batch size
# - Shuffle option
# - Number of worker processes for data loading
##################################################
def get_dataloader(dataset, batch_size=16, shuffle=False, num_workers=2):
    return DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers
    )


##################################################
# Step 11
# Main function - coordinates the entire training process
#
# This function:
# 1. Loads and prepares data
# 2. Performs stratified train/validation/test splits
# 3. Creates dataset objects with appropriate transformations
# 4. Runs hyperparameter optimization with Optuna
# 5. Trains the final model with best parameters
# 6. Evaluates on test set and reports metrics
# 7. Saves the trained model
##################################################
def main():
    ##################################################
    # Step 11.1
    # Data loading from CSV file
    #
    # The CSV contains image paths, font types, and whether
    # the image is original or augmented
    ##################################################
    # Data loading and preparation
    csv_file = "data/augmented_dataset.csv"
    df = pd.read_csv(csv_file)

    ##################################################
    # Step 11.2
    # Stratified dataset splitting
    #
    # First split: 80% train+validation, 20% test
    # Second split: 64% train, 16% validation (of the original dataset)
    #
    # Using stratified sampling to ensure class balance (font types)
    # in all splits - this is crucial for imbalanced datasets
    ##################################################
    # Dataset splitting
    splitter = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    for train_idx, test_idx in splitter.split(df, df["Type"]):
        train_val_df = df.iloc[train_idx].reset_index(drop=True)
        test_df = df.iloc[test_idx].reset_index(drop=True)

    splitter_val = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    for train_idx, val_idx in splitter_val.split(train_val_df, train_val_df["Type"]):
        train_df = train_val_df.iloc[train_idx].reset_index(drop=True)
        val_df = train_val_df.iloc[val_idx].reset_index(drop=True)

    print("Train set size:", len(train_df))
    print("Validation set size:", len(val_df))
    print("Test set size:", len(test_df))

    ##################################################
    # Step 11.3
    # Image preprocessing and transformation pipeline
    #
    # Modified to use ImageNet normalization values since
    # we're using a pre-trained model trained on ImageNet
    ##################################################
    # Reduce image size to 224x224 (to fit into GPU) and use ImageNet normalization values
    transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
            ),  # ImageNet normalization
        ]
    )

    ##################################################
    # Step 11.4
    # Creating dataset objects for train, validation and test sets
    #
    # Using the custom FontDataset class with:
    # - The appropriate dataframe subset
    # - The defined transformations
    # - Image caching enabled for faster training
    ##################################################
    global train_dataset, val_dataset, test_dataset
    train_dataset = FontDataset(train_df, transform=transform, use_cache=True)
    val_dataset = FontDataset(val_df, transform=transform, use_cache=True)
    test_dataset = FontDataset(test_df, transform=transform, use_cache=True)

    ##################################################
    # Step 11.5
    # Hyperparameter optimization with Optuna
    #
    # Running 5 trials to find optimal:
    # - Learning rate
    # - Batch size
    # - Optimizer type (Adam vs SGD)
    #
    # Each trial trains a model for a few epochs and reports validation loss
    ##################################################
    # Optimize study settings for faster trials
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=5)

    print("Best hyperparameters found: ", study.best_params)

    ##################################################
    # Step 11.6
    # Final model training with optimal hyperparameters
    #
    # Using the best parameters from Optuna to:
    # - Create dataloaders with optimal batch size
    # - Set up the model, loss function, and optimizer
    # - Configure learning rate scheduler and early stopping
    # - Train for up to 40 epochs (with early stopping possible)
    # - Save the best model based on validation loss
    ##################################################
    # Final training with the best parameters
    best_params = study.best_params
    best_lr = best_params["lr"]
    best_batch_size = best_params["batch_size"]
    best_optimizer_name = best_params["optimizer"]
    best_freeze_layers = best_params.get(
        "freeze_layers", True
    )  # Default to True if not in params

    # Use few workers to avoid multiprocessing issues
    train_loader = get_dataloader(
        train_dataset, batch_size=best_batch_size, shuffle=True, num_workers=2
    )
    val_loader = get_dataloader(
        val_dataset, batch_size=best_batch_size, shuffle=False, num_workers=2
    )
    test_loader = get_dataloader(
        test_dataset, batch_size=best_batch_size, shuffle=False, num_workers=2
    )

    device = (
        torch.device("cuda") if torch.cuda.is_available()
        else torch_directml.device()
    )  # Using DirectML for GPU acceleration with AMD RX 6600 XT - CPU was too slow
    model = create_model(num_classes=11, freeze_layers=best_freeze_layers)
    model.to(device)

    criterion = nn.CrossEntropyLoss()

    # Apply different learning rates based on whether layers are frozen
    if best_freeze_layers:
        optimizer_params = [{"params": model.classifier.parameters(), "lr": best_lr}]
    else:
        optimizer_params = [
            {"params": model.classifier.parameters(), "lr": best_lr},
            {
                "params": [
                    p for n, p in model.named_parameters() if "classifier" not in n
                ],
                "lr": best_lr / 10,
            },
        ]

    if best_optimizer_name == "Adam":
        optimizer = optim.Adam(optimizer_params)
    else:
        optimizer = optim.SGD(optimizer_params, momentum=0.9)

    # Add learning rate scheduler for the final training
    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=2)
    early_stopping = EarlyStopping(patience=5)

    num_epochs = 40  # Set a reasonable number of epochs for final training
    best_val_loss = float("inf")
    best_model_state = None

    print("Starting final training...")
    for epoch in range(num_epochs):
        print(f"Epoch {epoch + 1}/{num_epochs}")
        train_loss, train_acc = train_one_epoch(
            model, optimizer, criterion, train_loader, device
        )
        val_loss, val_acc, _, _ = evaluate(model, criterion, val_loader, device)

        print(
            f"Epoch {epoch + 1} - Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} - Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}"
        )

        scheduler.step(val_loss)
        early_stopping(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = model.state_dict()
            torch.save(best_model_state, "saved_models/current_best_model.pth")

        if early_stopping.early_stop:
            print("Early stopping triggered")
            break

    ##################################################
    # Step 11.7
    # Final model evaluation on test set
    #
    # This step:
    # - Loads the best model saved during training
    # - Evaluates it on the unseen test set
    # - Prints detailed classification metrics:
    #   * Overall accuracy
    #   * Per-class precision, recall, and F1-score
    #   * Confusion matrix to visualize predictions
    # - Saves the final model for future use
    ##################################################
    # Load the best model for evaluation
    if best_model_state:
        model.load_state_dict(best_model_state)

    # Final evaluation
    test_loss, test_acc, test_preds, test_labels = evaluate(
        model, criterion, test_loader, device
    )
    print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.4f}")
    print("Classification Report:")
    print(classification_report(test_labels, test_preds))
    print("Confusion Matrix:")
    print(confusion_matrix(test_labels, test_preds))

    # Save the final model
    torch.save(model.state_dict(), "saved_models/ancient_font_classifier.pth")
    print("Model saved as 'ancient_font_classifier.pth'")


##################################################
# Step 12
# Python script entry point
#
# This section:
# - Ensures proper multiprocessing support (especially for Windows)
# - Calls the main function to start the entire training pipeline
##################################################
if __name__ == "__main__":
    mp.freeze_support()  # For Windows compatibility
    main()
    # Note: The script assumes the presence of the CSV file (augmented_dataset.csv) with image paths and labels
    # and that the images are stored in the specific directory structure mentioned in the README
    # Ensure the CSV file and images are correctly set up before running the script
    # The script also assumes the presence of the required libraries and dependencies
    # Install any missing libraries using pip or conda as needed from the requirements.txt file
    # Example: pip install -r requirements.txt
    # Python 3.10 is a must for this version of the script as DirectML is not compatible with Python 3.11+
    # The script is designed to be run in an environment with GPU support for optimal performance
    # If running on CPU, the training will be slower but still functional
