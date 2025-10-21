import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader, TensorDataset, random_split
import logging
from typing import Tuple, Optional

# Postavi logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class LeakDetectorModel(nn.Module):
    def __init__(self, input_size: int, hidden_size: int = 64, num_classes: int = 1):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=16, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(16)
        self.conv2 = nn.Conv1d(in_channels=16, out_channels=32, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(32)
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)
        self.fc1_input_size = self._get_conv_output(input_size)
        self.fc1 = nn.Linear(self.fc1_input_size, hidden_size)
        self.bn3 = nn.BatchNorm1d(hidden_size)
        self.fc2 = nn.Linear(hidden_size, num_classes)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

        nn.init.xavier_uniform_(self.conv1.weight)
        nn.init.xavier_uniform_(self.conv2.weight)
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        self.conv1.bias.data.fill_(0.01)
        self.conv2.bias.data.fill_(0.01)
        self.fc1.bias.data.fill_(0.01)
        self.fc2.bias.data.fill_(0.01)

    def _get_conv_output(self, input_size):
        with torch.no_grad():
            x = torch.zeros(1, 1, input_size)
            x = self.bn1(nn.ReLU()(self.conv1(x)))
            x = self.pool(self.bn2(nn.ReLU()(self.conv2(x))))
            return int(x.nelement() / x.shape[0])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.bn1(self.relu(self.conv1(x)))
        x = self.pool(self.bn2(self.relu(self.conv2(x))))
        x = x.view(x.size(0), -1)
        x = self.bn3(self.relu(self.fc1(x)))
        x = self.sigmoid(self.fc2(x))
        return x

def train_model(model: nn.Module, train_loader: DataLoader, val_loader: DataLoader, num_epochs: int = 50, learning_rate: float = 0.0001, patience: int = 5) -> None:
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    best_val_loss = float('inf')
    patience_counter = 0

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            if torch.isnan(outputs).any():
                print("NaN detektovan u outputs! Pauziram treniranje.")
                print(f"Inputs min/max: {inputs.min().item()}, {inputs.max().item()}")
                return
            loss = criterion(outputs, labels)
            if torch.isnan(loss).any():
                print("NaN detektovan u loss! Pauziram treniranje.")
                return
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        train_loss = running_loss / len(train_loader)
        
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for inputs, labels in val_loader:
                outputs = model(inputs)
                val_loss += criterion(outputs, labels).item()
        val_loss = val_loss / len(val_loader)
        
        logger.info(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
        with open('training_history.txt', 'a') as f:
            f.write(f"Epoch {epoch+1}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}\n")
    
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), 'best_leak_detector_model.pth')
        else:
            patience_counter += 1
            if patience_counter >= patience:
                logger.info(f"Early stopping triggered after {epoch+1} epochs.")
                break

def evaluate_model(model: nn.Module, test_loader: DataLoader, thresholds: list = [0.4, 0.5, 0.6]) -> dict:
    model.eval()
    results = {}
    
    for threshold in thresholds:
        correct = 0
        total = 0
        tp, fp, fn = 0, 0, 0

        with torch.no_grad():
            for inputs, labels in test_loader:
                outputs = model(inputs)
                predicted = (outputs >= threshold).float()
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                tp += ((predicted == 1) & (labels == 1)).sum().item()
                fp += ((predicted == 1) & (labels == 0)).sum().item()
                fn += ((predicted == 0) & (labels == 1)).sum().item()

        accuracy = correct / total
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        results[f"Threshold {threshold}"] = {"Accuracy": accuracy, "Precision": precision, "Recall": recall}
    
    return results

def prepare_data(df_scaled: np.ndarray, labels: pd.Series) -> Tuple[torch.Tensor, torch.Tensor]:
    logger.info("Priprema podataka za treniranje.")
    if df_scaled.shape[1] > 0:
        features = df_scaled
    else:
        raise ValueError("Nema feature-a u df_scaled!")
    
    if np.any(np.isnan(features)) or np.any(np.isinf(features)):
        print("Pronađeni NaN ili inf vrednosti u feature-ima! Popunjavam sa 0.")
        features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
    if np.any(np.isnan(labels.values)) or np.any(np.isinf(labels.values)):
        print("Pronađeni NaN ili inf vrednosti u labelama! Popunjavam sa 0.")
        labels = pd.Series(np.nan_to_num(labels.values, nan=0.0, posinf=0.0, neginf=0.0))

    features_tensor = torch.FloatTensor(features).unsqueeze(1)  # [batch, 1, features]
    labels_tensor = torch.FloatTensor(labels.values).unsqueeze(1)
    
    if torch.any(torch.isnan(features_tensor)) or torch.any(torch.isinf(features_tensor)):
        print("Pronađeni NaN ili inf u features_tensor! Popunjavam sa 0.")
        features_tensor = torch.nan_to_num(features_tensor, nan=0.0, posinf=0.0, neginf=0.0)
    if torch.any(torch.isnan(labels_tensor)) or torch.any(torch.isinf(labels_tensor)):
        print("Pronađeni NaN ili inf u labels_tensor! Popunjavam sa 0.")
        labels_tensor = torch.nan_to_num(labels_tensor, nan=0.0, posinf=0.0, neginf=0.0)

    return features_tensor, labels_tensor

def predict_leakage(model: nn.Module, data: np.ndarray, threshold: float = 0.5) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        # Prilagođeno: Uklanja nepotrebnu dimenziju, očekuje [batch, 1, features]
        data_tensor = torch.FloatTensor(data)  # [batch, features]
        if data_tensor.dim() == 2:
            data_tensor = data_tensor.unsqueeze(1)  # [batch, 1, features]
        outputs = model(data_tensor)
        predictions = (outputs >= threshold).float().numpy().flatten()
    return predictions

if __name__ == "__main__":
    from data_processor import DataProcessor

    # Učitaj i obradi podatke
    processor = DataProcessor("../data/location_aware_gis_leakage_dataset.csv")
    df = processor.load_data()
    df_clean = processor.clean_data()
    df_features = processor.feature_engineering()
    df_scaled, scaler, labels = processor.scale_data()

    # Provera podataka pre konverzije
    if np.any(np.isnan(df_scaled)) or np.any(np.isinf(df_scaled)):
        print("Pronađeni NaN ili inf u skaliranim podacima! Popunjavam sa 0.")
        df_scaled = np.nan_to_num(df_scaled, nan=0.0, posinf=0.0, neginf=0.0)
    if np.any(np.isnan(labels)) or np.any(np.isinf(labels)):
        print("Pronađeni NaN ili inf u labelama! Popunjavam sa 0.")
        labels = np.nan_to_num(labels, nan=0.0, posinf=0.0, neginf=0.0)

    # Pripremi podatke
    X, y = prepare_data(df_scaled, labels)
    dataset = TensorDataset(X, y)
    total_size = len(dataset)
    train_size = int(0.8 * total_size)
    val_size = int(0.1 * total_size)
    test_size = total_size - train_size - val_size
    train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    # Inicijalizuj i treniraj model
    model = LeakDetectorModel(input_size=X.shape[2])
    train_model(model, train_loader, val_loader, patience=5)

    # Evaluacija modela sa različitim pragovima
    results = evaluate_model(model, test_loader, thresholds=[0.4, 0.5, 0.6])
    for threshold, metrics in results.items():
        logger.info(f"{threshold} - Accuracy: {metrics['Accuracy']:.4f}, Precision: {metrics['Precision']:.4f}, Recall: {metrics['Recall']:.4f}")

    # Testiranje predikcije na delu test seta
    test_features, test_labels = next(iter(test_loader))
    predictions = predict_leakage(model, test_features.numpy(), threshold=0.5)
    logger.info(f"Primer predikcija na test setu (prvih 5): {predictions[:5]}")
    logger.info(f"Stvarne vrednosti (prvih 5): {test_labels[:5].numpy().flatten()}")
    np.savetxt('test_predictions.csv', predictions, delimiter=',')
    np.savetxt('test_labels.csv', test_labels.numpy().flatten(), delimiter=',')
    # Nakon train_model i evaluate_model
    torch.save(model, 'leak_detector_model_full.pth')