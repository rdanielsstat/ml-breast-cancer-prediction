import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
from medmnist import BreastMNIST
import numpy as np

# Project setup
# Set seeds for reproducibility
def set_seed(seed = 1601):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

set_seed(1601)

# Detect best device
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

# Force CPU usage for deterministic results (comment out to use GPU/MPS)
device = torch.device('cpu')

print(f"Using device: {device}")

# Model architecture
# Create CNN model, allow conv and fc dropout rates to be set manually
class BreastCNN(nn.Module):
    def __init__(self, conv_drop_rate = 0.0, fc_drop_rate = 0.3):
        super().__init__()

        # Block 1
        self.conv1 = nn.Conv2d(1, 16, kernel_size = 3, padding = 1)
        self.bn1 = nn.BatchNorm2d(16)

        # Block 2
        self.conv2 = nn.Conv2d(16, 32, kernel_size = 3, padding = 1)
        self.bn2 = nn.BatchNorm2d(32)

        # Block 3
        self.conv3 = nn.Conv2d(32, 64, kernel_size = 3, padding = 1)
        self.bn3 = nn.BatchNorm2d(64)

        # Block 4
        self.conv4 = nn.Conv2d(64, 128, kernel_size = 3, padding = 1)
        self.bn4 = nn.BatchNorm2d(128)

        self.pool = nn.MaxPool2d(2, 2)

        # Dropouts
        self.drop_conv = nn.Dropout2d(p = conv_drop_rate)
        self.drop_fc = nn.Dropout(p = fc_drop_rate)

        # Adaptive pooling for input-size independence
        self.gap = nn.AdaptiveAvgPool2d((4, 4))

        # Fully connected
        self.fc = nn.Linear(128 * 4 * 4, 2)

    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))

        x = self.drop_conv(x)

        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        x = self.pool(F.relu(self.bn4(self.conv4(x))))

        x = self.gap(x)

        x = x.view(x.size(0), -1)

        x = self.drop_fc(x)
        x = self.fc(x)

        return x

# Data loaders
def make_loaders():
    set_seed(1601)
    val_test_transform = transforms.Compose([transforms.ToTensor()])
    
    train_dataset = BreastMNIST(split = 'train', download = True, size = 128, transform = val_test_transform)
    val_dataset = BreastMNIST(split = 'val', download = True, size = 128, transform = val_test_transform)
    test_dataset = BreastMNIST(split = 'test', download = True, size = 128, transform = val_test_transform)
    
    return (
        DataLoader(train_dataset, batch_size = 32, shuffle = True, num_workers = 0),
        DataLoader(val_dataset, batch_size = 32, shuffle = False, num_workers = 0),
        DataLoader(test_dataset, batch_size = 32, shuffle = False, num_workers = 0),
    )

# Training epoch
def run_epoch(model, loader, criterion, optimizer = None):
    if optimizer:
        model.train()
    else:
        model.eval()
    
    running_loss = 0.0
    correct = 0
    total = 0
    
    context = torch.enable_grad() if optimizer else torch.no_grad()
    with context:
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device).view(-1).long()
            
            if optimizer:
                optimizer.zero_grad()
            
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            if optimizer:
                loss.backward()
                optimizer.step()
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    
    return running_loss / len(loader), 100. * correct / total

# Main training function
def train_final_model():
    print("Training final model ...")
    
    # Load data
    train_loader, val_loader, test_loader = make_loaders()
    
    # Initialize model (V1 best settings)
    model = BreastCNN(conv_drop_rate = 0.0, fc_drop_rate = 0.3).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), weight_decay = 1e-4)
    
    # Training loop
    best_val_acc = 0.0
    num_epochs = 30
    
    for epoch in range(num_epochs):
        train_loss, train_acc = run_epoch(model, train_loader, criterion, optimizer)
        val_loss, val_acc = run_epoch(model, val_loader, criterion)
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            # Save best model
            torch.save(model.state_dict(), 'best_model.pth')
            status = "✓ Saved"
        else:
            status = ""
        
        print(f"Epoch {epoch+1:02d} | "
              f"Train: {train_acc:.1f}% ({train_loss:.4f}) | "
              f"Val: {val_acc:.1f}% ({val_loss:.4f}) {status}")
    
    # Load best model and evaluate on test
    model.load_state_dict(torch.load('best_model.pth'))
    _, test_acc = run_epoch(model, test_loader, criterion)
    
    print(f"\nFinal test accuracy: {test_acc:.2f}%")
    
    # Save the final model for deployment
    torch.save({
        'model_state_dict': model.state_dict(),
        'model_architecture': 'BreastCNN',
        'c_drop': 0.0,
        'f_drop': 0.3,
        'test_accuracy': test_acc,
    }, 'final_model.pth')
    
    print("Model saved to 'final_model.pth'")
    
    return model

if __name__ == "__main__":
    model = train_final_model()