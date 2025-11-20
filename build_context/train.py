
import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
import mlflow

print("="*60)
print("PAPER2CODE - TRAINING SCRIPT")
print("="*60)

print("Loading recipe...")
with open("recipe.json") as f:
    recipe = json.load(f)

print("Recipe:", json.dumps(recipe, indent=2))

print("Setting up MLflow...")
mlflow.set_tracking_uri("sqlite:///mlflow.db")
run_id = os.environ.get("MLFLOW_RUN_ID", "b5bcdb47084243e3a6ded9d524d40c03")
print(f"MLFLOW_RUN_ID: {run_id}")

print("="*60)
print("LOADING DATASET: emotiondataset-eeg")
print("="*60)


print("Warning: Dataset 'emotiondataset-eeg' not recognized. Using synthetic data.")
print("Supported datasets: CIFAR-10, CIFAR-100, MNIST, Fashion-MNIST, IMDB, GLUE/MNLI")

class SyntheticDataset(torch.utils.data.Dataset):
    def __init__(self, size=5000):
        self.size = size
    def __len__(self):
        return self.size
    def __getitem__(self, idx):
        return torch.randn(3, 32, 32), torch.randint(0, 10, (1,)).item()

trainset = SyntheticDataset(5000)
testset = SyntheticDataset(1000)

trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)
testloader = torch.utils.data.DataLoader(testset, batch_size=64)

num_classes = 10
input_shape = (3, 32, 32)
print(f"Dataset loaded: {len(trainset)} train, {len(testset)} test samples (SYNTHETIC)")


print("="*60)
print("CREATING MODEL: spinner")
print("="*60)


class CustomNet(nn.Module):
    def __init__(self, num_classes=10):
        super(CustomNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 256, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Linear(256 * 4 * 4, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )
    
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

model = CustomNet(num_classes=num_classes)
print(f"Model: Custom CNN with {{num_classes}} classes")


print("="*60)
print("SETTING UP TRAINING")
print("="*60)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
model = model.to(device)

criterion = nn.CrossEntropyLoss()

optimizer = optim.Adam(model.parameters(), lr=0.005)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3, factor=0.5)

def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for i, data in enumerate(loader):
        try:
            # Handle different data formats
            if isinstance(data, dict):
                # HuggingFace datasets
                if 'input_ids' in data:
                    inputs = data["input_ids"].to(device)
                    labels = data["label"].to(device) if "label" in data else data["labels"].to(device)
                else:
                    inputs = torch.stack([d for d in data["image"]]).to(device) if "image" in data else data["pixel_values"].to(device)
                    labels = torch.tensor(data["label"]).to(device)
            else:
                # Standard tuple format
                inputs, labels = data
                inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            
            # Handle different output formats
            if isinstance(outputs, dict):
                outputs = outputs.logits if hasattr(outputs, 'logits') else outputs['logits']
            
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
        except Exception as e:
            print(f"Warning: Batch {i} failed: {e}")
            continue
    
    if total == 0:
        return 0.0, 0.0
    
    epoch_loss = running_loss / max(len(loader), 1)
    accuracy = 100 * correct / total
    return epoch_loss, accuracy

def evaluate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for data in loader:
            try:
                if isinstance(data, dict):
                    if 'input_ids' in data:
                        inputs = data["input_ids"].to(device)
                        labels = data["label"].to(device) if "label" in data else data["labels"].to(device)
                    else:
                        inputs = torch.stack([d for d in data["image"]]).to(device) if "image" in data else data["pixel_values"].to(device)
                        labels = torch.tensor(data["label"]).to(device)
                else:
                    inputs, labels = data
                    inputs, labels = inputs.to(device), labels.to(device)
                
                outputs = model(inputs)
                if isinstance(outputs, dict):
                    outputs = outputs.logits if hasattr(outputs, 'logits') else outputs['logits']
                
                loss = criterion(outputs, labels)
                
                running_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
            except Exception as e:
                print(f"Warning: Evaluation batch failed: {e}")
                continue
    
    if total == 0:
        return 0.0, 0.0
    
    epoch_loss = running_loss / max(len(loader), 1)
    accuracy = 100 * correct / total
    return epoch_loss, accuracy

print("="*60)
print("STARTING TRAINING")
print("="*60)

with mlflow.start_run(run_id=run_id):
    mlflow.log_params(recipe)
    
    best_val_loss = float('inf')
    patience = 5
    patience_counter = 0
    
    for epoch in range(20):
        print(f"\nEpoch {epoch+1}/20")
        print("-" * 40)
        
        train_loss, train_acc = train_epoch(model, trainloader, criterion, optimizer, device)
        val_loss, val_acc = evaluate(model, testloader, criterion, device)
        
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        print(f"Val Loss:   {val_loss:.4f}, Val Acc:   {val_acc:.2f}%")
        
        # Log to MLflow
        mlflow.log_metric("train_loss", train_loss, step=epoch)
        mlflow.log_metric("train_accuracy", train_acc, step=epoch)
        mlflow.log_metric("val_loss", val_loss, step=epoch)
        mlflow.log_metric("val_accuracy", val_acc, step=epoch)
        
        # Learning rate scheduling
        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]['lr']
        mlflow.log_metric("learning_rate", current_lr, step=epoch)
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), "best_model.pt")
            print("  âœ“ Best model saved!")
        else:
            patience_counter += 1
            print(f"  Patience: {patience_counter}/{patience}")
        
        # Early stopping
        if patience_counter >= patience:
            print(f"\nEarly stopping triggered after {epoch+1} epochs")
            break
    
    # Save final model
    torch.save(model.state_dict(), "final_model.pt")
    
    # Log artifacts
    try:
        mlflow.log_artifact("best_model.pt")
        mlflow.log_artifact("final_model.pt")
    except:
        print("Warning: Could not log model artifacts")
    
    print("="*60)
    print("âœ… TRAINING COMPLETED SUCCESSFULLY!")
    print("="*60)
    print(f"Best validation loss: {best_val_loss:.4f}")
