def get_dockerfile_template(torch_install_line: str, pip_install_line: str) -> str:
    """Returns the Dockerfile content as a formatted string."""
    return f"""
FROM python:3.10-slim
WORKDIR /app

# Install torch
{torch_install_line}

# Install other libraries
{pip_install_line}

COPY train.py /app/train.py
COPY recipe.json /app/recipe.json

CMD ["python", "train.py"]
"""


def fallback_train_py():
    """A minimal, runnable PyTorch script, used if the Coder LLM fails."""
    fallback_code = """
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import mlflow
import os

print("--- RUNNING FALLBACK SCRIPT ---")

MLFLOW_TRACKING_URI = "sqlite:///mlflow.db"
MLFLOW_RUN_ID = os.environ.get("MLFLOW_RUN_ID", "default_run")

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

try:
    with mlflow.start_run(run_id=MLFLOW_RUN_ID):
        class DummyDataset(Dataset):
            def __len__(self): return 50
            def __getitem__(self, idx): return torch.randn(10), torch.randn(1)

        class SimpleModel(nn.Module):
            def __init__(self):
                super(SimpleModel, self).__init__()
                self.layer = nn.Linear(10, 1)
            def forward(self, x): return self.layer(x)

        model = SimpleModel()
        dataset = DummyDataset()
        dataloader = DataLoader(dataset, batch_size=8)
        criterion = nn.MSELoss()
        optimizer = optim.SGD(model.parameters(), lr=0.01)

        print("Model, Dataloader, and Optimizer set up. Starting training...")

        best_loss = float("inf")
        for epoch in range(20):
            epoch_loss = 0.0
            for inputs, labels in dataloader:
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()

            avg_loss = epoch_loss / len(dataloader)
            print(f"Epoch {epoch+1}/20, Loss: {avg_loss:.4f}")
            mlflow.log_metric("epoch_loss", avg_loss, step=epoch)

            if avg_loss < best_loss:
                best_loss = avg_loss
                torch.save(model.state_dict(), "fallback_best_model.pt")

        torch.save(model.state_dict(), "fallback_final_model.pt")
        print("Saved fallback models âœ…")

        print("--- FALLBACK SCRIPT COMPLETED ---")

except Exception as e:
    print(f"--- MLflow logging failed: {e} ---")
    print("--- FALLBACK SCRIPT COMPLETED (without logging) ---")
"""
    return fallback_code


def render_train_py(recipe: dict, run_id: str) -> str:
    """
    Generate training script that adapts to ANY dataset/model based on recipe.
    Dynamically selects appropriate data loading and model architecture.
    """
    
    # ============================================
    # EXTRACT AND CLEAN RECIPE VALUES
    # ============================================
    
    # Dataset name
    dataset_name = str(recipe.get("dataset", "")).lower()
    
    # Model architecture
    model_arch = str(recipe.get("model_architecture", "")).lower()
    
    # Batch size - with validation
    try:
        batch_size = int(recipe.get("batch_size", 32))
        if batch_size <= 0 or batch_size > 1024:
            batch_size = 32
    except:
        batch_size = 32
    
    # Learning rate - with validation
    try:
        lr = float(recipe.get("learning_rate", 0.001))
        if lr <= 0 or lr > 1:
            lr = 0.001
    except:
        lr = 0.001
    
    # Optimizer - CLEAN IT!
    optimizer_raw = str(recipe.get("optimizer", "Adam"))
    optimizer_map = {
        "adam": "Adam",
        "adamw": "AdamW",
        "sgd": "SGD",
        "rmsprop": "RMSprop",
        "adagrad": "Adagrad"
    }
    
    # Extract optimizer name from potentially long description
    optimizer_name = "Adam"  # default
    if len(optimizer_raw) <= 50:  # If it's short, use it directly
        for key, value in optimizer_map.items():
            if key == optimizer_raw.lower().strip():
                optimizer_name = value
                break
    else:  # If it's a long description, search for optimizer name in it
        for key, value in optimizer_map.items():
            if key in optimizer_raw.lower():
                optimizer_name = value
                break
    
    # Loss function - CLEAN IT!
    loss_fn_raw = str(recipe.get("loss_function", "CrossEntropyLoss"))
    
    # Handle weird loss function values
    if "[MASK]" in loss_fn_raw or len(loss_fn_raw) > 100 or "document" in loss_fn_raw.lower():
        loss_fn = "CrossEntropyLoss"
    elif "cross" in loss_fn_raw.lower() or "entropy" in loss_fn_raw.lower():
        loss_fn = "CrossEntropyLoss"
    elif "bce" in loss_fn_raw.lower() or "binary" in loss_fn_raw.lower():
        loss_fn = "BCEWithLogitsLoss"
    elif "mse" in loss_fn_raw.lower():
        loss_fn = "MSELoss"
    else:
        loss_fn = "CrossEntropyLoss"
    
    # Epochs
    epochs = 20  # Reasonable default
    
    # ============================================
    # DYNAMIC DATASET SELECTION
    # ============================================
    
    data_loading_code = _get_data_loading_code(dataset_name, batch_size)
    
    # ============================================
    # DYNAMIC MODEL SELECTION
    # ============================================
    
    model_creation_code = _get_model_creation_code(model_arch)
    
    # ============================================
    # DYNAMIC LOSS FUNCTION
    # ============================================
    
    loss_code = _get_loss_function_code(loss_fn)
    
    # ============================================
    # COMPLETE TRAINING SCRIPT
    # ============================================
    
    code = f"""
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
run_id = os.environ.get("MLFLOW_RUN_ID", "{run_id}")
print(f"MLFLOW_RUN_ID: {{run_id}}")

print("="*60)
print("LOADING DATASET: {dataset_name}")
print("="*60)

{data_loading_code}

print("="*60)
print("CREATING MODEL: {model_arch}")
print("="*60)

{model_creation_code}

print("="*60)
print("SETTING UP TRAINING")
print("="*60)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {{device}}")
model = model.to(device)

{loss_code}

optimizer = optim.{optimizer_name}(model.parameters(), lr={lr})
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
            print(f"Warning: Batch {{i}} failed: {{e}}")
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
                print(f"Warning: Evaluation batch failed: {{e}}")
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
    
    for epoch in range({epochs}):
        print(f"\\nEpoch {{epoch+1}}/{epochs}")
        print("-" * 40)
        
        train_loss, train_acc = train_epoch(model, trainloader, criterion, optimizer, device)
        val_loss, val_acc = evaluate(model, testloader, criterion, device)
        
        print(f"Train Loss: {{train_loss:.4f}}, Train Acc: {{train_acc:.2f}}%")
        print(f"Val Loss:   {{val_loss:.4f}}, Val Acc:   {{val_acc:.2f}}%")
        
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
            print(f"  Patience: {{patience_counter}}/{{patience}}")
        
        # Early stopping
        if patience_counter >= patience:
            print(f"\\nEarly stopping triggered after {{epoch+1}} epochs")
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
    print(f"Best validation loss: {{best_val_loss:.4f}}")
"""
    
    return code


def _get_data_loading_code(dataset_name: str, batch_size: int) -> str:
    """
    Dynamically generate data loading code based on dataset name.
    Supports: CIFAR-10, CIFAR-100, MNIST, ImageNet, IMDB, custom datasets.
    """
    
    dataset_name = dataset_name.lower()
    
    # ===== COMPUTER VISION DATASETS =====
    
    if "cifar-10" in dataset_name or "cifar10" in dataset_name:
        return f"""
import torchvision
import torchvision.transforms as transforms

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
])

print("Downloading CIFAR-10 dataset...")
trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

trainloader = torch.utils.data.DataLoader(trainset, batch_size={batch_size}, shuffle=True, num_workers=2)
testloader = torch.utils.data.DataLoader(testset, batch_size={batch_size}, shuffle=False, num_workers=2)

num_classes = 10
input_shape = (3, 32, 32)
print(f"Dataset loaded: {{len(trainset)}} train, {{len(testset)}} test samples")
"""
    
    elif "cifar-100" in dataset_name or "cifar100" in dataset_name:
        return f"""
import torchvision
import torchvision.transforms as transforms

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
])

print("Downloading CIFAR-100 dataset...")
trainset = torchvision.datasets.CIFAR100(root='./data', train=True, download=True, transform=transform)
testset = torchvision.datasets.CIFAR100(root='./data', train=False, download=True, transform=transform)

trainloader = torch.utils.data.DataLoader(trainset, batch_size={batch_size}, shuffle=True, num_workers=2)
testloader = torch.utils.data.DataLoader(testset, batch_size={batch_size}, shuffle=False, num_workers=2)

num_classes = 100
input_shape = (3, 32, 32)
print(f"Dataset loaded: {{len(trainset)}} train, {{len(testset)}} test samples")
"""
    
    elif "mnist" in dataset_name:
        return f"""
import torchvision
import torchvision.transforms as transforms

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

print("Downloading MNIST dataset...")
trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
testset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)

trainloader = torch.utils.data.DataLoader(trainset, batch_size={batch_size}, shuffle=True, num_workers=2)
testloader = torch.utils.data.DataLoader(testset, batch_size={batch_size}, shuffle=False, num_workers=2)

num_classes = 10
input_shape = (1, 28, 28)
print(f"Dataset loaded: {{len(trainset)}} train, {{len(testset)}} test samples")
"""
    
    elif "fashion" in dataset_name and "mnist" in dataset_name:
        return f"""
import torchvision
import torchvision.transforms as transforms

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.2860,), (0.3530,))
])

print("Downloading Fashion-MNIST dataset...")
trainset = torchvision.datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform)
testset = torchvision.datasets.FashionMNIST(root='./data', train=False, download=True, transform=transform)

trainloader = torch.utils.data.DataLoader(trainset, batch_size={batch_size}, shuffle=True, num_workers=2)
testloader = torch.utils.data.DataLoader(testset, batch_size={batch_size}, shuffle=False, num_workers=2)

num_classes = 10
input_shape = (1, 28, 28)
print(f"Dataset loaded: {{len(trainset)}} train, {{len(testset)}} test samples")
"""
    
    # ===== NLP DATASETS =====
    
    elif "imdb" in dataset_name:
        return f"""
from datasets import load_dataset
from transformers import AutoTokenizer
from torch.utils.data import DataLoader

print("Downloading IMDB dataset...")
dataset = load_dataset("imdb")
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=512)

print("Tokenizing dataset...")
tokenized_datasets = dataset.map(tokenize_function, batched=True)
tokenized_datasets = tokenized_datasets.remove_columns(["text"])
tokenized_datasets = tokenized_datasets.rename_column("label", "labels")
tokenized_datasets.set_format("torch")

trainloader = DataLoader(tokenized_datasets["train"], batch_size={batch_size}, shuffle=True)
testloader = DataLoader(tokenized_datasets["test"], batch_size={batch_size})

num_classes = 2
print(f"Dataset loaded: {{len(tokenized_datasets['train'])}} train, {{len(tokenized_datasets['test'])}} test samples")
"""
    
    elif "glue" in dataset_name or "sst" in dataset_name or "mnli" in dataset_name:
        return f"""
from datasets import load_dataset
from transformers import AutoTokenizer
from torch.utils.data import DataLoader

print("Downloading GLUE dataset...")
try:
    dataset = load_dataset("glue", "sst2")
    split_name = "validation"
except:
    dataset = load_dataset("glue", "mnli")
    split_name = "validation_matched"

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

def tokenize_function(examples):
    if "sentence" in examples:
        return tokenizer(examples["sentence"], padding="max_length", truncation=True, max_length=128)
    elif "premise" in examples:
        return tokenizer(examples["premise"], examples["hypothesis"], padding="max_length", truncation=True, max_length=128)
    else:
        return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=128)

print("Tokenizing dataset...")
tokenized_datasets = dataset.map(tokenize_function, batched=True)
tokenized_datasets = tokenized_datasets.rename_column("label", "labels")
tokenized_datasets.set_format("torch")

trainloader = DataLoader(tokenized_datasets["train"], batch_size={batch_size}, shuffle=True)
testloader = DataLoader(tokenized_datasets[split_name], batch_size={batch_size})

num_classes = len(set(tokenized_datasets["train"]["labels"].numpy()))
print(f"Dataset loaded: {{len(tokenized_datasets['train'])}} train, {{len(tokenized_datasets[split_name])}} test samples")
"""
    
    # ===== IMAGENET (SUBSET) =====
    
    elif "imagenet" in dataset_name:
        return f"""
import torchvision
import torchvision.transforms as transforms

# Using ImageNet subset or ImageNette for speed
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

print("Note: Using ImageNet requires manual download. Using dummy data as fallback.")
print("For real ImageNet, download from https://image-net.org/")

# Fallback to dummy data
class DummyImageNet(torch.utils.data.Dataset):
    def __init__(self, size=10000):
        self.size = size
    def __len__(self):
        return self.size
    def __getitem__(self, idx):
        return torch.randn(3, 224, 224), torch.randint(0, 1000, (1,)).item()

trainset = DummyImageNet(10000)
testset = DummyImageNet(1000)

trainloader = torch.utils.data.DataLoader(trainset, batch_size={batch_size}, shuffle=True, num_workers=2)
testloader = torch.utils.data.DataLoader(testset, batch_size={batch_size}, shuffle=False, num_workers=2)

num_classes = 1000
input_shape = (3, 224, 224)
print(f"Dataset loaded: {{len(trainset)}} train, {{len(testset)}} test samples")
"""
    
    # ===== FALLBACK: DUMMY DATA =====
    
    else:
        return f"""
print("Warning: Dataset '{dataset_name}' not recognized. Using synthetic data.")
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

trainloader = torch.utils.data.DataLoader(trainset, batch_size={batch_size}, shuffle=True)
testloader = torch.utils.data.DataLoader(testset, batch_size={batch_size})

num_classes = 10
input_shape = (3, 32, 32)
print(f"Dataset loaded: {{len(trainset)}} train, {{len(testset)}} test samples (SYNTHETIC)")
"""


def _get_model_creation_code(model_arch: str) -> str:
    """
    Dynamically generate model creation code based on architecture name.
    Supports: ResNet, VGG, BERT, GPT, EfficientNet, custom models.
    """
    
    model_arch = model_arch.lower()
    
    # ===== RESNETS =====
    
    if "resnet-50" in model_arch or "resnet50" in model_arch:
        return """
import torchvision.models as models
model = models.resnet50(pretrained=False, num_classes=num_classes)
print(f"Model: ResNet-50 with {{num_classes}} classes")
"""
    
    elif "resnet-18" in model_arch or "resnet18" in model_arch:
        return """
import torchvision.models as models
model = models.resnet18(pretrained=False, num_classes=num_classes)
print(f"Model: ResNet-18 with {{num_classes}} classes")
"""
    
    elif "resnet" in model_arch:
        return """
import torchvision.models as models
# Default to ResNet-18 for speed
model = models.resnet18(pretrained=False, num_classes=num_classes)
print(f"Model: ResNet-18 with {{num_classes}} classes")
"""
    
    # ===== VGG =====
    
    elif "vgg" in model_arch:
        return """
import torchvision.models as models
model = models.vgg16(pretrained=False, num_classes=num_classes)
print(f"Model: VGG-16 with {{num_classes}} classes")
"""
    
    # ===== EFFICIENTNET =====
    
    elif "efficientnet" in model_arch:
        return """
import torchvision.models as models
model = models.efficientnet_b0(pretrained=False, num_classes=num_classes)
print(f"Model: EfficientNet-B0 with {{num_classes}} classes")
"""
    
    # ===== BERT =====
    
    elif "bert" in model_arch:
        return """
from transformers import AutoModelForSequenceClassification
model = AutoModelForSequenceClassification.from_pretrained(
    "bert-base-uncased",
    num_labels=num_classes
)
print(f"Model: BERT-base with {{num_classes}} classes")
"""
    
    # ===== GPT =====
    
    elif "gpt" in model_arch:
        return """
from transformers import GPT2ForSequenceClassification
model = GPT2ForSequenceClassification.from_pretrained(
    "gpt2",
    num_labels=num_classes
)
model.config.pad_token_id = model.config.eos_token_id
print(f"Model: GPT-2 with {{num_classes}} classes")
"""
    
    # ===== CUSTOM CNN =====
    
    else:
        return """
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
"""


def _get_loss_function_code(loss_fn: str) -> str:
    """Generate appropriate loss function code."""
    
    # Already cleaned in render_train_py, so just return the code
    if "Cross" in loss_fn or "Entropy" in loss_fn:
        return "criterion = nn.CrossEntropyLoss()"
    elif "BCE" in loss_fn:
        return "criterion = nn.BCEWithLogitsLoss()"
    elif "MSE" in loss_fn:
        return "criterion = nn.MSELoss()"
    elif "L1" in loss_fn or "MAE" in loss_fn:
        return "criterion = nn.L1Loss()"
    else:
        return "criterion = nn.CrossEntropyLoss()  # Default"