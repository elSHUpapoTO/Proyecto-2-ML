# 1. Instalar MedMNIST (si no está instalado)
!pip install medmnist


import tqdm
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import torchvision.transforms as transforms
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import pandas as pd

import medmnist
from medmnist import INFO, Evaluator

print(f"MedMNIST v{medmnist.__version__} @ {medmnist.HOMEPAGE}")

# 3. Configuración del Experimento
data_flag = 'pneumoniamnist'
download = True

NUM_EPOCHS = 15
BATCH_SIZE = 64
lr = 0.001
HIDDEN_SIZE = 128

info = INFO[data_flag]
task = info['task']
n_channels = info['n_channels']
n_classes = len(info['label'])

DataClass = getattr(medmnist, info['python_class'])

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Usando dispositivo: {device}")

# 4. Preprocesamiento y Carga de Datos
data_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[.5], std=[.5])
])

def get_dataloaders(resolution, batch_size, download):
    print(f"\nCargando dataset {resolution}x{resolution}...")
    train_dataset = DataClass(split='train', transform=data_transform, download=download, size=resolution)
    val_dataset = DataClass(split='val', transform=data_transform, download=download, size=resolution)
    test_dataset = DataClass(split='test', transform=data_transform, download=download, size=resolution)

    train_loader = data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = data.DataLoader(dataset=val_dataset, batch_size=batch_size*2, shuffle=False)
    test_loader = data.DataLoader(dataset=test_dataset, batch_size=batch_size*2, shuffle=False)

    print(f"Dataset {resolution}x{resolution} - Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")
    return train_loader, val_loader, test_loader, train_dataset.labels

def calculate_class_weights(labels_array):
    if labels_array.ndim > 1:
        labels_array = labels_array.flatten()
    
    unique_labels, counts = np.unique(labels_array, return_counts=True)
    class_counts = dict(zip(unique_labels, counts))
    total_samples = sum(class_counts.values())

    weights = np.ones(n_classes) 
    
    for i in range(n_classes):
        if i in class_counts and class_counts[i] > 0:
            weights[i] = total_samples / (n_classes * class_counts[i])
        else:
            weights[i] = total_samples
    
    return torch.tensor(weights, dtype=torch.float).to(device)

# Cargar datasets para ambas resoluciones y calcular pesos
train_loader_64, val_loader_64, test_loader_64, train_labels_64 = get_dataloaders(64, BATCH_SIZE, download)
train_loader_224, val_loader_224, test_loader_224, train_labels_224 = get_dataloaders(224, BATCH_SIZE, download)

class_weights_64 = calculate_class_weights(train_labels_64)
class_weights_224 = calculate_class_weights(train_labels_224)

print(f"\nPesos de clase para 64x64: {class_weights_64.tolist()}")
print(f"Pesos de clase para 224x224: {class_weights_224.tolist()}")

# 5. Definición del Modelo (MLP y Mezcla de Expertos)
class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, num_classes)
        # print(f"MLP: input_size={input_size}, hidden_size={hidden_size}") # Debug print

    def forward(self, x):
        x = x.view(x.size(0), -1)
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out

class Expert(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(Expert, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, num_classes)
        )
        # print(f"Expert: input_size={input_size}, hidden_size={hidden_size}") # Debug print

    def forward(self, x):
        # Aplanar la entrada para el Expert antes de pasarla a la red
        x_flat = x.view(x.size(0), -1)
        return self.net(x_flat)

class GatingNetwork(nn.Module):
    def __init__(self, input_size, num_experts):
        super(GatingNetwork, self).__init__()
        self.fc = nn.Linear(input_size, num_experts)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x_flat = x.view(x.size(0), -1)
        return self.softmax(self.fc(x_flat))

class MoE(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes, num_experts):
        super(MoE, self).__init__()
        self.num_experts = num_experts
        self.experts = nn.ModuleList([Expert(input_size, hidden_size, num_classes) for _ in range(num_experts)])
        self.gating = GatingNetwork(input_size, num_experts)

    def forward(self, x):
        gates = self.gating(x)

        expert_outputs = [expert(x) for expert in self.experts]
        
        expert_outputs_tensor = torch.stack(expert_outputs, dim=2) 
        
        gates_expanded = gates.unsqueeze(1) 

        out = torch.sum(gates_expanded * expert_outputs_tensor, dim=2) 
        return out

# 6. Funciones de Entrenamiento y Evaluación
def train_model(model, train_loader, criterion, optimizer, num_epochs, model_name=""):
    model.train()
    print(f"\nIniciando entrenamiento para {model_name} en {device}...")
    for epoch in range(num_epochs):
        running_loss = 0.0 # Reiniciar running_loss para cada época
        for inputs, labels in tqdm.tqdm(train_loader, desc=f"[{model_name}] Epoch {epoch+1}/{num_epochs}"):
            inputs, labels = inputs.to(device), labels.to(device).long()
            labels = labels.view(-1)
            optimizer.zero_grad()
            outputs = model(inputs)

            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * inputs.size(0)

        epoch_loss = running_loss / len(train_loader.dataset)
        print(f"[{model_name}] Epoch {epoch+1} Loss: {epoch_loss:.4f}")
    print(f"Entrenamiento para {model_name} completado.")

def evaluate_model(model, data_loader, task_type, model_name="", data_set_name=""):
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for inputs, labels in tqdm.tqdm(data_loader, desc=f"Evaluando {model_name} en {data_set_name}"):
            inputs, labels = inputs.to(device), labels.to(device)
            labels = labels.view(-1)
            outputs = model(inputs)

            if task_type == 'binary-class':
                predicted = torch.argmax(outputs, dim=1)
            else:
                predicted = torch.argmax(outputs, dim=1)

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average='binary', pos_label=1)
    recall = recall_score(all_labels, all_preds, average='binary', pos_label=1)
    f1 = f1_score(all_labels, all_preds, average='binary', pos_label=1)
    cm = confusion_matrix(all_labels, all_preds)

    print(f"\nResultados de evaluación para {model_name} en {data_set_name}:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-score: {f1:.4f}")
    print("Matriz de Confusión:\n", cm)

    return {"accuracy": accuracy, "precision": precision, "recall": recall, "f1_score": f1, "confusion_matrix": cm.tolist()}

def run_experiment(model_type, resolution, num_experts=None, balanced=False):
    print(f"\n--- Ejecutando Experimento: {model_type} | Resolución: {resolution}x{resolution} | Expertos: {num_experts if num_experts else 'N/A'} | Balanceado: {balanced} ---")
    
    input_size = n_channels * resolution * resolution
    model_name = f"{model_type}_{resolution}x{resolution}{'_MoE'+str(num_experts) if model_type=='MoE' else ''}{'_Balanced' if balanced else ''}"
    
    if model_type == 'MLP':
        model = MLP(input_size, HIDDEN_SIZE, n_classes).to(device)
    elif model_type == 'MoE':
        if num_experts is None:
            raise ValueError("MoE requires 'num_experts' to be specified.")
        model = MoE(input_size, HIDDEN_SIZE, n_classes, num_experts).to(device)
    else:
        raise ValueError("Tipo de modelo no soportado.")
    
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    if resolution == 64:
        train_loader, val_loader, test_loader = train_loader_64, val_loader_64, test_loader_64
        weights = class_weights_64 if balanced else None
    else:
        train_loader, val_loader, test_loader = train_loader_224, val_loader_224, test_loader_224
        weights = class_weights_224 if balanced else None

    if balanced:
        criterion = nn.CrossEntropyLoss(weight=weights)
    else:
        criterion = nn.CrossEntropyLoss()
    
    train_model(model, train_loader, criterion, optimizer, NUM_EPOCHS, model_name)
    
    val_results = evaluate_model(model, val_loader, task, model_name, "Validation Set")
    test_results = evaluate_model(model, test_loader, task, model_name, "Test Set")
    
    return {
        "model_type": model_type,
        "resolution": resolution,
        "num_experts": num_experts,
        "balanced": balanced,
        "validation_metrics": val_results,
        "test_metrics": test_results
    }

# 7. Ejecución de Experimentos
all_experiment_results = []

# Experimentos MLP
for res in [64, 224]:
    for bal in [False, True]:
        result = run_experiment('MLP', resolution=res, balanced=bal)
        all_experiment_results.append(result)

# Experimentos Mezcla de Expertos (MoE)
for res in [64, 224]:
    for num_exp in [2, 3, 5]:
        for bal in [False, True]:
            result = run_experiment('MoE', resolution=res, num_experts=num_exp, balanced=bal)
            all_experiment_results.append(result)

# 8. Almacenar y Presentar Resultados (opcional, pero recomendado)
results_df = pd.DataFrame(all_experiment_results)

flattened_results = []
for index, row in results_df.iterrows():
    val_metrics = {f"val_{k}": v for k, v in row['validation_metrics'].items()}
    test_metrics = {f"test_{k}": v for k, v in row['test_metrics'].items()}
    
    flattened_results.append({
        "model_type": row['model_type'],
        "resolution": row['resolution'],
        "num_experts": row['num_experts'],
        "balanced": row['balanced'],
        **val_metrics,
        **test_metrics
    })

final_results_df = pd.DataFrame(flattened_results)

print("\n--- Resumen de Todos los Experimentos ---")
display_cols = ['model_type', 'resolution', 'num_experts', 'balanced', 
                'test_accuracy', 'test_precision', 'test_recall', 'test_f1_score']
print(final_results_df[display_cols].to_string())

final_results_df.to_csv("experiment_results.csv", index=False)
print("\nResultados guardados en 'experiment_results.csv'")