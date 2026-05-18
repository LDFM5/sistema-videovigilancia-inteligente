import torch
import torch.nn as nn
import numpy as np

# ==========================================
# 1. DEFINE LA ARQUITECTURA DE TU GRU
# ==========================================
# Esta clase DEBE ser idéntica a la que usaste para entrenar el modelo.
class ActionGRU(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(ActionGRU, self).__init__()
        self.hidden_size = hidden_size
        
        # Capa GRU
        self.gru = nn.GRU(input_size, hidden_size, num_layers=1, batch_first=True)
        
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 32), # Capa 0
            nn.ReLU(),                  # Capa 1
            nn.Dropout(0.5),            # Capa 2 (El valor del dropout no importa para la carga)
            nn.Linear(32, num_classes)  # Capa 3
        )

    def forward(self, x):
        # x shape: (batch, sequence_length, input_size)
        out, _ = self.gru(x)
        # Tomamos solo la salida del último frame de la secuencia
        out = self.fc(out[:, -1, :]) 
        return out

# ==========================================
# 2. CARGADOR DEL MODELO
# ==========================================
def load_behavior_model(pth_path, input_size=34, hidden_size=64, num_classes=2):
    """Carga el modelo GRU y lo prepara para inferencia en CPU o GPU."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model = ActionGRU(input_size, hidden_size, num_classes)
    
    # Cargar los pesos. map_location asegura que cargue bien sea en CPU o GPU
    model.load_state_dict(torch.load(pth_path, map_location=device))
    model.to(device)
    model.eval() # Modo inferencia (apaga el dropout y gradientes)
    
    print(f"✅ Modelo GRU cargado en {device}")
    return model, device

# ==========================================
# 3. LÓGICA DE PREDICCIÓN
# ==========================================
def predict_behavior(gru_model, device, buffer_esqueletos, clases_nombres):
    """
    Convierte el historial de una persona en un tensor y hace la predicción.
    """
    # Convertimos la lista de deques a un array de numpy
    # Shape esperado: (sequence_length, input_size)
    seq_array = np.array(buffer_esqueletos)
    
    # Convertimos a tensor y añadimos la dimensión del batch: (1, seq_len, input_size)
    seq_tensor = torch.tensor(seq_array, dtype=torch.float32).unsqueeze(0).to(device)
    
    with torch.no_grad(): # Sin gradientes para máxima velocidad
        outputs = gru_model(seq_tensor)
        # Obtenemos la clase ganadora
        _, predicted = torch.max(outputs.data, 1)
        clase_idx = predicted.item()
        
    return clases_nombres[clase_idx]