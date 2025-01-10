import torch
from torchvision.models import mobilenet_v2
import torch.nn as nn

# Charger le modèle PyTorch
model = mobilenet_v2(weights=None)
model.classifier[1] = nn.Linear(model.last_channel, 2)
model.load_state_dict(torch.load("modele_de_mais.pth", map_location=torch.device("cpu"), weights_only=True))
model.eval()

# Créer un exemple d'entrée
dummy_input = torch.randn(1, 3, 64, 64)

# Exporter le modèle au format ONNX
onnx_model_path = "modele_de_mais.onnx"
torch.onnx.export(
    model,
    dummy_input,
    onnx_model_path,
    export_params=True,
    opset_version=11,
    input_names=["input"],
    output_names=["output"],
    dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}}
)

print(f"Modèle ONNX sauvegardé dans : {onnx_model_path}")
