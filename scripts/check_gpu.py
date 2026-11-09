"""Vérifie l'installation de PyTorch et la présence d'un GPU CUDA."""
import torch

print("torch :", torch.__version__)
print("CUDA disponible :", torch.cuda.is_available())
if torch.cuda.is_available():
    print("version CUDA (build) :", torch.version.cuda)
    print("GPU :", torch.cuda.get_device_name(0))
else:
    print("Pas de GPU utilisable : exécution sur CPU,", torch.get_num_threads(), "threads.")

device = "cuda" if torch.cuda.is_available() else "cpu"
x = torch.randn(256, 256, device=device)
print("Test de calcul sur", device, ":", float((x @ x).sum()) == float((x @ x).sum()))
