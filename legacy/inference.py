"""Inspect predictions from the checkpoint produced by cnn.py."""
import argparse
import numpy as np
import torch
from cnn import WaferCNN

parser = argparse.ArgumentParser()
parser.add_argument("--index", type=int, default=0, help="Map index in full_dataset.npz")
parser.add_argument("--samples", type=int, default=20, help="Number of random maps to inspect")
args = parser.parse_args()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
checkpoint = torch.load("wafer_cnn.pth", map_location=device, weights_only=False)
if "model_state" not in checkpoint or "classes" not in checkpoint:
    raise RuntimeError(
        "wafer_cnn.pth is the old subset-model checkpoint. Run `python cnn.py` "
        "to train and save a full-dataset checkpoint before running inference."
    )
classes = checkpoint["classes"]
data = np.load("full_dataset.npz")
maps, labels = data["X"], data["y"]
if not 0 <= args.index < len(maps): raise ValueError(f"--index must be between 0 and {len(maps) - 1}")
model = WaferCNN(len(classes)).to(device)
model.load_state_dict(checkpoint["model_state"]); model.eval()

def predict(index):
    image = torch.from_numpy(maps[index]).unsqueeze(0).unsqueeze(0).float().to(device)
    with torch.inference_mode(): probabilities = torch.softmax(model(image), dim=1)[0].cpu()
    return probabilities.argmax().item(), probabilities

predicted, probabilities = predict(args.index)
print("WAFER MAP INFERENCE")
print(f"Sample index : {args.index}\nActual class : {classes[labels[args.index]]}\nPredicted    : {classes[predicted]}\nConfidence   : {probabilities[predicted]:.2%}\n")
for name, probability in zip(classes, probabilities): print(f"{name:10s}: {probability:.2%}")
indices = torch.randperm(len(maps))[:args.samples].tolist(); correct = 0
print("\nRANDOM-SAMPLE CHECK")
for index in indices:
    predicted, probabilities = predict(index); actual = int(labels[index]); correct += predicted == actual
    print(f"{index:6d} | actual: {classes[actual]:10s} | predicted: {classes[predicted]:10s} | confidence: {probabilities[predicted]:.2%}")
print(f"Accuracy on this random sample: {correct / len(indices):.2%}")
