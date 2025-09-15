import torch
import torch.nn as nn
import torchvision
import argparse
from PIL import Image
from torchvision import transforms

def build_model(num_classes=2, weights_path='models/best_model.pth', device=None):
    """Construct the model, load trained weights, move to device, set eval."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = torchvision.models.efficientnet_b7(weights=None)
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.5, inplace=True),
        nn.Linear(model.classifier[1].in_features, num_classes)
    )
    state = torch.load(weights_path, map_location=device)
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model, device


def get_transform(image_size=(256, 256)):
    """Return the same transform used during training."""
    return transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor()
    ])


def predict(image_path, model, device, transform, return_confidence=True):
    """Run inference on a single image path and return class index (and confidence)."""
    image = Image.open(image_path).convert('RGB')
    tensor = transform(image).unsqueeze(0).to(device)
    with torch.inference_mode():
        logits = model(tensor)
        if return_confidence:
            probs = torch.softmax(logits, dim=1)
            conf, pred = torch.max(probs, 1)
            return pred.item(), conf.item()
        else:
            _, pred = torch.max(logits, 1)
            return pred.item()

def main():
    parser = argparse.ArgumentParser(description='CNN Inference Script')
    parser.add_argument('image_path', type=str, help='Path to the input image')
    args = parser.parse_args()

    class_names = ['Faulty', 'Normal']

    model, device = build_model(num_classes=len(class_names), weights_path='models/best_model.pth')
    transform = get_transform((256, 256))

    try:
        pred_idx, conf = predict(args.image_path, model, device, transform, return_confidence=True)
    except FileNotFoundError:
        print(f"Image not found: {args.image_path}")
        return
    except Exception as e:
        print(f"Error during inference: {e}")
        return

    pred_name = class_names[pred_idx] if pred_idx < len(class_names) else str(pred_idx)
    print(f"Predicted class: {pred_name} (confidence: {conf:.3f})")

if __name__ == "__main__":
    main()
