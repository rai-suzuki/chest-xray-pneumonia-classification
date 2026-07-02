import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
from torchvision import transforms
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# Grad-CAM用のインポート (pip install grad-cam が必要です)
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

from dataset import get_dataloaders
from model import get_resnet18

def evaluate_model(model, dataloader, device, class_names):
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
    print(f"Accuracy : {accuracy_score(all_labels, all_preds):.4f}")
    print(f"Precision: {precision_score(all_labels, all_preds):.4f}")
    print(f"Recall   : {recall_score(all_labels, all_preds):.4f}")
    print(f"F1-score : {f1_score(all_labels, all_preds):.4f}")
    
    cm = confusion_matrix(all_labels, all_preds)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.show()

def visualize_gradcam(model, image_path, true_label_name, device):
    for param in model.parameters():
        param.requires_grad = True

    img_pil = Image.open(image_path).convert('RGB')
    mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
    
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
    input_tensor = transform(img_pil).unsqueeze(0).to(device)
    
    img_resized = img_pil.resize((224, 224))
    img_display = np.float32(img_resized) / 255.0

    model.eval()
    with torch.no_grad():
        output = model(input_tensor)
        prob = torch.softmax(output, dim=1)[0, 1].item()
        pred_idx = torch.argmax(output, dim=1).item()
        pred_label_name = 'PNEUMONIA' if pred_idx == 1 else 'NORMAL'

    # ResNet18の場合は layer4[-1] をターゲットにする
    target_layers = [model.layer4[-1]]
    cam = GradCAM(model=model, target_layers=target_layers)
    targets = [ClassifierOutputTarget(pred_idx)]
    
    grayscale_cam = cam(input_tensor=input_tensor, targets=targets)[0, :]
    visualization = show_cam_on_image(img_display, grayscale_cam, use_rgb=True)

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow(img_display)
    axes[0].set_title(f"Original Image\nTrue: {true_label_name}")
    axes[0].axis('off')
    
    axes[1].imshow(visualization)
    axes[1].set_title(f"Grad-CAM\nPred: {pred_label_name} (Prob: {prob:.2f})")
    axes[1].axis('off')
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataloaders, dataset_sizes, class_names, mean, std = get_dataloaders()
    
    # 保存したモデルの読み込み
    model = get_resnet18(fine_tune=False).to(device)
    # model.load_state_dict(torch.load("best_model.pth")) # 学習済み重みがある場合はアンコメント
    
    print("--- Evaluation on Test Set ---")
    evaluate_model(model, dataloaders['test'], device, class_names)
    
    # Grad-CAMのテスト実行例
    # sample_path = './chest_xray/test/PNEUMONIA/person1_virus_7.jpeg'
    # visualize_gradcam(model, sample_path, "PNEUMONIA", device)