# detection/explainability.py
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import os
from torch.nn import IntegratedGradients, Occlusion
from lime import lime_image
from PIL import Image

# Configuration (adjust based on your model)
config = {
    "normalize_mean": [0.485, 0.456, 0.406],  # Example values for ImageNet
    "normalize_std": [0.229, 0.224, 0.225],   # Example values for ImageNet
    "device": torch.device("cuda" if torch.cuda.is_available() else "cpu")
}

class_names = ['Healthy', 'Foot Infection', 'Mouth Infection']  # Adjust based on your model

class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        self.register_hooks()
        self.model.eval()

    def register_hooks(self):
        def forward_hook(module, input, output):
            self.activations = output.detach()

        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0].detach()

        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_backward_hook(backward_hook)

    def generate_cam(self, input_tensor, target_class=None):
        output = self.model(input_tensor)

        if target_class is None:
            target_class = output.argmax(dim=1).item()

        self.model.zero_grad()
        one_hot = torch.zeros_like(output)
        one_hot[0, target_class] = 1
        output.backward(gradient=one_hot, retain_graph=True)

        weights = torch.mean(self.gradients, dim=(2, 3), keepdim=True)
        cam = torch.sum(weights * self.activations, dim=1, keepdim=True)
        cam = F.relu(cam)
        cam = F.interpolate(cam, size=input_tensor.shape[2:], mode='bilinear', align_corners=False)
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)

        return cam.squeeze().cpu().numpy(), output

def apply_explainability(model, input_tensor, target_class, technique='integrated_gradients'):
    model.eval()
    input_tensor = input_tensor.float().to(config["device"])

    if technique == 'integrated_gradients':
        input_tensor = input_tensor.requires_grad_()
        ig = IntegratedGradients(model)
        attributions = ig.attribute(input_tensor, target=target_class, n_steps=50)
        return attributions

    elif technique == 'gradcam':
        if hasattr(model, 'layer4'):  # ResNet
            target_layer = model.layer4[-1]
        elif hasattr(model, 'features'):  # VGG, DenseNet
            target_layer = model.features[-1]
        elif hasattr(model, 'blocks'):  # Vision Transformer
            target_layer = model.blocks[-1]
        else:
            target_layer = list(model.children())[-2]

        grad_cam = GradCAM(model, target_layer)
        cam, _ = grad_cam.generate_cam(input_tensor, target_class)
        cam_tensor = torch.tensor(cam).unsqueeze(0).unsqueeze(0)
        cam_tensor = cam_tensor.repeat(1, 3, 1, 1)
        return cam_tensor.to(config["device"])

    elif technique == 'occlusion':
        occlusion = Occlusion(model)
        attributions = occlusion.attribute(
            input_tensor,
            target=target_class,
            strides=(3, 8, 8),
            sliding_window_shapes=(3, 15, 15)
        )
        return attributions

    elif technique == 'lime':
        def batch_predict(images):
            batch = torch.stack([torch.tensor(img.astype(np.float32)).permute(2, 0, 1) for img in images]).float()
            batch = batch.to(config["device"])

            for i in range(batch.shape[0]):
                for c in range(3):
                    batch[i, c] = (batch[i, c] - config["normalize_mean"][c]) / config["normalize_std"][c]

            with torch.no_grad():
                output = model(batch)
                probs = torch.nn.functional.softmax(output, dim=1)
            return probs.cpu().numpy()

        input_np = input_tensor.cpu().detach().numpy()
        input_np = np.transpose(input_np, (0, 2, 3, 1))
        mean = np.array(config["normalize_mean"])
        std = np.array(config["normalize_std"])
        input_np = std * input_np + mean
        input_np = np.clip(input_np[0], 0, 1)
        input_np_uint8 = (input_np * 255).astype(np.uint8)

        explainer = lime_image.LimeImageExplainer()
        explanation = explainer.explain_instance(
            input_np_uint8,
            batch_predict,
            top_labels=3,
            hide_color=0,
            num_samples=1000
        )

        temp, mask = explanation.get_image_and_mask(
            target_class,
            positive_only=True,
            num_features=10,
            hide_rest=False
        )

        mask_tensor = torch.tensor(mask).float().unsqueeze(0).unsqueeze(0)
        mask_tensor = mask_tensor.repeat(1, 3, 1, 1)
        return mask_tensor.to(config["device"])

    else:
        raise ValueError(f"Technique {technique} not supported")

def visualize_attributions(original_image, attributions, title, technique, save_path=None):
    original_image = original_image.cpu().detach().numpy()
    original_image = np.transpose(original_image, (0, 2, 3, 1))

    mean = np.array(config["normalize_mean"])
    std = np.array(config["normalize_std"])
    original_image = std * original_image + mean
    original_image = np.clip(original_image[0], 0, 1)

    plt.figure(figsize=(15, 5))

    plt.subplot(1, 3, 1)
    plt.imshow(original_image)
    plt.title('Original Image')
    plt.axis('off')

    plt.subplot(1, 3, 2)
    if technique in ['lime', 'gradcam']:
        attributions = attributions.cpu().detach().numpy()
        attr_sum = attributions[0, 0]
    else:
        attributions = attributions.cpu().detach().numpy()
        attributions = np.transpose(attributions, (0, 2, 3, 1))
        attr_sum = np.sum(np.abs(attributions[0]), axis=2)

    attr_sum = (attr_sum - attr_sum.min()) / (attr_sum.max() - attr_sum.min() + 1e-8)
    plt.imshow(attr_sum, cmap='jet')
    plt.title(f'{technique.replace("_", " ").title()}')
    plt.axis('off')

    plt.subplot(1, 3, 3)
    plt.imshow(original_image)
    plt.imshow(attr_sum, cmap='jet', alpha=0.5)
    plt.title(f'{technique.replace("_", " ").title()} - Overlay')
    plt.axis('off')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        # Convert and save as PIL image for Django storage
        plt.savefig(save_path.replace('.png', '_pil.png'))
        img = Image.open(save_path.replace('.png', '_pil.png'))
        img.save(save_path, 'PNG', quality=95)

    plt.close()

def generate_explainability_images(model, input_tensor, result_instance):
    techniques = ['integrated_gradients', 'gradcam', 'occlusion', 'lime']
    target_class = model(input_tensor).argmax(dim=1).item()

    for technique in techniques:
        try:
            attributions = apply_explainability(model, input_tensor, target_class, technique)
            save_path = f'media/explainability/{result_instance.id}_{technique}.png'
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            visualize_attributions(input_tensor, attributions, f"Test {result_instance.id}", technique, save_path)

            # Update TestResult with the saved image path
            if technique == 'integrated_gradients':
                result_instance.integrated_gradients_image = save_path
            elif technique == 'gradcam':
                result_instance.gradcam_image = save_path
            elif technique == 'occlusion':
                result_instance.occlusion_image = save_path
            elif technique == 'lime':
                result_instance.lime_image = save_path
            result_instance.save()
        except Exception as e:
            print(f"Error with {technique} for result {result_instance.id}: {e}")