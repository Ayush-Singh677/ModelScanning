import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torchvision.models import resnet50, ResNet50_Weights
from PIL import Image
import cv2

class AdversarialAttackDetector:
    def __init__(self, model=None, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        
        if model is None:
            self.model = resnet50(weights=ResNet50_Weights.DEFAULT)
        else:
            self.model = model
            
        self.model.to(self.device)
        self.model.eval()
        
        self.mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1).to(self.device)
        self.std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1).to(self.device)
        
        with open('imagenet_classes.txt') as f:
            self.labels = [line.strip() for line in f.readlines()]
    
    def preprocess_image(self, image_path):
        """Load and preprocess an image for the model."""
        image = Image.open(image_path).convert('RGB')
        transform = transforms.Compose([
            transforms.Resize(512),
            transforms.CenterCrop(511),
            transforms.ToTensor(),
        ])
        image_tensor = transform(image).unsqueeze(0).to(self.device)
        return image_tensor
    
    def normalize(self, image):
        """Normalize the image using ImageNet mean and std."""
        return (image - self.mean) / self.std
    
    def denormalize(self, image):
        """Convert normalized image back to original scale."""
        return image * self.std + self.mean
    
    def predict(self, image_tensor):
        """Make a prediction on the image."""
        normalized_image = self.normalize(image_tensor)
        with torch.no_grad():
            output = self.model(normalized_image)
        
        probabilities = F.softmax(output, dim=1)
        conf, pred_class = torch.max(probabilities, 1)
        
        return {
            'class_id': pred_class.item(),
            'class_name': self.labels[pred_class.item()],
            'confidence': conf.item()
        }
    
    def generate_adversarial_example(self, image_tensor, target_class=500, epsilon=0.20, alpha=0.10, iterations=200):
        """
        Generate an adversarial example using Projected Gradient Descent (PGD).
        
        Args:
            image_tensor: Input image tensor
            target_class: Target class for targeted attack (None for untargeted)
            epsilon: Maximum perturbation amount
            alpha: Step size for each iteration
            iterations: Number of PGD iterations
        """
        perturbed_image = image_tensor.clone().detach().requires_grad_(True)
        
        original_pred = self.predict(image_tensor)
        
        for i in range(iterations):
            normalized_image = self.normalize(perturbed_image)
            output = self.model(normalized_image)
            
            if target_class is None:
                loss = -F.cross_entropy(output, torch.tensor([original_pred['class_id']]).to(self.device))
            else:
                loss = F.cross_entropy(output, torch.tensor([target_class]).to(self.device))
            
            self.model.zero_grad()
            loss.backward()
            
            with torch.no_grad():
                perturbed_image.data = perturbed_image.data + alpha * perturbed_image.grad.sign()
                
                delta = perturbed_image.data - image_tensor.data
                delta = torch.clamp(delta, -epsilon, epsilon)
                perturbed_image.data = image_tensor.data + delta
                
                perturbed_image.data = torch.clamp(perturbed_image.data, 0, 1)
            
            perturbed_image.grad.zero_()
            
            if i % 10 == 0:
                adv_pred = self.predict(perturbed_image)
                if adv_pred['class_id'] != original_pred['class_id']:
                    print(f"Attack successful at iteration {i}")
                    break
        
        return perturbed_image
    
    def apply_spatial_smoothing(self, image_tensor, kernel_size=5):
        """Apply spatial smoothing as a defense mechanism."""
        image_np = image_tensor.squeeze(0).permute(1, 2, 0).detach().cpu().numpy()
        
        smoothed_image = cv2.GaussianBlur(image_np, (kernel_size, kernel_size), 0)
        
        smoothed_tensor = torch.from_numpy(smoothed_image).permute(2, 0, 1).unsqueeze(0).to(self.device)
        return smoothed_tensor
    
    def compute_perturbation(self, original, adversarial):
        """Compute and visualize the perturbation."""
        return torch.abs(adversarial - original)
    
    def evaluate_model_robustness(self, test_images, perturbation_types=['pgd'], epsilons=[0.2, 0.4, 0.5]):
        """
        Evaluate model robustness against various perturbations.
        
        Args:
            test_images: List of image paths
            perturbation_types: List of perturbation methods to try
            epsilons: List of perturbation strengths to test
        """
        results = {}
        
        for image_path in test_images:
            image_tensor = self.preprocess_image(image_path)
            original_pred = self.predict(image_tensor)
            
            image_results = {
                'original': original_pred,
                'perturbations': {}
            }
            
            for p_type in perturbation_types:
                for eps in epsilons:
                    if p_type == 'pgd':
                        adv_image = self.generate_adversarial_example(image_tensor, epsilon=eps)
                        adv_pred = self.predict(adv_image)
                        
                        defended_image = self.apply_spatial_smoothing(adv_image)
                        defended_pred = self.predict(defended_image)
                        
                        image_results['perturbations'][f'{p_type}_{eps}'] = {
                            'adversarial': adv_pred,
                            'defended': defended_pred,
                            'success': adv_pred['class_id'] != original_pred['class_id'],
                            'defense_success': defended_pred['class_id'] == original_pred['class_id']
                        }
            
            results[image_path] = image_results
        
        return results
    
    def visualize_results(self, image_path, epsilon=0.20):
        """
        Visualize original, adversarial, and defended images with predictions.
        """
        image_tensor = self.preprocess_image(image_path)
    
        original_pred = self.predict(image_tensor)
    
        adv_image = self.generate_adversarial_example(image_tensor, epsilon=epsilon)
        adv_pred = self.predict(adv_image)
    
        defended_image = self.apply_spatial_smoothing(adv_image)
        defended_pred = self.predict(defended_image)
    
        perturbation = self.compute_perturbation(image_tensor, adv_image)
    
        orig_img = np.clip(self.denormalize(image_tensor).squeeze(0).permute(1, 2, 0).cpu().numpy(), 0, 1)
        adv_img = np.clip(self.denormalize(adv_image).squeeze(0).permute(1, 2, 0).detach().cpu().numpy(), 0, 1)
        def_img = np.clip(self.denormalize(defended_image).squeeze(0).permute(1, 2, 0).cpu().numpy(), 0, 1)
    
        pert_img = perturbation.squeeze(0).permute(1, 2, 0).detach().cpu().numpy()
        pert_img = (pert_img - pert_img.min()) / (pert_img.max() - pert_img.min())
        
        fig, axs = plt.subplots(1, 4, figsize=(20, 5))
        
        axs[0].imshow(orig_img)
        axs[0].set_title(f"Original: {original_pred['class_name']}\nConfidence: {original_pred['confidence']:.2f}")
        axs[0].axis('off')
        
        axs[1].imshow(pert_img)
        axs[1].set_title(f"Perturbation (ε={epsilon})")
        axs[1].axis('off')
        
        axs[2].imshow(adv_img)
        axs[2].set_title(f"Adversarial: {adv_pred['class_name']}\nConfidence: {adv_pred['confidence']:.2f}")
        axs[2].axis('off')
        
        axs[3].imshow(def_img)
        axs[3].set_title(f"Defended: {defended_pred['class_name']}\nConfidence: {defended_pred['confidence']:.2f}")
        axs[3].axis('off')
        
        plt.tight_layout()
        plt.show()
        
        print("Original classification:", original_pred['class_name'], f"({original_pred['confidence']:.2f})")
        print("Adversarial classification:", adv_pred['class_name'], f"({adv_pred['confidence']:.2f})")
        print("Defended classification:", defended_pred['class_name'], f"({defended_pred['confidence']:.2f})")
        
        if adv_pred['class_id'] != original_pred['class_id']:
            print("Attack successful!")
        else:
            print("Attack failed.")
            
        if defended_pred['class_id'] == original_pred['class_id']:
            print("Defense successful!")
        else:
            print("Defense failed.")

def main():
    detector = AdversarialAttackDetector()
    
    image_path = 'elephant.png'
    
    detector.visualize_results(image_path, epsilon=0.20)
    
    test_images = ['gazelle.png', 'zebra.png', 'elephant.png']
    results = detector.evaluate_model_robustness(test_images)
    
    for img_path, result in results.items():
        print(f"\nImage: {img_path}")
        print(f"Original class: {result['original']['class_name']}")
        
        for pert_name, pert_result in result['perturbations'].items():
            print(f"  {pert_name}:")
            print(f"    Adversarial class: {pert_result['adversarial']['class_name']}")
            print(f"    Attack success: {pert_result['success']}")
            print(f"    Defended class: {pert_result['defended']['class_name']}")
            print(f"    Defense success: {pert_result['defense_success']}")

if __name__ == "__main__":
    main()