import torch
import torch.nn as nn
from torchvision import models, transforms
import numpy as np
from pathlib import Path
import json
from typing import Dict, List
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
import cv2


class DamageClassifier:
    def __init__(self, model_path: str = None):
        """Initialize the damage classification model."""
        # Define damage categories
        self.categories = [
            'structural_crack',
            'water_damage',
            'roof_damage',
            'broken_windows',
            'foundation_issues'
        ]
        
        # Load pre-trained ResNet model
        self.model = models.resnet50(pretrained=True)
        # Modify final layer for our classification task
        num_features = self.model.fc.in_features
        self.model.fc = nn.Linear(num_features, len(self.categories))
        
        # Define image transformations
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
        ])
        
        if model_path and Path(model_path).exists():
            self.model.load_state_dict(torch.load(model_path))
        
        self.model.eval()
    
    def process_image(self, image: np.ndarray) -> Dict[str, float]:
        """
        Process a single image to detect damage types.
        
        Args:
            image: Input image as numpy array
            
        Returns:
            Dictionary with confidence scores for each damage type
        """
        # Transform image
        image_tensor = self.transform(image)
        image_tensor = image_tensor.unsqueeze(0)  # Add batch dimension
        
        # Get predictions
        with torch.no_grad():
            outputs = self.model(image_tensor)
            probabilities = torch.sigmoid(outputs)[0]  # Use sigmoid for multi-label
        
        # Create results dictionary
        results = {}
        for category, probability in zip(self.categories, probabilities):
            results[category] = float(probability)
        
        return results


class ExperimentEvaluator:
    def __init__(self, classifier: DamageClassifier):
        """Initialize evaluator with classifier model."""
        self.classifier = classifier
        
    def load_human_annotations(self, annotation_file: str) -> Dict:
        """Load human-labeled ground truth annotations."""
        with open(annotation_file, 'r') as f:
            return json.load(f)
    
    def evaluate_image(self, 
                      image_path: str, 
                      human_annotations: List[str],
                      threshold: float = 0.5) -> Dict[str, float]:
        """
        Evaluate single image against human annotations.
        
        Args:
            image_path: Path to image
            human_annotations: List of damage types present in the image
            threshold: Confidence threshold for positive prediction
            
        Returns:
            Dictionary with evaluation metrics
        """
        # Read image
        print(f"Loading image from: {image_path}")
        image = cv2.imread(image_path)
        if image is None:
            print(f"Failed to load image: {image_path}")
            return {}
        print(f"Image shape: {image.shape}")
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Get model predictions
        predictions = self.classifier.process_image(image)
        
        # Convert to binary predictions using threshold
        binary_preds = {
            category: (score >= threshold)
            for category, score in predictions.items()
        }
        
        # Create ground truth binary vector
        binary_truth = {
            category: (category in human_annotations)
            for category in self.classifier.categories
        }
        
        # Calculate metrics
        metrics = {}
        for category in self.classifier.categories:
            metrics[category] = {
                'predicted': binary_preds[category],
                'actual': binary_truth[category],
                'confidence': predictions[category]
            }
        
        return metrics
    
    def run_experiment(self, 
                      test_images_dir: str, 
                      annotations_file: str,
                      threshold: float = 0.5) -> Dict:
        """
        Run full experiment on test dataset.
        
        Args:
            test_images_dir: Directory containing test images
            annotations_file: Path to human annotations file
            threshold: Confidence threshold for positive prediction
            
        Returns:
            Dictionary containing overall experiment results
        """
        human_annotations = self.load_human_annotations(annotations_file)
        images = list(Path(test_images_dir).glob('*.jpg'))
        
        all_predictions = []
        all_truth = []
        
        # Collect results for all images
        metrics_dict = {}
        for image_path in images:
            image_id = image_path.stem
            if image_id in human_annotations:
                metrics = self.evaluate_image(
                    str(image_path),
                    human_annotations[image_id],
                    threshold
                )
                metrics_dict[image_id] = metrics
                
                # Append results for each category
                for category in self.classifier.categories:
                    all_predictions.append(metrics[category]['predicted'])
                    all_truth.append(metrics[category]['actual'])
        
        # Calculate overall metrics
        precision, recall, f1, _ = precision_recall_fscore_support(
            all_truth, all_predictions, average='weighted'
        )
        accuracy = accuracy_score(all_truth, all_predictions)
        
        per_category = {}
        for category in self.classifier.categories:
            cat_preds = [m[category]['predicted'] for m in metrics_dict.values()]
            cat_truth = [m[category]['actual'] for m in metrics_dict.values()]
            p, r, f, _ = precision_recall_fscore_support(
                cat_truth, cat_preds, average='binary'
            )
            per_category[category] = {
                'precision': p,
                'recall': r,
                'f1_score': f
            }
        
        return {
            'overall': {
                'precision': precision,
                'recall': recall,
                'f1_score': f1,
                'accuracy': accuracy
            },
            'per_category': per_category
        }


def main():
    # Initialize classifier and evaluator
    classifier = DamageClassifier()
    evaluator = ExperimentEvaluator(classifier)
    
    # Run experiment
    results = evaluator.run_experiment(
        test_images_dir='images',
        annotations_file='annotations.json',
        threshold=0.5
    )
    
    # Print results
    print("\nOverall Results:")
    print(f"Accuracy: {results['overall']['accuracy']:.3f}")
    print(f"Precision: {results['overall']['precision']:.3f}")
    print(f"Recall: {results['overall']['recall']:.3f}")
    print(f"F1 Score: {results['overall']['f1_score']:.3f}")
    
    print("\nPer-Category Results:")
    for category, metrics in results['per_category'].items():
        print(f"\n{category}:")
        print(f"  Precision: {metrics['precision']:.3f}")
        print(f"  Recall: {metrics['recall']:.3f}")
        print(f"  F1 Score: {metrics['f1_score']:.3f}")

if __name__ == "__main__":
    main()