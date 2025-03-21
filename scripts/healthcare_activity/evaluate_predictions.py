#!/usr/bin/env python3
"""
Evaluate the performance of the fine-tuned Qwen2.5-VL model on healthcare activity classification.
This script analyzes prediction outputs from LLaMA-Factory and computes accuracy metrics.
"""

import json
import os
import re
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
from collections import Counter
from pathlib import Path

def load_jsonl(file_path):
    """Load data from a JSONL file."""
    with open(file_path, 'r', encoding='utf-8') as f:
        return [json.loads(line) for line in f]

def load_json(file_path):
    """Load data from a JSON file."""
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def extract_activity_number(text):
    """Extract the activity number (1-5) from model output text."""
    # First, try to find a standalone digit 1-5
    match = re.search(r'\b[1-5]\b', text)
    if match:
        return match.group(0)
    
    # If not found, look for any digit 1-5 anywhere in the text
    for char in text:
        if char in "12345":
            return char
    
    # If still not found, return None
    return None

def get_activity_name(activity_number):
    """Map activity number to activity name."""
    mapping = {
        "1": "Arranging/checking drip bottle",
        "2": "Bed cleaning/arrangement",
        "3": "Blood pressure checking",
        "4": "Patient admission",
        "5": "Patient discharge"
    }
    return mapping.get(activity_number, "Unknown")

def plot_confusion_matrix(cm, classes, output_path):
    """Plot and save confusion matrix."""
    plt.figure(figsize=(10, 8))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    
    # Add text annotations
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], 'd'),
                     ha="center", va="center",
                     color="white" if cm[i, j] > thresh else "black")
    
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig(output_path)
    print(f"Confusion matrix saved to {output_path}")
    plt.close()

def main():
    # Parse command line arguments
    import argparse
    parser = argparse.ArgumentParser(description='Evaluate model predictions on healthcare activity classification')
    parser.add_argument('--predictions', type=str, required=True,
                        help='Path to the JSONL file containing model predictions')
    parser.add_argument('--test-data', type=str, required=True,
                        help='Path to the JSON file containing test data')
    parser.add_argument('--output-dir', type=str, default='./evaluation_results',
                        help='Directory to save evaluation results')
    args = parser.parse_args()
    
    predictions_path = args.predictions
    test_data_path = args.test_data
    output_dir = Path(args.output_dir)
    
    # Create output directory if it doesn't exist
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # Load data
    print(f"Loading predictions from {predictions_path}")
    predictions = load_jsonl(predictions_path)
    
    print(f"Loading test data from {test_data_path}")
    test_data = load_json(test_data_path)
    
    # Validate data lengths
    print(f"Found {len(predictions)} predictions and {len(test_data)} test examples")
    if len(predictions) != len(test_data):
        print(f"WARNING: Number of predictions ({len(predictions)}) does not match number of test examples ({len(test_data)})")
        min_len = min(len(predictions), len(test_data))
        predictions = predictions[:min_len]
        test_data = test_data[:min_len]
        print(f"Truncating to {min_len} examples for evaluation")
    
    # Extract ground truth and predictions
    ground_truths = []
    predicted_labels = []
    results = []
    errors = []
    
    for i, (pred, test_item) in enumerate(zip(predictions, test_data)):
        # Extract ground truth from test data
        ground_truth = test_item["messages"][1]["content"].strip()
        ground_truths.append(ground_truth)
        
        # Extract prediction directly from the predict field in the JSONL
        predicted_label = pred.get("predict", "").strip()
        
        # Handle cases where prediction is empty or invalid
        if not predicted_label or predicted_label not in "12345":
            predicted_label = "0"  # Use "0" to indicate failure to extract a valid label
            print(f"WARNING: Invalid prediction: {predicted_label}")
        
        predicted_labels.append(predicted_label)
        
        # Record result
        is_correct = predicted_label == ground_truth
        result = {
            "example_id": i,
            "ground_truth": ground_truth,
            "ground_truth_activity": get_activity_name(ground_truth),
            "prediction": predicted_label,
            "prediction_activity": get_activity_name(predicted_label),
            "model_output": pred.get("predict", ""),  # Use the raw predict field
            "correct": is_correct,
            "video_path": test_item["videos"][0] if "videos" in test_item else "Unknown"
        }
        results.append(result)
        
        # Record errors for analysis
        if not is_correct:
            errors.append(result)
    
    # Calculate overall accuracy
    correct_count = sum(1 for r in results if r["correct"])
    accuracy = correct_count / len(results) * 100
    print(f"\nOverall accuracy: {accuracy:.2f}% ({correct_count}/{len(results)})")
    
    # Calculate per-class accuracy
    class_counts = Counter(ground_truths)
    class_correct = {}
    for r in results:
        gt = r["ground_truth"]
        if gt not in class_correct:
            class_correct[gt] = 0
        if r["correct"]:
            class_correct[gt] += 1
    
    print("\nPer-class accuracy:")
    for cls in sorted(class_counts.keys()):
        accuracy = class_correct.get(cls, 0) / class_counts[cls] * 100
        activity_name = get_activity_name(cls)
        print(f"Class {cls} ({activity_name}): {accuracy:.2f}% ({class_correct.get(cls, 0)}/{class_counts[cls]})")
    
    # Create confusion matrix
    labels = sorted(list(set(ground_truths + predicted_labels)))
    if "0" in labels:  # Remove "0" if it's only in predictions, not in ground truth
        if "0" not in ground_truths:
            labels.remove("0")
    
    cm = confusion_matrix(ground_truths, predicted_labels, labels=labels)
    
    # Get class names for visualization
    class_names = [f"{label} - {get_activity_name(label)}" for label in labels]
    
    # Generate classification report
    report = classification_report(ground_truths, predicted_labels, labels=labels, target_names=class_names, digits=3)
    print("\nClassification Report:")
    print(report)
    
    # Plot confusion matrix
    cm_path = output_dir / "confusion_matrix.png"
    plot_confusion_matrix(cm, class_names, cm_path)
    
    # Save detailed results to file
    results_path = output_dir / "evaluation_results.json"
    with open(results_path, 'w', encoding='utf-8') as f:
        json.dump({
            "overall_accuracy": accuracy,
            "per_class_accuracy": {cls: class_correct.get(cls, 0) / class_counts[cls] * 100 for cls in class_counts},
            "results": results,
            "errors": errors,
            "classification_report": report
        }, f, indent=2)
    
    print(f"\nDetailed evaluation results saved to {results_path}")
    print(f"Found {len(errors)} errors out of {len(results)} examples")
    
    # Print some example errors for debugging
    if errors:
        print("\nExample errors:")
        for i, err in enumerate(errors[:5]):  # Show first 5 errors
            print(f"Example {err['example_id']}:")
            print(f"  Video: {err['video_path']}")
            print(f"  True: {err['ground_truth']} ({err['ground_truth_activity']})")
            print(f"  Pred: {err['prediction']} ({err['prediction_activity']})")
            print(f"  Model output: {err['model_output'][:100]}...")
            print()

if __name__ == "__main__":
    main()
