#!/usr/bin/env python3
"""
Evaluate a fine-tuned Qwen 2.5 VL model on the healthcare activity recognition test set.
"""

import os
import json
import argparse
import numpy as np
from typing import List, Dict, Any, Tuple
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import torch
from PIL import Image
import cv2
import tempfile
from pathlib import Path
import re

def load_model_and_tokenizer(model_path: str, lora: bool = True) -> Tuple[Any, Any]:
    """Load the model and tokenizer."""
    print(f"Loading base model from {model_path}")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True
    )
    
    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.float16
    )
    
    # Apply LoRA weights if needed
    if lora and os.path.exists(os.path.join(model_path, "adapter_config.json")):
        print("Loading LoRA adapter")
        model = PeftModel.from_pretrained(model, model_path)
    
    return model, tokenizer

def extract_frame_from_video(video_path: str, frame_idx: int = None) -> np.ndarray:
    """Extract a representative frame from a video file."""
    cap = cv2.VideoCapture(video_path)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    if frame_count == 0:
        raise ValueError(f"Could not read frames from {video_path}")
    
    # If no frame index is provided, take the middle frame
    if frame_idx is None:
        frame_idx = frame_count // 2
    
    # Ensure frame_idx is within bounds
    frame_idx = min(max(0, frame_idx), frame_count - 1)
    
    # Set the position and read the frame
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ret, frame = cap.read()
    cap.release()
    
    if not ret:
        raise ValueError(f"Could not read frame {frame_idx} from {video_path}")
    
    # Convert BGR to RGB
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    return frame

def predict_activity(model, tokenizer, video_path: str, prompt: str) -> str:
    """Generate a prediction for the video."""
    try:
        # Extract a frame from the video
        frame = extract_frame_from_video(video_path)
        pil_image = Image.fromarray(frame)
        
        # Prepare the prompt with video placeholder
        if "<video>" not in prompt:
            prompt = f"<video>{prompt}"
        
        # Generate the response
        response = model.chat(tokenizer, query=prompt, history=[], images=[pil_image])
        return response
    except Exception as e:
        return f"Error processing video: {str(e)}"

def extract_activity_number(text: str) -> int:
    """Extract the activity number (1-5) from the model's response."""
    # Simple pattern to find the first occurrence of a number 1-5
    match = re.search(r'activity\s*([1-5])|([1-5])\s*[:\.)]', text)
    if match:
        # Return the first captured group that matches
        return int(match.group(1) or match.group(2))
    
    # If no match with context, just look for any standalone digit 1-5
    match = re.search(r'\b[1-5]\b', text)
    if match:
        return int(match.group(0))
    
    return 0  # No valid number found

def evaluate_model(model, tokenizer, test_data_path: str) -> Dict[str, float]:
    """Evaluate the model on the test dataset."""
    # Load test data
    with open(test_data_path, 'r') as f:
        test_data = json.load(f)
    
    print(f"Loaded {len(test_data)} test samples")
    
    # Define evaluation metrics
    correct = 0
    total = 0
    results = []
    
    # Process each sample
    for sample in tqdm(test_data):
        user_prompt = sample['messages'][0]['content']
        # Extract the numeric part of the response (ground truth)
        gt_response = sample['messages'][1]['content']
        video_path = sample['videos'][0]
        
        # For ground truth, extract the activity number
        gt_number = extract_activity_number(gt_response)
        if gt_number == 0:
            # If extraction failed, use the first digit in the response
            for char in gt_response:
                if char.isdigit() and 1 <= int(char) <= 5:
                    gt_number = int(char)
                    break
        
        # Skip if we couldn't determine ground truth
        if gt_number == 0:
            print(f"Warning: Could not determine ground truth for sample: {gt_response}")
            continue
        
        # Make prediction
        pred_response = predict_activity(model, tokenizer, video_path, user_prompt)
        pred_number = extract_activity_number(pred_response)
        
        # Record result
        is_correct = (pred_number == gt_number)
        if is_correct:
            correct += 1
        total += 1
        
        # Save detailed result
        results.append({
            'video': video_path,
            'prompt': user_prompt,
            'ground_truth': gt_number,
            'prediction': pred_number,
            'pred_response': pred_response,
            'is_correct': is_correct
        })
        
        # Print progress occasionally
        if total % 10 == 0:
            print(f"Processed {total} samples, accuracy so far: {correct/total:.4f}")
    
    # Calculate accuracy
    accuracy = correct / total if total > 0 else 0
    
    # Save detailed results
    output_dir = os.path.dirname(test_data_path)
    results_path = os.path.join(output_dir, "evaluation_results.json")
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    return {
        'accuracy': accuracy,
        'correct': correct,
        'total': total,
        'results_saved_to': results_path
    }

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate fine-tuned Qwen-VL model on healthcare activity recognition")
    parser.add_argument("--model_path", type=str, required=True,
                       help="Path to the fine-tuned model directory")
    parser.add_argument("--test_data", type=str, 
                       default="/home/mukul.ranjan/projects/video-vlm/LLaMA-Factory/data/healthcare_activity/activity_dataset_enhanced_test.json",
                       help="Path to the test dataset JSON file")
    parser.add_argument("--lora", action="store_true", default=True,
                       help="Whether the model uses LoRA fine-tuning")
    
    args = parser.parse_args()
    
    # Load model and tokenizer
    model, tokenizer = load_model_and_tokenizer(args.model_path, args.lora)
    
    # Evaluate the model
    results = evaluate_model(model, tokenizer, args.test_data)
    
    # Print results
    print("\n===== Evaluation Results =====")
    print(f"Accuracy: {results['accuracy']:.4f} ({results['correct']}/{results['total']})")
    print(f"Detailed results saved to: {results['results_saved_to']}")
