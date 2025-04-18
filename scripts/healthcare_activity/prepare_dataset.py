#!/usr/bin/env python3
"""
Prepare healthcare activity video dataset for fine-tuning with LLaMA Factory.
"""

import os
import json
import random
from pathlib import Path
import argparse
from tqdm import tqdm

def get_activity_label(dir_name):
    """Map directory name to activity label."""
    mapping = {
        "arranged_drip_bottle": "Arranging/checking drip bottle",
        "bed_cleaned": "Bed cleaning/arrangement",
        "bp_checked": "Blood pressure checking",
        "patient_admitted": "Patient admission",
        "patient_discharged": "Patient discharge"
    }
    return mapping.get(dir_name, "Unknown")

def get_activity_label_number(dir_name):
    """Map directory name to activity label number 1-5."""
    mapping = {
        "arranged_drip_bottle": "1",
        "bed_cleaned": "2",
        "bp_checked": "3",
        "patient_admitted": "4",
        "patient_discharged": "5"
    }
    return mapping.get(dir_name, "0")

def create_balanced_dataset(data_dir, output_file, train_ratio=0.8, max_per_class=None):
    """
    Create a balanced dataset for training with LLaMA Factory.
    
    Args:
        data_dir: Directory containing activity video folders
        output_file: Output JSON file path
        train_ratio: Ratio of train:test split
        max_per_class: Maximum samples per class (if None, use all)
    """
    dataset = []
    
    activity_dirs = [d for d in os.listdir(data_dir) 
                     if os.path.isdir(os.path.join(data_dir, d)) and not d.startswith('.')]
    
    print(f"Found {len(activity_dirs)} activity directories")
    
    # Different prompt templates for variety
    prompt_templates = [
        "Watch this healthcare video and identify which activity is being performed. Choose from: 1. Arranging/checking drip bottle, 2. Bed cleaning/arrangement, 3. Blood pressure checking, 4. Patient admission, 5. Patient discharge. Respond with the number of the correct activity.",
        
        "As a medical professional, identify the healthcare activity in this video. Options: 1. Arranging/checking drip bottle, 2. Bed cleaning/arrangement, 3. Blood pressure checking, 4. Patient admission, 5. Patient discharge. Answer with just the number.",
        
        "Based on this healthcare video, which of these activities is being shown? 1. Arranging/checking drip bottle, 2. Bed cleaning/arrangement, 3. Blood pressure checking, 4. Patient admission, 5. Patient discharge. Provide just the number.",
        
        "Analyze this healthcare video and determine the activity being performed. Select from: 1. Arranging/checking drip bottle, 2. Bed cleaning/arrangement, 3. Blood pressure checking, 4. Patient admission, 5. Patient discharge. Give only the activity number."
    ]
    
    # Detailed prompt templates (more conversational)
    detailed_prompt_templates = [
        "I'm studying healthcare activities in hospitals. Can you analyze this video and tell me which activity is being performed? Choose from: 1. Arranging/checking drip bottle, 2. Bed cleaning/arrangement, 3. Blood pressure checking, 4. Patient admission, 5. Patient discharge. Just provide the corresponding number.",
        
        "For my healthcare workflow analysis, I need to classify this video. Which activity is shown: 1. Arranging/checking drip bottle, 2. Bed cleaning/arrangement, 3. Blood pressure checking, 4. Patient admission, 5. Patient discharge? Please respond with only the number.",
        
        "I'm working on a hospital efficiency project. What activity is depicted in this video? Is it: 1. Arranging/checking drip bottle, 2. Bed cleaning/arrangement, 3. Blood pressure checking, 4. Patient admission, 5. Patient discharge? Answer with the single digit only."
    ]
    
    # Combine all prompt templates
    all_prompt_templates = prompt_templates + detailed_prompt_templates
    
    train_data = []
    test_data = []
    
    for activity_dir in activity_dirs:
        if activity_dir.startswith('.'):
            continue
            
        activity_path = os.path.join(data_dir, activity_dir)
        video_files = [f for f in os.listdir(activity_path) 
                       if f.lower().endswith(('.mp4', '.avi', '.mov'))]
        
        # Shuffle and limit per class if specified
        random.shuffle(video_files)
        if max_per_class and len(video_files) > max_per_class:
            video_files = video_files[:max_per_class]
            
        print(f"Processing {activity_dir}: {len(video_files)} videos")
        
        # Get activity labels
        activity_label = get_activity_label(activity_dir)
        activity_number = get_activity_label_number(activity_dir)
        
        # Split into train and test
        split_idx = int(len(video_files) * train_ratio)
        train_videos = video_files[:split_idx]
        test_videos = video_files[split_idx:]
        
        # Create training samples
        for video_file in tqdm(train_videos, desc=f"Processing {activity_dir} train"):
            video_path = os.path.join(activity_path, video_file)
            # Use absolute path instead of relative path
            abs_video_path = os.path.abspath(video_path)
            
            # Choose a random prompt template
            prompt = random.choice(all_prompt_templates)
            
            sample = {
                "messages": [
                    {
                        "role": "user",
                        "content": f"<video>{prompt}"
                    },
                    {
                        "role": "assistant",
                        "content": f"{activity_number}"
                    }
                ],
                "videos": [
                    abs_video_path
                ]
            }
            
            train_data.append(sample)
            
        # Create test samples (using standard prompt for consistency)
        for video_file in tqdm(test_videos, desc=f"Processing {activity_dir} test"):
            video_path = os.path.join(activity_path, video_file)
            # Use absolute path instead of relative path
            abs_video_path = os.path.abspath(video_path)
            
            # Use a standard prompt for test data
            prompt = "Identify the healthcare activity shown in this video. Select from: 1. Arranging/checking drip bottle, 2. Bed cleaning/arrangement, 3. Blood pressure checking, 4. Patient admission, 5. Patient discharge. Respond with only the number."
            
            sample = {
                "messages": [
                    {
                        "role": "user",
                        "content": f"<video>{prompt}"
                    },
                    {
                        "role": "assistant",
                        "content": f"{activity_number}"
                    }
                ],
                "videos": [
                    abs_video_path
                ]
            }
            
            test_data.append(sample)
    
    # Save train and test datasets
    output_dir = os.path.dirname(output_file)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    train_output = output_file.replace('.json', '_train.json')
    test_output = output_file.replace('.json', '_test.json')
    
    with open(train_output, 'w') as f:
        json.dump(train_data, f, indent=2)
    
    with open(test_output, 'w') as f:
        json.dump(test_data, f, indent=2)
    
    print(f"Created {len(train_data)} training samples and {len(test_data)} test samples")
    print(f"Train data saved to: {train_output}")
    print(f"Test data saved to: {test_output}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare video activity dataset for LLaMA Factory")
    parser.add_argument("--data_dir", type=str, 
                       default="/l/users/mukul.ranjan/video_data/activity_clips/activity_clips/",
                       help="Directory containing activity video folders")
    parser.add_argument("--output_file", type=str, 
                       default="/home/mukul.ranjan/projects/video-vlm/LLaMA-Factory/data/healthcare_activity/activity_dataset_new.json",
                       help="Output JSON file path")
    parser.add_argument("--train_ratio", type=float, default=0.8,
                       help="Ratio of train:test split")
    parser.add_argument("--max_per_class", type=int, default=None,
                       help="Maximum samples per class")
    
    args = parser.parse_args()
    
    create_balanced_dataset(
        args.data_dir,
        args.output_file,
        args.train_ratio,
        args.max_per_class
    )
