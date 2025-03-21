#!/usr/bin/env python3
"""
Gradio app for preparing healthcare activity video datasets for fine-tuning with LLaMA Factory.
"""

import os
import json
import random
import gradio as gr
import shutil
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

def generate_detailed_response(activity_number, activity_label):
    """Generate a detailed response for the assistant instead of just a number."""
    response_templates = [
        f"The video shows activity {activity_number}: {activity_label}. I can identify this based on the specific actions and setting visible in the healthcare environment.",
        
        f"Based on my analysis, this is activity {activity_number}: {activity_label}. The healthcare professional's actions and the clinical context clearly indicate this procedure.",
        
        f"This is activity {activity_number}: {activity_label}. The video depicts the characteristic steps and equipment associated with this healthcare procedure.",
        
        f"After reviewing the video, I can confirm this is activity {activity_number}: {activity_label}. The workflow and interactions shown are typical of this healthcare process.",
        
        f"I identified activity {activity_number}: {activity_label}. The clinical setting and sequence of actions performed by the healthcare worker are consistent with this procedure."
    ]
    
    return random.choice(response_templates)

def create_balanced_dataset(data_dir, output_file, train_ratio=0.8, max_per_class=None, detailed_responses=True, progress=gr.Progress()):
    """
    Create a balanced dataset for training with LLaMA Factory.
    
    Args:
        data_dir: Directory containing activity video folders
        output_file: Output JSON file path
        train_ratio: Ratio of train:test split
        max_per_class: Maximum samples per class (if None, use all)
        detailed_responses: Whether to use detailed responses instead of just numbers
        progress: Gradio progress tracker
    
    Returns:
        Summary message of the dataset creation process
    """
    dataset = []
    
    activity_dirs = [d for d in os.listdir(data_dir) 
                     if os.path.isdir(os.path.join(data_dir, d)) and not d.startswith('.')]
    
    progress(0, desc=f"Found {len(activity_dirs)} activity directories")
    
    # Different prompt templates for variety
    prompt_templates = [
        "Watch this healthcare video and identify which activity is being performed. Choose from: 1. Arranging/checking drip bottle, 2. Bed cleaning/arrangement, 3. Blood pressure checking, 4. Patient admission, 5. Patient discharge. Explain your choice.",
        
        "As a medical professional, identify the healthcare activity in this video. Options: 1. Arranging/checking drip bottle, 2. Bed cleaning/arrangement, 3. Blood pressure checking, 4. Patient admission, 5. Patient discharge. Provide the number and explain why.",
        
        "Based on this healthcare video, which of these activities is being shown? 1. Arranging/checking drip bottle, 2. Bed cleaning/arrangement, 3. Blood pressure checking, 4. Patient admission, 5. Patient discharge. Provide your answer with a brief explanation.",
        
        "Analyze this healthcare video and determine the activity being performed. Select from: 1. Arranging/checking drip bottle, 2. Bed cleaning/arrangement, 3. Blood pressure checking, 4. Patient admission, 5. Patient discharge. Give the activity number and describe what indicates this activity."
    ]
    
    # Detailed prompt templates (more conversational)
    detailed_prompt_templates = [
        "I'm studying healthcare activities in hospitals. Can you analyze this video and tell me which activity is being performed? Choose from: 1. Arranging/checking drip bottle, 2. Bed cleaning/arrangement, 3. Blood pressure checking, 4. Patient admission, 5. Patient discharge. Please explain your reasoning.",
        
        "For my healthcare workflow analysis, I need to classify this video. Which activity is shown: 1. Arranging/checking drip bottle, 2. Bed cleaning/arrangement, 3. Blood pressure checking, 4. Patient admission, 5. Patient discharge? Why did you choose this option?",
        
        "I'm working on a hospital efficiency project. What activity is depicted in this video? Is it: 1. Arranging/checking drip bottle, 2. Bed cleaning/arrangement, 3. Blood pressure checking, 4. Patient admission, 5. Patient discharge? Explain the key indicators that helped you identify this activity."
    ]
    
    # Combine all prompt templates
    all_prompt_templates = prompt_templates + detailed_prompt_templates
    
    train_data = []
    test_data = []
    
    for idx, activity_dir in enumerate(activity_dirs):
        if activity_dir.startswith('.'):
            continue
            
        activity_path = os.path.join(data_dir, activity_dir)
        video_files = [f for f in os.listdir(activity_path) 
                       if f.lower().endswith(('.mp4', '.avi', '.mov'))]
        
        # Shuffle and limit per class if specified
        random.shuffle(video_files)
        if max_per_class and len(video_files) > max_per_class:
            video_files = video_files[:max_per_class]
            
        progress(idx/len(activity_dirs), desc=f"Processing {activity_dir}: {len(video_files)} videos")
        
        # Get activity labels
        activity_label = get_activity_label(activity_dir)
        activity_number = get_activity_label_number(activity_dir)
        
        # Split into train and test
        split_idx = int(len(video_files) * train_ratio)
        train_videos = video_files[:split_idx]
        test_videos = video_files[split_idx:]
        
        # Create training samples
        for i, video_file in enumerate(train_videos):
            progress((idx + (i/len(train_videos))/len(activity_dirs))/2, desc=f"Processing {activity_dir} train")
            video_path = os.path.join(activity_path, video_file)
            # Use absolute path instead of relative path
            abs_video_path = os.path.abspath(video_path)
            
            # Choose a random prompt template
            prompt = random.choice(all_prompt_templates)
            
            # Create assistant response (detailed or just number)
            if detailed_responses:
                assistant_response = generate_detailed_response(activity_number, activity_label)
            else:
                assistant_response = f"{activity_number}"
            
            sample = {
                "messages": [
                    {
                        "role": "user",
                        "content": f"<video>{prompt}"
                    },
                    {
                        "role": "assistant",
                        "content": assistant_response
                    }
                ],
                "videos": [
                    abs_video_path
                ]
            }
            
            train_data.append(sample)
            
        # Create test samples (using standard prompt for consistency)
        for i, video_file in enumerate(test_videos):
            progress(0.5 + (idx + (i/len(test_videos))/len(activity_dirs))/2, desc=f"Processing {activity_dir} test")
            video_path = os.path.join(activity_path, video_file)
            # Use absolute path instead of relative path
            abs_video_path = os.path.abspath(video_path)
            
            # Use a standard prompt for test data
            prompt = "Identify the healthcare activity shown in this video. Select from: 1. Arranging/checking drip bottle, 2. Bed cleaning/arrangement, 3. Blood pressure checking, 4. Patient admission, 5. Patient discharge. Explain your answer."
            
            # Create assistant response (detailed or just number)
            if detailed_responses:
                assistant_response = generate_detailed_response(activity_number, activity_label)
            else:
                assistant_response = f"{activity_number}"
            
            sample = {
                "messages": [
                    {
                        "role": "user",
                        "content": f"<video>{prompt}"
                    },
                    {
                        "role": "assistant",
                        "content": assistant_response
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
    
    summary = f"Created {len(train_data)} training samples and {len(test_data)} test samples\n"
    summary += f"Train data saved to: {train_output}\n"
    summary += f"Test data saved to: {test_output}"
    
    return summary, train_output, test_output

def preview_dataset(json_file, num_samples=5):
    """Preview a few samples from the dataset."""
    try:
        with open(json_file, 'r') as f:
            data = json.load(f)
        
        if not data:
            return "Dataset is empty"
        
        samples = random.sample(data, min(num_samples, len(data)))
        preview_text = f"Showing {len(samples)} samples from {len(data)} total entries:\n\n"
        
        for i, sample in enumerate(samples):
            preview_text += f"Sample {i+1}:\n"
            preview_text += f"User: {sample['messages'][0]['content']}\n"
            preview_text += f"Assistant: {sample['messages'][1]['content']}\n"
            preview_text += f"Video: {sample['videos'][0]}\n\n"
        
        return preview_text
    except Exception as e:
        return f"Error previewing dataset: {str(e)}"

def launch_ui():
    """Launch the Gradio UI for dataset creation."""
    with gr.Blocks(title="Healthcare Activity Dataset Creator") as app:
        gr.Markdown("# Healthcare Activity Dataset Creator for LLaMA Factory")
        gr.Markdown("Create datasets for fine-tuning vision-language models on healthcare activity recognition")
        
        with gr.Tab("Create Dataset"):
            with gr.Row():
                with gr.Column():
                    data_dir = gr.Textbox(
                        label="Data Directory", 
                        placeholder="/path/to/activity/videos",
                        value="/l/users/mukul.ranjan/video_data/activity_clips/activity_clips/"
                    )
                    
                    output_file = gr.Textbox(
                        label="Output JSON File", 
                        placeholder="/path/to/output.json",
                        value="/home/mukul.ranjan/projects/video-vlm/LLaMA-Factory/data/healthcare_activity/activity_dataset.json"
                    )
                    
                    with gr.Row():
                        train_ratio = gr.Slider(
                            label="Train/Test Split Ratio", 
                            minimum=0.5, 
                            maximum=0.9, 
                            value=0.8,
                            step=0.05
                        )
                        
                        max_per_class = gr.Number(
                            label="Max Samples Per Class (0 for all)", 
                            value=400,
                            precision=0
                        )
                    
                    with gr.Row():
                        detailed_responses = gr.Checkbox(
                            label="Use Detailed Responses (recommended)", 
                            value=True
                        )
                    
                    create_button = gr.Button("Create Dataset", variant="primary")
                
                with gr.Column():
                    output_info = gr.Textbox(label="Processing Output", lines=10)
                    train_file = gr.Textbox(label="Train File Path", visible=False)
                    test_file = gr.Textbox(label="Test File Path", visible=False)
            
            create_button.click(
                fn=create_balanced_dataset,
                inputs=[data_dir, output_file, train_ratio, max_per_class, detailed_responses],
                outputs=[output_info, train_file, test_file]
            )
        
        with gr.Tab("Preview Dataset"):
            with gr.Row():
                with gr.Column():
                    preview_file = gr.Textbox(
                        label="Dataset JSON File", 
                        placeholder="/path/to/dataset.json"
                    )
                    
                    num_samples = gr.Slider(
                        label="Number of Samples to Preview", 
                        minimum=1, 
                        maximum=20, 
                        value=5,
                        step=1
                    )
                    
                    preview_button = gr.Button("Preview Dataset")
                
                with gr.Column():
                    preview_output = gr.Textbox(label="Dataset Preview", lines=20)
            
            preview_button.click(
                fn=preview_dataset,
                inputs=[preview_file, num_samples],
                outputs=preview_output
            )
            
            # Connect train file to preview
            train_file.change(
                fn=lambda x: x,
                inputs=train_file,
                outputs=preview_file
            )
        
        gr.Markdown("## Instructions")
        gr.Markdown("""
        1. **Create Dataset Tab**:
           - Enter the path to your video directory (containing activity subdirectories)
           - Set the output JSON file path
           - Adjust train/test ratio and samples per class as needed
           - Click "Create Dataset" to generate training data
        
        2. **Preview Dataset Tab**:
           - Enter the path to a generated JSON dataset file
           - Select the number of random samples to preview
           - Click "Preview Dataset" to see sample entries
        
        The dataset format follows LLaMA Factory's vision-language instruction tuning format,
        which can be used directly with Qwen-VL models.
        """)
    
    return app

if __name__ == "__main__":
    app = launch_ui()
    app.launch(share=True)
