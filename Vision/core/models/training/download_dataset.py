#!/usr/bin/env python3

from roboflow import Roboflow

def download_dataset(api_key, workspace, project, version):
    """
    Download dataset from Roboflow.
    
    Args:
        api_key: Your Roboflow API key
        workspace: Roboflow workspace name
        project: Roboflow project name
        version: Dataset version number
    
    Returns:
        Path to downloaded dataset
    """
    print("Downloading dataset from Roboflow...")
    
    # Initialize Roboflow
    rf = Roboflow(api_key=api_key)
    project_obj = rf.workspace(workspace).project(project)
    version_obj = project_obj.version(version)
    dataset = version_obj.download("yolov8")
    
    print(f"Dataset downloaded to: {dataset.location}")
    print("Ready for training!")
    
    return dataset.location

if __name__ == "__main__":
    # Replace with your actual values
    dataset_path = download_dataset(
        api_key="Y0G66vmOV2YEL42JtXAX",
        workspace="coin-counter-sskgh", 
        project="chess-bjirs",
        version=7
    )
    print(f"\nDataset path: {dataset_path}")