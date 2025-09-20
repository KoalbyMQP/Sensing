#!/usr/bin/env python3

import json
import depthai as dai
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any


class DepthAIModelManager:
    """
    Basic model manager for DepthAI models.
    
    This class provides essential functionality for loading and managing
    neural network models that run on OAK-D cameras.
    """
    
    def __init__(self, models_base_path: str = "models"):
        """
        Initialize the model manager.
        
        Args:
            models_base_path: Base directory containing model files
        """
        self.models_path = Path(models_base_path)
        self.models_path.mkdir(parents=True, exist_ok=True)
        
        # Store loaded models
        self.loaded_models = {}
        
        # Default model configuration
        self.default_config = {
            "nn_config": {
                "output_format": "detection",
                "confidence_threshold": 0.5,
                "input_size": "416x416"
            },
            "mappings": {
                "labels": []
            }
        }
    
    def load_model_config(self, model_name: str) -> Optional[Dict]:
        """
        Load model configuration from JSON file.
        
        Args:
            model_name: Name of the model (should match directory name)
            
        Returns:
            Model configuration dictionary or None if failed
        """
        try:
            config_path = self.models_path / model_name / f"{model_name}.json"
            
            if not config_path.exists():
                print(f"Config file not found: {config_path}")
                return None
            
            with open(config_path, 'r') as f:
                config = json.load(f)
            
            print(f"Loaded config for {model_name}")
            return config
            
        except Exception as e:
            print(f"Failed to load config for {model_name}: {e}")
            return None
    
    def find_model_blob(self, model_name: str) -> Optional[Path]:
        """
        Find the blob file for a model.
        
        Args:
            model_name: Name of the model
            
        Returns:
            Path to blob file or None if not found
        """
        model_dir = self.models_path / model_name
        
        if not model_dir.exists():
            print(f"Model directory not found: {model_dir}")
            return None
        
        # Look for blob files with common extensions
        blob_extensions = ['.blob', '.bin']
        
        for ext in blob_extensions:
            blob_path = model_dir / f"{model_name}{ext}"
            if blob_path.exists():
                print(f"Found blob file: {blob_path}")
                return blob_path
        
        print(f"No blob file found for {model_name}")
        return None
    
    def register_model(self, model_name: str) -> bool:
        """
        Register a model by loading its configuration and blob file.
        
        Args:
            model_name: Name of the model to register
            
        Returns:
            True if successfully registered, False otherwise
        """
        try:
            # Load configuration
            config = self.load_model_config(model_name)
            if not config:
                return False
            
            # Find blob file
            blob_path = self.find_model_blob(model_name)
            if not blob_path:
                return False
            
            # Store model info
            self.loaded_models[model_name] = {
                'name': model_name,
                'config': config,
                'blob_path': blob_path,
                'registered': True,
                'created_nn_node': False
            }
            
            print(f"Successfully registered model: {model_name}")
            return True
            
        except Exception as e:
            print(f"Failed to register model {model_name}: {e}")
            return False
    
    def create_nn_node(self, pipeline: dai.Pipeline, model_name: str) -> Optional[dai.node.NeuralNetwork]:
        """
        Create a neural network node in the DepthAI pipeline.
        
        Args:
            pipeline: DepthAI pipeline object
            model_name: Name of the registered model
            
        Returns:
            Neural network node or None if failed
        """
        if model_name not in self.loaded_models:
            print(f"Model {model_name} not registered")
            return None
        
        try:
            model_info = self.loaded_models[model_name]
            blob_path = model_info['blob_path']
            
            # Create neural network node
            nn_node = pipeline.create(dai.node.NeuralNetwork)
            nn_node.setBlobPath(str(blob_path))
            
            # Mark as created
            model_info['created_nn_node'] = True
            model_info['nn_node'] = nn_node
            
            print(f"Created NN node for {model_name}")
            return nn_node
            
        except Exception as e:
            print(f"Failed to create NN node for {model_name}: {e}")
            return None
    
    def get_model_info(self, model_name: str) -> Dict:
        """
        Get information about a registered model.
        
        Args:
            model_name: Name of the model
            
        Returns:
            Dictionary with model information
        """
        if model_name not in self.loaded_models:
            return {"error": f"Model {model_name} not found"}
        
        model_info = self.loaded_models[model_name]
        
        return {
            "name": model_info['name'],
            "registered": model_info['registered'],
            "blob_path": str(model_info['blob_path']),
            "config": model_info['config'],
            "nn_node_created": model_info['created_nn_node']
        }
    
    def list_models(self) -> List[str]:
        """
        List all registered models.
        
        Returns:
            List of model names
        """
        return list(self.loaded_models.keys())

