from datetime import datetime
import pytz
from dateutil import parser
import os
import logging
import json
import torch
from torch.cuda.amp import autocast
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
from tqdm.auto import tqdm
from typing import Union, Optional, Any, Tuple, Dict
from pathlib import Path
import shutil
import tempfile
import re
from pprint import pformat

from pathlib import Path
from datasets import Dataset
import faiss
import yaml
from box import Box
import yaml
from transformers import PreTrainedModel, PreTrainedTokenizer
from accelerate import Accelerator
from peft import PeftModelForCausalLM

from peft import PeftModel
from huggingface_hub import list_repo_files, upload_file, upload_folder, hf_hub_download

def setup_logger(log_dir='logs', log_level=logging.INFO, enable_logging=False):
    if not enable_logging:
        logging.disable(logging.CRITICAL)
        return None

    # Get the logger instance
    logger = logging.getLogger(__name__)

    # Check if the logger already has handlers (to prevent duplicate logging)
    if not logger.hasHandlers():
        # Ensure the log directory exists
        os.makedirs(log_dir, exist_ok=True)
        
        # Create a unique log file name
        log_file = os.path.join(log_dir, f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
        
        # Set up basic configuration for logging
        logging.basicConfig(
            level=log_level,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()  # This will print logs to console as well
            ]
        )
        
        logger.info(f"Logging initialized. Log file: {log_file}")
    
    return logger

logger = setup_logger(enable_logging=True)
# Here to prevent circular import with rag.py
from .rag import RAGSequence

def log_structured_data(logger, data, indent=2):
    """
    Cleanly log structured data with proper formatting and indentation.
    
    Args:
        logger: Logger instance
        data: Data to log (dict, list etc)
        indent: Indentation level for formatting
    """
    if isinstance(data, (dict, list)):
        # For complex nested structures
        formatted = pformat(data, indent=indent)
        logger.info(f"\n=== Structured Data ===\n{formatted}")
    else:
        # For simple JSON-serializable data
        formatted = json.dumps(data, indent=indent)
        logger.info(f"\n=== JSON Data ===\n{formatted}")


# utils for processing documents
def parse_date(date_str):
    if not date_str or date_str == "1":
        return None
    try:
        if len(date_str) <= 4 and date_str.isdigit():  # Year only
            return datetime(int(date_str), 1, 1, tzinfo=pytz.UTC)
        else:  # 3-letter-month Year
            return datetime.strptime(date_str, "%b %Y").replace(tzinfo=pytz.UTC)
    except ValueError:
        try:
            return parser.parse(date_str).replace(tzinfo=pytz.UTC)
        except (ValueError, TypeError):
            print(f"Warning: Unable to parse date string: {date_str}")
            return None       
        
def load_json(file_path):
    with open(file_path, 'r') as f:
        return json.load(f)

def split_text(text: str, n=100, character=" "):
    text = text.split(character)
    return [character.join(text[i : i + n]).strip() for i in range(0, len(text), n)]

def load_experiment_config(repo_id: str, experiment_path: str) -> Box:
    """
    Download and load experiment config from hub.
    
    Args:
        repo_id: HuggingFace repo ID
        experiment_path: Path to experiment directory
    
    Returns:
        Box: Loaded config object
    """
    with tempfile.TemporaryDirectory() as temp_dir:
        try:
            # hardcoded config.yml
            config_path = os.path.join(experiment_path, "config.yml")
            local_config = hf_hub_download(
                repo_id=repo_id,
                filename=config_path,
                local_dir=temp_dir
            )
            
            with open(local_config, 'r') as f:
                config = Box(yaml.safe_load(f))
                
            return config
            
        except Exception as e:
            logger.error(f"Error loading config from hub: {e}")
            raise


class HubUploader:
    """Handler for uploading content to HuggingFace Hub with proper path handling."""
    
    @staticmethod
    def upload_to_hub(
        content: Union[Dict, Any],
        path_in_repo: str,
        repo_id: str,
        commit_message: Optional[str] = None
    ) -> None:
        """
        Upload content to HuggingFace Hub handling nested structures and maintaining paths.
        
        Args:
            content: Content to upload (Dict or supported type)
            path_in_repo: Base path in repo
            repo_id: HuggingFace repo ID
            commit_message: Optional commit message
        """
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_base = Path(temp_dir)
            
            try:
                # Since Box inherits from dict
                if isinstance(content, dict) and not isinstance(content, Box):
                    # For dict content, save each component and upload entire folder
                    for key, value in content.items():
                        HubUploader._save_with_path(
                            value, 
                            temp_base, 
                            key,
                        )
                    
                    # After creating desired dir stucture, upload dir
                    upload_folder(
                        folder_path=str(temp_base),
                        path_in_repo=path_in_repo,
                        repo_id=repo_id,
                        commit_message=commit_message or f"Upload to {path_in_repo}"
                    )
                else:
                    # For single objects, use direct save and upload
                    save_path = HubUploader._save_content(content, temp_base)
                    if save_path.is_dir():
                        upload_folder(
                            folder_path=str(save_path),
                            path_in_repo=path_in_repo,
                            repo_id=repo_id,
                            commit_message=commit_message or f"Upload to {path_in_repo}"
                        )
                    else:
                        upload_file(
                            path_or_fileobj=str(save_path),
                            path_in_repo=path_in_repo,
                            repo_id=repo_id,
                            commit_message=commit_message or f"Upload to {path_in_repo}"
                        )
                
                logger.info(f"Successfully uploaded to {repo_id}/{path_in_repo}")
                
            except Exception as e:
                logger.error(f"Failed to upload content: {e}")
                raise

    @staticmethod
    def _save_with_path(content: Any, base_path: Path, key: str) -> None:
        """Save content maintaining directory structure and handling nested dicts."""
        if isinstance(content, dict):
            # Create directory for nested dict
            current_path = base_path / key
            os.makedirs(current_path, exist_ok=True)
            
            # Handle special cases for training state       
            
            if key == 'training_state':
                for state_key, state_value in content.items():
                    if state_key == 'optimizer':
                        try:
                            # Handle Accelerate's AcceleratedOptimizer wrapper
                            if hasattr(state_value, 'optimizer'):
                                # Extract the inner optimizer
                                inner_optimizer = state_value.optimizer
                                logger.info(f"Detected AcceleratedOptimizer wrapping: {type(inner_optimizer)}")
                                
                                # Build complete state
                                full_state = {
                                    'param_groups': state_value.param_groups,
                                    'state': state_value.state,
                                    # Include scaler state if present
                                    'scaler': state_value.scaler.state_dict() if hasattr(state_value, 'scaler') else None
                                }
                                
                                torch.save(full_state, current_path / 'optimizer.pt')
                            else:
                                # Direct saving
                                torch.save(state_value.state_dict(), current_path / 'optimizer.pt')
                                
                            logger.info(f"Optimizer state saved (size {os.path.getsize(current_path / 'optimizer.pt')/1024**2:.2f}MB)")
                        except Exception as e:
                            logger.warning(f"Error saving optimizer state: {e}")
                            # Fallback to saving a simple placeholder to avoid load errors
                            torch.save({'param_groups': [], 'state': {}}, current_path / 'optimizer.pt')
                    elif state_key == 'scheduler':
                        torch.save(state_value.state_dict(), current_path / 'scheduler.pt')
                    elif state_key == 'progress':
                        torch.save(state_value, current_path / 'progress.pt')
                    else:
                        save_path = HubUploader._save_content(state_value, current_path / state_key)
            else:
                # Regular nested dict handling
                for subkey, subvalue in content.items():
                    HubUploader._save_with_path(
                        subvalue,
                        current_path,
                        subkey
                    )
        else:
            save_path = HubUploader._save_content(content, base_path / key)

    @staticmethod
    def _save_content(content: Any, path: Path) -> Path:
        """Save individual content items with appropriate methods."""
        os.makedirs(path.parent, exist_ok=True)
        
        if isinstance(content, Dataset):
            os.makedirs(path, exist_ok=True)
            content.save_to_disk(str(path))
            return path
            
        elif isinstance(content, faiss.Index):
            faiss_path = path.with_suffix('.faiss')
            faiss.write_index(content, str(faiss_path))
            return faiss_path
            
        elif isinstance(content, Box):
            yaml_path = path.with_suffix('.yml')
            with open(yaml_path, 'w') as f:
                yaml.dump(content.to_dict(), f, default_flow_style=False)
            return yaml_path
            
        elif hasattr(content, 'save_pretrained'):
            # NOTE: save vocab_size of base model as PEFT's save_pretrained() only stores adapter weights
            # and adapter_config.json which does not contain base_config.json's info
            # NEW NOTE: no need to store base_config as len(tokenizer) will suffice to resize
            os.makedirs(path, exist_ok=True)
            # if isinstance(content, PeftModelForCausalLM):
            #     base_config = content.get_base_model().config.to_dict()
            #     with open(path / "base_config.json", 'w') as f:
            #         json.dump(base_config, f)
            content.save_pretrained(path)
            return path
            
        elif isinstance(content, (Optimizer, _LRScheduler)):
            state_path = path.with_suffix('.pt')
            torch.save(content.state_dict(), state_path)
            return state_path
            
        elif isinstance(content, (dict, list)):
            data_path = path.with_suffix('.pt')
            torch.save(content, data_path)
            return data_path
            
        else:
            raise ValueError(f"Unsupported content type: {type(content)}")
    

def get_latest_checkpoint(repo_id: str, experiment_path: str) -> Tuple[str, int]:
    """
    Get the latest checkpoint path and epoch number from HuggingFace repository.
    
    Args:
        repo_id: HuggingFace repository ID
        experiment_path: Path to experiment directory
        
    Returns:
        tuple: (checkpoint_path, epoch_number)
    """
    # Match only the epoch number
    checkpoint_pattern = re.compile(rf"{experiment_path}/checkpoint-epoch-(\d+)(?:/.*)?$")
    
    # Get unique checkpoint paths by removing file-specific suffixes
    checkpoint_epochs = set()
    for file in list_repo_files(repo_id):
        if match := checkpoint_pattern.match(file):
            checkpoint_epochs.add(int(match.group(1)))
    
    if not checkpoint_epochs:
        return None, None
        
    latest_epoch = max(checkpoint_epochs)
    latest_checkpoint = f"{experiment_path}/checkpoint-epoch-{latest_epoch}"
    
    return latest_checkpoint, latest_epoch


def load_training_state(repo_id: str, checkpoint_path: str) -> Tuple[Optional[dict], Optional[dict], Optional[dict]]:
    """
    Load training state (optimizer, scheduler, progress) from checkpoint.
    
    Args:
        repo_id: HuggingFace repo ID
        checkpoint_path: Path to checkpoint directory
    
    Returns:
        Tuple of (optimizer_state, scheduler_state, progress)
    """
    def load_file(repo_id, file_path):
        """Helper function to safely download and load a checkpoint file."""
        description = os.path.basename(file_path)
        try:
            file = hf_hub_download(repo_id=repo_id, filename=file_path)
            state = torch.load(file)
            logger.info(f"Successfully loaded {description}")
            return state
        except FileNotFoundError:
            logger.warning(f"{description} not found at {file_path}. Skipping...")
        except Exception as e:
            logger.error(f"Unexpected error loading {file_path}: {e}")
        return None

    optimizer_state = load_file(repo_id, os.path.join(checkpoint_path, "optimizer.pt"))
    scheduler_state = load_file(repo_id, os.path.join(checkpoint_path, "scheduler.pt"))
    progress = load_file(repo_id, os.path.join(checkpoint_path, "progress.pt"))

    return optimizer_state, scheduler_state, progress

def log_parameter_info(model, name):
    trainable_params = sum(p.numel() for n, p in model.named_parameters() if p.requires_grad)
    frozen_params = sum(p.numel() for n, p in model.named_parameters() if not p.requires_grad)
    dtypes = set(p.dtype for p in model.parameters())
    if len(dtypes) > 1:
        count = {
            precision: sum(p.numel() for n, p in model.named_parameters() if p.dtype == precision) 
            for precision in dtypes
        }
        precision_info = (
            "\n  Per-precision parameter counts: {"
            + ", ".join(f"{k}: {v:,}" for k, v in count.items())
            + "}"
        )
    else:
        precision_info = ""

    logger.info(
        f"\n{name}:"
        f"\n  Precision set: {dtypes}"
        f"{precision_info}"
        f"\n  Trainable parameters: {trainable_params:,}"
        f"\n  Frozen parameters: {frozen_params:,}\n"
    )
 
def log_grad_norm(model):
    total_norm = 0
    for p in model.parameters():
        if p.grad is not None:
            param_norm = p.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
    total_norm = total_norm ** 0.5  
    return total_norm

def log_memory_usage(stage):
    allocated = torch.cuda.memory_allocated() / 1024**3
    reserved = torch.cuda.memory_reserved() / 1024**3
    logger.info(f"Memory {stage}: Allocated={allocated:.1f}GB, Reserved={reserved:.1f}GB")

# evaluation and early stopping
def evaluate(model, dataloader, epoch, val_count, accelerator, is_supervised):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for i, batch in enumerate(tqdm(dataloader, desc=f"Validation {epoch+1}.{val_count+1}")):
            inputs = {k: (v if is_supervised and k == 'labels' else v.to(accelerator.device)) for k, v in batch.items()}
            with accelerator.autocast():
                outputs = model(**inputs)
                if is_supervised:
                    loss = outputs
                else:
                    loss = outputs.loss
                total_loss += loss.item()
    model.to(accelerator.device)
    return total_loss / len(dataloader)


class EarlyStopping:
    def __init__(self, patience=3, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None

    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif self.best_loss - val_loss > self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False