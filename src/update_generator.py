import torch
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

from transformers import (
    AutoTokenizer, AutoModelForCausalLM, DPRQuestionEncoder,
    DataCollatorForLanguageModeling, AutoConfig, PreTrainedModel,
    get_linear_schedule_with_warmup, get_cosine_schedule_with_warmup,
    BitsAndBytesConfig
)
from box import Box
import yaml
import os
import json
import pickle
import bitsandbytes as bnb
from tqdm.auto import tqdm

import shutil
import wandb
import re
from huggingface_hub import upload_file, upload_folder, hf_hub_download, snapshot_download, list_repo_files, HfApi
import huggingface_hub
import tempfile
from typing import Tuple, List, Dict, Any, Union
import gc

from .dataloader import uft_dataloader, sft_dataloader
from .update_retriever import CustomRetriever
from .rag import RAGSequence
from .utils import (
    EarlyStopping, evaluate, load_json, setup_logger, get_latest_checkpoint, HubUploader,
    load_training_state, log_parameter_info, log_grad_norm, log_memory_usage
)
logger = setup_logger(enable_logging=True)


from peft import get_peft_config, get_peft_model, LoraConfig, TaskType, PeftModel, PeftConfig
from torch.cuda.amp import autocast, GradScaler
from accelerate import Accelerator


def resume_from_checkpoint(
    repo_id: str, 
    experiment_path: str, 
    config: Box,
    retriever_docs: List[Dict],
    accelerator: Accelerator,
) -> Tuple[Union[PreTrainedModel, RAGSequence], Any, Any, Dict]:
    """
    Resume training from latest checkpoint based on experiment type and stage. 
    Only triggers when latest checkpoint < config checkpoint. See load_or_finetune()
    
    Logic flow:
    1. For UFT/SFT only:
        - Check latest checkpoint in experiment_path
        - Resume from that point
    
    2. For combined training:
        - First check if UFT is complete by looking in "uft" subfolder
        - If UFT complete but SFT not started, load UFT checkpoint and start SFT
        - If SFT in progress, resume from latest SFT checkpoint in "sft" subfolder
        - If UFT in progress, resume from latest UFT checkpoint
        
    Returns:
        Tuple of (model, optimizer_state, scheduler_state, progress)
    """
    # try:
    exp_type = config.model.experiment.type
    
    is_supervised, is_transition = False, False
    
    # Determine which stage to resume from
    if exp_type == 'combined':
        # Check UFT completion
        uft_checkpoint, uft_epoch = get_latest_checkpoint(
            repo_id, 
            os.path.join(experiment_path, "uft")
        )
        uft_complete = (
            uft_epoch == config.model.training.unsupervised.optimization.num_epochs 
            if uft_checkpoint else False
        )
        
        # Check SFT progress
        sft_checkpoint, sft_epoch = get_latest_checkpoint(
            repo_id,
            os.path.join(experiment_path, "sft")
        )
        
        # Won't get the case when full process is completed here, handled by load_or_fintune()
        # hence not handled here
        if sft_checkpoint:  # SFT in progress
            latest_checkpoint = sft_checkpoint
            is_supervised = True
        elif uft_complete:  # UFT done, start SFT
            logger.info("UFT stage of combined complete, initiating SFT (in resume_from_checkpoint)")
            latest_checkpoint = uft_checkpoint
            is_supervised = True  # It will be supervised from now
            is_transition = True
        else:  # Resume UFT
            latest_checkpoint = uft_checkpoint
            is_supervised = False
            
    else:  # UFT/SFT only
        latest_checkpoint, latest_epoch = get_latest_checkpoint(repo_id, experiment_path)
        is_supervised = (exp_type == 'sft-only')
        
        
    # Load from latest checkpoint (intermediary epoch)
    resumed_model = load_trained_model(
        repo_id, 
        latest_checkpoint, 
        config, 
        retriever_docs, 
        accelerator,
        is_transition=is_transition,
        is_trainable=True,  # always since resuming training
        is_supervised=is_supervised
    )
    
    optimizer_state, scheduler_state, progress = None, None, None
    if not is_transition:   
        optimizer_state, scheduler_state, progress = load_training_state(
            repo_id=repo_id,
            checkpoint_path=os.path.join(latest_checkpoint, "training_state")
        )      
    
    return resumed_model, optimizer_state, scheduler_state, progress, latest_checkpoint, is_supervised


def train(
    model,
    tokenizer,
    accelerator,
    train_dataloader,
    val_dataloader,
    repo_id,
    experiment_path,
    gradient_accumulation_steps,
    learning_rate,
    num_epochs,
    eval_frequency,
    do_eval,
    weight_decay,
    warmup_ratio,
    max_grad_norm,
    do_early_stopping,
    patience,
    min_delta,
    optimizer_state=None,
    scheduler_state=None,
    progress=None,
    is_supervised=False,
    stage="",
    save_optimizer_and_scheduler=False
):
    uploader = HubUploader()
    
    total_steps = len(train_dataloader) * num_epochs // gradient_accumulation_steps
    optimizer = bnb.optim.AdamW8bit(
        filter(lambda p: p.requires_grad, model.parameters()), 
        lr=learning_rate, 
        weight_decay=weight_decay
    )
    
    trainable_params = sum(p.numel() for group in optimizer.param_groups for p in group['params'])
    logger.info(f"Optimizer will train {trainable_params:,} parameters")

    # Cosine scheduler with warmup
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, 
        num_warmup_steps=int(total_steps * warmup_ratio), 
        num_training_steps=total_steps
    )

    if optimizer_state and scheduler_state:
        optimizer.load_state_dict(optimizer_state)
        scheduler.load_state_dict(scheduler_state)  
              
    if val_dataloader:
        model, optimizer, train_dataloader, val_dataloader = accelerator.prepare(model, optimizer, train_dataloader, val_dataloader)
    else:
        model, optimizer, train_dataloader = accelerator.prepare(model, optimizer, train_dataloader)

    # hard coded project name
    if progress and 'wandb_run_id' in progress:
        # Resume existing run
        wandb.init(
            project="rag-optimization",
            name=f"run_{experiment_path}",
            resume="allow",
            id=progress['wandb_run_id']
        )
        logger.info(f"Resuming wandb run with ID: {progress['wandb_run_id']}")
        
        # Set training state from progress
        start_epoch = progress['epoch']
        optimizer_step = progress['optimizer_step']
        all_losses = progress['all_losses']
        epoch_losses = progress['epoch_losses']
        val_losses = progress.get('val_losses', [])
        best_val_loss = min(val_losses) if val_losses else float('inf')
    else:
        # Start a new run
        wandb.init(project="rag-optimization", name=f"run_{experiment_path}")
        
        # Initialize training state
        start_epoch = 0
        optimizer_step = 0
        all_losses = []
        best_val_loss = float('inf')   
        val_losses = []
        
    eval_steps = max((len(train_dataloader) // gradient_accumulation_steps) // eval_frequency, 1)
    early_stopping = EarlyStopping(patience=patience, min_delta=min_delta)
 
    if is_supervised:
        model.generator.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    else:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        
    torch.set_grad_enabled(True)

    for epoch in range(start_epoch, num_epochs):
        model.train()
        epoch_losses = []
        total_loss = 0
        val_count = 0
        for i, batch in enumerate(tqdm(train_dataloader, desc=f"Epoch {epoch+1}")):
            inputs = {k: (v if is_supervised and k == 'answer_texts' else v.to(accelerator.device)) for k, v in batch.items()}
            with accelerator.autocast():
                outputs = model(**inputs)
                loss = (outputs if is_supervised else outputs.loss) / gradient_accumulation_steps
                
            accelerator.backward(loss)
            total_loss += loss.item() * gradient_accumulation_steps
            epoch_losses.append(loss.item() * gradient_accumulation_steps)
            logger.info(f"Step {i+1}, Loss: {loss.item() * gradient_accumulation_steps:.4f}")

            if (i + 1) % gradient_accumulation_steps == 0:
                trainable_params = [p for p in model.parameters() if p.requires_grad]
                accelerator.clip_grad_norm_(trainable_params, max_grad_norm)
                grad_norm = log_grad_norm(model)
                
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                
                optimizer_step += 1
                # averaging actual loss per batch by grad accumulation, ie loss on which gradient was computed
                # adding .backward(outputs.loss / grad accumulation) for grad_acummulation such losses
                # these are the same
                avg_loss = total_loss / gradient_accumulation_steps
                all_losses.append(avg_loss)
                logger.info(f"Optimizer step: {i+1}, Loss: {avg_loss:.4f}, Grad norm: {grad_norm:.4f}")
                wandb.log({"train_loss": avg_loss, "learning_rate": scheduler.get_last_lr()[0], "grad_norm": grad_norm})
                total_loss = 0
                
                if optimizer_step % eval_steps == 0 and do_eval:
                    val_loss = evaluate(model, val_dataloader, epoch, val_count, accelerator, is_supervised)
                    model.train()  # back to train
                    val_losses.append(val_loss)
                    val_count += 1
                    logger.info(f"Global step {optimizer_step} validation loss: {val_loss:.4f}")
                    wandb.log({"val_loss": val_loss})
                    
                    # TODO: hardcoded min_epoch before saving best val model
                    if val_loss < best_val_loss and epoch >= 3:
                        best_val_loss = val_loss
                        ckpt_name = f"best_model_epoch_{epoch+1}_step_{optimizer_step}"
                        progress = {
                            'wandb_run_id': wandb.run.id,
                            'epoch': epoch+1,
                            'optimizer_step': optimizer_step, 
                            'all_losses': all_losses, 
                            'epoch_losses': epoch_losses, 
                            'val_losses': val_losses, 
                        }
                        checkpoint = {
                            'generator': {
                                'model': model.generator if is_supervised else model,
                                'tokenizer': tokenizer,
                            },
                            'training_state': {
                                'optimizer': optimizer,
                                'scheduler': scheduler,
                                'progress': progress
                            }
                        }

                        if is_supervised:
                            checkpoint['retriever'] = {
                                'question_encoder': model.question_encoder
                            }
                        # stage is "" when not combined
                        path_in_repo = os.path.join(experiment_path, stage, "val", ckpt_name)

                        uploader.upload_to_hub(
                            content=checkpoint,
                            path_in_repo=path_in_repo,
                            repo_id=repo_id
                        )                            
                 
                    if early_stopping(val_loss) and do_early_stopping:
                        logger.info(f"Early stopping triggered at epoch {epoch+1}, step {optimizer_step}")
                        return

        avg_train_loss = sum(epoch_losses) / len(epoch_losses)
        logger.info(f"Epoch {epoch + 1} average training loss: {avg_train_loss:.4f}")
        wandb.log({"epoch_train_loss": avg_train_loss, "epoch": epoch + 1})
        
        ckpt_name = f"checkpoint-epoch-{epoch+1}"
        progress = {
            'wandb_run_id': wandb.run.id,
            'epoch': epoch+1,
            'optimizer_step': optimizer_step, 
            'all_losses': all_losses, 
            'epoch_losses': epoch_losses, 
            'val_losses': val_losses
        }
        
        
        checkpoint = {
            'generator': {
                'model':  model.generator if is_supervised else model,
                'tokenizer': tokenizer,
            },
            'training_state': {
                'optimizer': optimizer,
                'scheduler': scheduler,
                'progress': progress
            }
        }

        if is_supervised:
            checkpoint['retriever'] = {
                'question_encoder': model.question_encoder
            }

        path_in_repo = os.path.join(experiment_path, stage, ckpt_name)
        uploader.upload_to_hub(
            content=checkpoint,
            path_in_repo=path_in_repo,
            repo_id=repo_id
        )

    if isinstance(model, RAGSequence):
        model.to_cpu()
        model.generator = None
        model.question_encoder = None
    else:
        model.to('cpu')
    torch.cuda.empty_cache()
    gc.collect()
    
    log_memory_usage("after training")

    name = "End-to-end supervised" if is_supervised else "Unsupervised"
    logger.info(f"{name} fine-tuning completed!")
    
    wandb.finish()

def load_specific_checkpoint(repo_id, specific_checkpoint, retriever_docs, accelerator):
    """
    Load a model from a specific checkpoint path.
    
    Args:
        repo_id: Repository ID
        specific_checkpoint: Path to the specific checkpoint to load
        retriever_docs: Documents for retriever initialization
        accelerator: Accelerator for device management
        
    Returns:
        Loaded model from the specified checkpoint
    """
    logger.info(f"Loading specific checkpoint: {specific_checkpoint}")
    
    # Determine experiment path by extracting from checkpoint path
    # Remove checkpoint-specific part to get the experiment directory
    experiment_path = specific_checkpoint
    if '/uft/' in specific_checkpoint:
        experiment_path = specific_checkpoint.split('/uft/')[0]
    elif '/sft/' in specific_checkpoint:
        experiment_path = specific_checkpoint.split('/sft/')[0] 
    elif '/checkpoint-epoch-' in specific_checkpoint:
        experiment_path = os.path.dirname(specific_checkpoint)
    
    # Load config from repository based on experiment path
    config_path = os.path.join(experiment_path, "config.yml")
    logger.info(f"config path on Hub: {config_path}")
    try:
        config_file = hf_hub_download(repo_id, config_path)
        with open(config_file, "r") as file:
            config = Box(yaml.safe_load(file))
            logger.info(f"Loaded config from: {config_path}")
    except Exception as e:
        logger.error(f"Error loading config from: {config_path}: {e}")
        raise
    

    # Log checkpoint details
    if '/val/' in specific_checkpoint:
        # Extract epoch and step info from validation checkpoint
        match = re.search(r'best_model_epoch_(\d+)_step_(\d+)', specific_checkpoint)
        if match:
            epoch_num, step_num = match.groups()
            logger.info(f"Loading validation checkpoint from epoch: {epoch_num}, optimizer step: {step_num}")
    else:
        # Extract epoch from regular checkpoint
        match = re.search(r'checkpoint-epoch-(\d+)', specific_checkpoint)
        if match:
            epoch_num = match.group(1)
            logger.info(f"Loading checkpoint from epoch: {epoch_num}")
    
    # Load the model
    model = load_trained_model(
        repo_id=repo_id,
        latest_checkpoint=specific_checkpoint,
        config=config,
        retriever_docs=retriever_docs,
        accelerator=accelerator,
    )
    
    return model
 
def load_or_finetune(retriever_docs, uft_docs, config, specific_checkpoint=None):
    # Initialize the accelerator with fp16, init first to use device in retriever/anywhere else
    # os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    accelerator = Accelerator(mixed_precision="fp16")
    

    if specific_checkpoint:
        return load_specific_checkpoint(
            repo_id=config.repo_id,
            specific_checkpoint=specific_checkpoint,
            retriever_docs=retriever_docs,
            accelerator=accelerator
        )
    
    annotated_train = load_json(config.dataset.train)
    human_annotated_train = load_json(config.dataset.human_train)
    human_annotated_test = load_json(config.dataset.test)
    
    # NOTE: only training on H_train intersection with U. May change to full H_train
    # target_indices = set(doc['index'] for doc in uft_docs).intersection(doc['index'] for doc in human_annotated_train)
    target_indices = set(doc['index'] for doc in human_annotated_train)
    sft_dataset = [doc for doc in human_annotated_train if doc['index'] in target_indices]
     
    repo_id = config.repo_id
    experiment_path = os.path.join("experiments", *config.model.experiment.values())
    exp_type = config.model.experiment.type
    
    # TODO: 
    # 1. add loading from config in Hub when evaluating multiple models, ie with eval_config.yml
    # 2. load an intermediate epoch functionality 
    subfolder = os.path.join(experiment_path, "uft") if (config.model.experiment.type == 'combined') else experiment_path
    latest_checkpoint, latest_epoch = get_latest_checkpoint(repo_id, subfolder)
    
    if config.model.experiment.type == "combined":
        sft_checkpoint = get_latest_checkpoint(repo_id, os.path.join(experiment_path, "sft"))
        if all(sft_checkpoint):  # Ensures both checkpoint and epoch exist
            latest_checkpoint, latest_epoch = sft_checkpoint
        
    model_exists = latest_epoch is not None and latest_checkpoint is not None
    if model_exists:
        logger.info(f"\n latest_checkpoint: {latest_checkpoint}\n latest_epoch: {latest_epoch}")

    # TODO: YAML filename harcoded here, make it more flexible in future
    config_path = os.path.join(experiment_path, "config.yml")
    try:
        hf_hub_download(config.repo_id, config_path)
        logger.info("config.yml already present on the Hub, skipping re-upload")
    except:
        uploader = HubUploader()
        uploader.upload_to_hub(
            config,
            path_in_repo=config_path,
            repo_id=repo_id
        )
        logger.info("config.yml uploaded to the Hub")
        
    # Model exists only when at least 1 epoch is completed, else fresh computation starts
    # A. Resume logic
    if model_exists:
        if exp_type == 'uft-only':
            config_epochs = config.model.training.unsupervised.optimization.num_epochs
        else: # has to be either sft-only or combined
            config_epochs = config.model.training.supervised.optimization.num_epochs
           
        # If training complete, load and return
        if config_epochs and latest_epoch == config_epochs:
            return load_trained_model(repo_id, latest_checkpoint, config, retriever_docs, accelerator)
        
        # Resume training from checkpoint
        else: 
            model, optimizer_state, scheduler_state, progress, latest_checkpoint, is_supervised = resume_from_checkpoint(
                repo_id=repo_id,
                experiment_path=experiment_path,
                config=config,
                retriever_docs=retriever_docs,
                accelerator=accelerator
            )
            if exp_type == 'combined':
                if is_supervised:
                    # Finish remaining epochs of SFT stage OR start SFT if transition
                    train_dataloader, val_dataloader, _ = sft_dataloader(
                        sft_dataset, 
                        human_annotated_test,
                        model.retriever.question_encoder_tokenizer,
                        model.generator_tokenizer,
                        **config.model.training.supervised.dataloader
                    )
                    
                    if optimizer_state and scheduler_state and progress:
                        logger.info("Resuming SFT stage of combined...")
                    else:  # it is a transition
                        logger.info("Initiating SFT stage of combined...")
                    stage = "sft"

                    train(
                        model, 
                        model.generator_tokenizer, 
                        accelerator,
                        train_dataloader, 
                        val_dataloader, 
                        repo_id, 
                        experiment_path,
                        **config.model.training.supervised.optimization,
                        optimizer_state=optimizer_state,
                        scheduler_state=scheduler_state,
                        progress=progress,
                        is_supervised=True,
                        stage=stage, 
                    )
                    
                else:
                    # Finish remaining epochs of UFT stage and then finish SFT stage
                    train_dataloader, val_dataloader = uft_dataloader(
                        uft_docs, 
                        model.generator_tokenizer, 
                        **config.model.training.unsupervised.dataloader     
                    )
                    
                    logger.info(f"Resuming UFT stage of combined (initiating epoch {latest_epoch+1})...")
                    stage = "uft"
                    
                    train(
                        model.generator, 
                        model.generator_tokenizer, 
                        accelerator,
                        train_dataloader,
                        val_dataloader,
                        repo_id, 
                        experiment_path,
                        **config.model.training.unsupervised.optimization,
                        optimizer_state=optimizer_state,
                        scheduler_state=scheduler_state,
                        progress=progress,
                        is_supervised=False,  
                        stage=stage
                    )
                    
                    # SFT stage
                    latest_checkpoint, latest_epoch = get_latest_checkpoint(repo_id, os.path.join(experiment_path, "uft"))
                    model = load_trained_model(repo_id, latest_checkpoint, config, retriever_docs, accelerator, 
                                                is_transition=True, is_trainable=True, is_supervised=True)
                    optimizer_state, scheduler_state, progress = load_training_state(
                        repo_id=repo_id,
                        checkpoint_path=latest_checkpoint
                    )
                    train_dataloader, val_dataloader, _ = sft_dataloader(
                        sft_dataset, 
                        human_annotated_test,
                        model.retriever.question_encoder_tokenizer,
                        model.generator_tokenizer,
                        **config.model.training.supervised.dataloader
                    )
                    
                    logger.info(
                        f"Initiating SFT stage of combined using..."
                        f"\n  latest_checkpoint: {latest_checkpoint}/uft\n  latest_epoch: {latest_epoch}"
                    )
                    stage = "sft"
                    
                    train(
                        model, 
                        model.generator_tokenizer, 
                        accelerator,
                        train_dataloader, 
                        val_dataloader, 
                        repo_id, 
                        experiment_path,
                        **config.model.training.supervised.optimization,
                        optimizer_state=optimizer_state,
                        scheduler_state=scheduler_state,
                        progress=progress,
                        is_supervised=True,
                        stage=stage, 
                    )

            else:
                if is_supervised:
                   # Finish remaining epochs of SFT for sft-only
                    train_dataloader, val_dataloader, _ = sft_dataloader(
                        sft_dataset, 
                        human_annotated_test,
                        model.retriever.question_encoder_tokenizer,
                        model.generator_tokenizer,
                        **config.model.training.supervised.dataloader
                    )

                    logger.info(f"Resuming SFT for sft-only (initiating epoch {latest_epoch+1})...")
                    stage = ""
                    
                    train(
                        model, 
                        model.generator_tokenizer, 
                        accelerator,
                        train_dataloader, 
                        val_dataloader, 
                        repo_id, 
                        experiment_path,
                        **config.model.training.supervised.optimization,
                        optimizer_state=optimizer_state,
                        scheduler_state=scheduler_state,
                        progress=progress,
                        is_supervised=True,
                        stage=stage, 
                    )
                else:
                    # Finish remaining epochs of UFT for uft-only
                    train_dataloader, val_dataloader = uft_dataloader(
                        uft_docs, 
                        model.generator_tokenizer, 
                        **config.model.training.unsupervised.dataloader     
                    )
                    
                    logger.info(f"Resuming UFT for uft-only (initiating epoch {latest_epoch+1})...")
                    stage = ""
                    
                    train(
                        model.generator, 
                        model.generator_tokenizer, 
                        accelerator,
                        train_dataloader, 
                        val_dataloader,
                        repo_id, 
                        experiment_path,
                        **config.model.training.unsupervised.optimization,
                        optimizer_state=optimizer_state,
                        scheduler_state=scheduler_state,
                        progress=progress,
                        is_supervised=False,  
                        stage=stage
                    )
                    
    # B. Fresh training logic              
    else:
        is_supervised = (exp_type == 'sft-only')
        model = load_base_model(config, retriever_docs, accelerator, is_supervised)
        
        if exp_type in ['uft-only', 'combined']:           
            train_dataloader, val_dataloader = uft_dataloader(
                uft_docs, 
                model.generator_tokenizer, 
                **config.model.training.unsupervised.dataloader     
            )  
                      
            logger_message = (
                "Initiating UFT stage of combined (first time)..." 
                if exp_type == "combined" 
                else "Initiating UFT for uft-only (first time)..."
            )
            logger.info(logger_message)
            stage = "uft" if exp_type == 'combined' else ""
            
            train(
                model.generator, 
                model.generator_tokenizer, 
                accelerator,
                train_dataloader, 
                val_dataloader,
                repo_id, 
                experiment_path,
                **config.model.training.unsupervised.optimization,
                is_supervised=False,  
                stage=stage
            )
        if exp_type in ['sft-only', 'combined']: 
            
            if exp_type == 'combined':
                # TODO: PEFT config will be same as unsupervised, can't change one trained adapter for another 
                # downstream training. check logic here
                latest_checkpoint, latest_epoch = get_latest_checkpoint(repo_id, os.path.join(experiment_path, "uft"))
                model = load_trained_model(repo_id, latest_checkpoint, config, retriever_docs, accelerator, 
                                           is_transition=True, is_trainable=True, is_supervised=True)

                logger.info(
                    f"Initiating SFT stage of combined using (first time)..."
                    f"\n  latest_checkpoint: {latest_checkpoint}/uft\n  latest_epoch: {latest_epoch}"
                )
                
            else:                      
                model.prepare_for_sft()
                logger.info(f"Initiating SFT for sft-only (first time)...")
            
            train_dataloader, val_dataloader, _ = sft_dataloader(
                sft_dataset,
                human_annotated_test,
                model.retriever.question_encoder_tokenizer,
                model.generator_tokenizer,
                **config.model.training.supervised.dataloader
            )
            stage = "sft" if exp_type == 'combined' else ""
            
            train(
                model, 
                model.generator_tokenizer, 
                accelerator,
                train_dataloader, 
                val_dataloader, 
                repo_id, 
                experiment_path,
                **config.model.training.supervised.optimization,
                is_supervised=True,
                stage=stage, 
            )
        
    # C. Return now trained model 
    subfolder = (
        os.path.join(experiment_path, "sft") 
        if (exp_type == 'combined') 
        else experiment_path
    )
    latest_checkpoint, latest_epoch = get_latest_checkpoint(repo_id, subfolder)
    
    return load_trained_model(repo_id, latest_checkpoint, config, retriever_docs, accelerator)
     
def load_base_model(config, retriever_docs, accelerator, is_supervised):
    
    question_encoder, custom_retriever = None, None
    
    if is_supervised:
        question_encoder = DPRQuestionEncoder.from_pretrained(
            config.model.retriever.question_encoder,
        ).to(accelerator.device)  # does not support device_map
    
        custom_retriever = CustomRetriever(config, retriever_docs, accelerator)
        logger.info("Loaded question encoder and retriever (for sft-only / SFT stage or transition of combined)")
    else:
        logger.info("No need to load question encoder and retriever (for uft-only / UFT stage of combined)")
    
    tokenizer = AutoTokenizer.from_pretrained(config.model.generator.base_model)
    
    generator = AutoModelForCausalLM.from_pretrained(
        config.model.generator.base_model,
        attn_implementation=config.model.generator.attn_implementation,
        torch_dtype=torch.float16,  # no significant impact on accuracy, see 03/14 notes point 2
        device_map="auto"  # to have accelerate compute most optimal map
    )
    log_parameter_info(generator, f"Pre-LoRA {config.model.generator.base_model}")
    
    model = RAGSequence(
        question_encoder,
        custom_retriever.retriever if custom_retriever else None,
        tokenizer, 
        generator, 
        accelerator
    )
    
    # Add <pad> token, to prevent pad_token=eos_token else otherwise
    # all pad tokens including an actual eos_token will be masked out
    # making the model never learn when to end a generation
    model.add_pad_token("<pad>")
    
    lora_config = config.model.generator.lora 

    if is_supervised:
        peft_config_qencoder = LoraConfig(
            task_type=TaskType.FEATURE_EXTRACTION,
            inference_mode=False,
            r=lora_config.r,
            lora_alpha=lora_config.alpha,
            lora_dropout=lora_config.dropout,
            target_modules=["query", "key", "value"],
            bias="none"
        )
        model.question_encoder = get_peft_model(model.question_encoder, peft_config_qencoder)
        logger.info("Created PEFT model for base question encoder (initiating SFT for sft-only)")

    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=lora_config.r,
        lora_alpha=lora_config.alpha,
        lora_dropout=lora_config.dropout
    )
    model.generator = get_peft_model(model.generator, peft_config)
    
    if is_supervised:
        logger.info("Created PEFT model for base generator (initiating SFT for sft-only)") 
    else:
        logger.info("Created PEFT model for base generator (initiating UFT for uft-only/combined)") 
       
    
    if is_supervised:
        log_parameter_info(model.question_encoder, config.model.retriever.question_encoder)
        
    log_parameter_info(model.generator, config.model.generator.base_model)
    if is_supervised:
        logger.info(
            "\nLoaded base RAGSequence model from given config:"
            f"\n  1. Question encoder {config.model.retriever.question_encoder}"
            f"\n  2. Custom retriever {config.model.retriever.base_model}"
            f"\n  3. Generator tokenizer and model {config.model.generator.base_model}\n"
        )
    else:
        logger.info(
            "\nLoaded base RAGSequence model from given config:"
            f"\n  1. Generator tokenizer and model {config.model.generator.base_model}\n"
        )
        
    log_memory_usage("after exiting load_base_model()")
    return model



def load_trained_model(
    repo_id, 
    latest_checkpoint, 
    config, 
    retriever_docs,
    accelerator, 
    is_transition=False, 
    is_trainable=False,
    is_supervised=False
):
    """
    Load a trained model from Hugging Face Hub.
    
    Args:
        repo_id: HF repository ID
        latest_checkpoint: Path to checkpoint in repo
        config: Configuration object
        retriever_docs: Documents for retriever
        accelerator: Accelerator for device management
        is_transition: Whether loading for transition from UFT to SFT
        is_trainable: Whether model is being loaded for continued training
        is_supervised: Whether the question encoder has to be trained
        
        Note that is_supervised (s) implies is_trainable (t), ie s => t (is True)
    """

    log_memory_usage("after calling load_trained_model()")
    
    # Load question encoder
    question_encoder, custom_retriever = None, None
    
    # Load question encoder when is_supervised => resumming sft-only/combined sft
    # or for inference when not is_trainable, this condition will 
    # not be triggered when resuming UFT of uft-only/combined run
    if is_supervised or not is_trainable:
        question_encoder = DPRQuestionEncoder.from_pretrained(
            config.model.retriever.question_encoder
        ).to(accelerator.device)

        # try to find a checkpoint (for either training or inference)
        try:
            question_encoder = PeftModel.from_pretrained(
                question_encoder,
                repo_id,
                subfolder=os.path.join(latest_checkpoint, "retriever", "question_encoder"),
                is_trainable=is_trainable # ensures param grads are as expected
            ).to(accelerator.device)
            
            if is_trainable:
                logger.info(f"Question encoder loaded from checkpoint (to resume SFT for either sft-only/combined)")
            else:
                logger.info(f"Question encoder loaded from checkpoint (to do inference for either sft-only/combined)")
        
        # If no SFT checkpoint found
        except:
            # And if is_trainable then must be a transition
            if is_trainable:
                # Create PEFT config for question encoder
                lora_config = config.model.generator.lora 
                peft_config_qencoder = LoraConfig(
                    task_type=TaskType.FEATURE_EXTRACTION,
                    r=lora_config.r,
                    lora_alpha=lora_config.alpha,
                    lora_dropout=lora_config.dropout,
                    target_modules=["query", "key", "value"],
                    inference_mode=False,
                    bias="none"
                )
                # Apply PEFT to question encoder
                question_encoder = get_peft_model(question_encoder, peft_config_qencoder)
                logger.info("Created PEFT model for base question encoder (UFT->SFT transition of combined)")
            # else it must be to inference a uft-only model
            else:
                logger.info("Loaded question encoder for uft-only inference")
            
        # Only for uft-only / UFT stage of combined resume, retriever will not initalize
        # else, for inference it will since is_trainable will be False
        custom_retriever = CustomRetriever(config, retriever_docs, accelerator)

    
    # Load tokenizer from the Hub (will have correct vocab size)
    tokenizer = AutoTokenizer.from_pretrained(
        repo_id, 
        subfolder=os.path.join(latest_checkpoint, "generator", "tokenizer")
    )
    
    quantization_config = None
    if not is_trainable:
        quantization_config = BitsAndBytesConfig(
            load_in_8bit=True
            # Add other parameters as needed
        )
    # Load base generator model
    generator = AutoModelForCausalLM.from_pretrained(
        config.model.generator.base_model,
        attn_implementation=config.model.generator.attn_implementation,
        quantization_config=quantization_config,
        torch_dtype=torch.float16,
        device_map="auto"
    )
     
    # Initialize RAG model with base components
    model = RAGSequence(
        question_encoder,
        custom_retriever.retriever if custom_retriever else None,
        tokenizer, 
        generator, 
        accelerator
    ) 
       
    # Ensure model matches latest UFT checkpoint model
    if is_transition:
        model.add_pad_token("<pad>")
    # If not a transition, rely on tokenizer's latest UFT/SFT checkpoint
    else:
        base_vocab_size = model.generator.config.vocab_size
        prev_vocab_size = len(model.generator_tokenizer)  # vocab size from previous run we are resuming for
        if base_vocab_size != prev_vocab_size:
            logger.info("Resizing model's embedding matrix vocab_size...")
            model.generator.resize_token_embeddings(prev_vocab_size)
            logger.info(f"Resized model's embedding matrix vocab_size from {base_vocab_size} to {prev_vocab_size}")
        
    # Load PEFT adapter safely
    model.generator = PeftModel.from_pretrained(
        model.generator,
        repo_id, 
        subfolder=os.path.join(latest_checkpoint, "generator", "model"),
        device_map="auto",
        is_trainable=is_trainable  # ensures param grads are as expected for training
    )
    
    # Modify after PEFT adapter is loaded, since internally PEFT validates shape 
    # to actual base model the adapters were created on, which did not have instruction tokens
    # Set up for SFT if transitioning from UFT
    if is_transition:
        model.prepare_for_sft()
        logger.info(f"Tokenizer prepared for SFT: \n{tokenizer}")
    
    # Log parameter info
    if is_supervised:
        log_parameter_info(model.question_encoder, config.model.retriever.question_encoder)
    log_parameter_info(model.generator, config.model.generator.base_model)
    

    
    def log_final_message(description, latest_checkpoint):
        logger.info(f"Loaded saved model from the Hub ({description}):\n {latest_checkpoint}")
    
    if is_supervised:
        if is_transition:
            log_final_message("transition detected, initiating SFT for combined", latest_checkpoint)
        else:
            log_final_message("resuming SFT for sft-only/combined", latest_checkpoint)
    else:
        if is_trainable:
            log_final_message("resuming UFT for uft-only/combined", latest_checkpoint)
        else:
            # Add to config to prevent generate() warning, see 03/21 notes point 1.
            model.generator.config.pad_token_id = model.generator_tokenizer.pad_token_id
            log_final_message("inference for either uft-only/sft-only/combined", latest_checkpoint)
            
    log_memory_usage("after exiting load_trained_model()")
    return model



# Pseudocode for load_or_finetune()
def load_or_ft():
    # retriever docs + uft docs + config given
    # check if model (uft/sft based on config) exists
        # if epochs == config epochs then load it and return
        # if epochs != config then resume (uft/sft/uft+sft) + save + load + return
    # if no model exists then based on config:
        # do uft / do sft / do uft then sft
        # save + load + return
    pass
