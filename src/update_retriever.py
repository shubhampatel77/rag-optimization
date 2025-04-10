from transformers import (
    DPRContextEncoder, DPRContextEncoderTokenizer,
    DPRQuestionEncoderTokenizer, AutoTokenizer, RagRetriever, RagConfig
)
from datasets import Features, Sequence, Value, Dataset, concatenate_datasets
import torch
import os
import pickle
import faiss
import logging
import numpy as np
from tqdm.auto import tqdm
from huggingface_hub import list_repo_files, hf_hub_download, upload_file, upload_folder, snapshot_download
import tempfile

from .utils import parse_date, split_text, setup_logger, HubUploader

logger = setup_logger(enable_logging=True)
torch.set_grad_enabled(False)

# TODO: once retriever_docs increases, remove explicit device handling using accelerator, instead use
# device_map="auto", refactor embed(), DPRContextEncoder, RagRetriever init too.
class CustomRetriever:
    def __init__(self, config, retriever_docs, accelerator):
        self.uploader = HubUploader()
        self.repo_id = config.repo_id
        self.tokenizer = DPRContextEncoderTokenizer.from_pretrained(config.model.retriever.ctx_encoder)
        self.context_encoder = DPRContextEncoder.from_pretrained(config.model.retriever.ctx_encoder).to(accelerator.device)
        self.retriever = self.initialize_retriever(config, retriever_docs, accelerator)
        self.dataset = None
        self.index = None
        
        self.max_doc_len = 2048 # TODO: come back and re-analyze

    # TODO: add feature to copy over dataset + index from other runs when no changes to it, to save compute time and mem
    def initialize_retriever(self, config, retriever_docs, accelerator):
        # The Hub stores all keys of the config in lexicographic order, messing up paths
        exp = config.model.experiment
        # Explicit ordering required
        experiment_path = os.path.join(
            "experiments", exp.type, exp.scoring, exp.variant, exp.run_id
        )
        # TODO: dataset/FAISS filename hardcoded here, make it more flexible in future
        # full name needed since, that will be the name of the file uploaded to Hub
        # TODO: may want to change memory to datastore
        dataset_path = os.path.join(experiment_path, "memory", "dataset")
        index_path = os.path.join(experiment_path, "memory", "index.faiss")
        
        if retriever_docs is None:
            self.load_or_create_base_index(config.dataset.base.data, config.dataset.base.index, config)
            return RagRetriever.from_pretrained(
                config.model.retriever.base_model,
                index_name="custom",
                passages_path=config.dataset.base.data,
                index_path=config.dataset.base.index,
            )

        else:
            if not self.check_hub_files(dataset_path, index_path):
                self.load_or_create_base_index(config.dataset.base.data, config.dataset.base.index, config)
                logger.info("Adding retriever docs to index...")
                self.process_retriever_docs(retriever_docs, accelerator)
                self.uploader.upload_to_hub(
                    self.dataset, 
                    path_in_repo=dataset_path,
                    repo_id=self.repo_id,
                )
                # Upload a FAISS index
                self.uploader.upload_to_hub(
                    self.index,
                    index_path,
                    repo_id=self.repo_id
                )
            else:
                logger.info("Loading existing dataset and index from the Hub...")
            
            return self.load_from_hub_and_instantiate(dataset_path, index_path, config)

    def check_hub_files(self, dataset_path, index_path):
        try:
            files = list_repo_files(self.repo_id)
            # Check if the index file exists
            index_exists = index_path in files
            # Check if the dataset folder exists by looking for its contents
            dataset_exists = any(file.startswith(dataset_path) for file in files)
            return index_exists and dataset_exists
        
        except Exception as e:
            logger.error(f"Error checking Hub files: {e}")
            return False
        
    def load_or_create_base_index(self, base_dataset_path, base_index_path, config):
        """Initialize or load base index and dataset."""
        # If files exist, try to load them
        if os.path.exists(base_dataset_path) and os.path.exists(base_index_path):
            try:
                self.dataset = Dataset.load_from_disk(base_dataset_path)
                self.index = faiss.read_index(base_index_path)
                
                # Verify logical integrity
                if self.index.ntotal == len(self.dataset):
                    logger.info(
                        f"Successfully loaded existing index ({self.index.ntotal} vectors) and "
                        f"dataset ({len(self.dataset)} entries)"
                    )
                    return
                else:
                    logger.warning(f"Size mismatch in saved base, index size: {self.index.ntotal} vs dataset size: {len(self.dataset)}")
                    
            except Exception as e:
                logger.error(f"Error loading existing files: {e}")
        
        # Create new if loading failed or files don't exist
        logger.info("Creating new base index and dataset...")
        self.dataset = RagRetriever.from_pretrained(
            config.model.retriever.base_model, 
            index_name="exact", 
            use_dummy_dataset=True
        ).index.dataset
    
        embeddings = np.array(self.dataset['embeddings'])
        
        self.index = faiss.IndexHNSWFlat(
            self.dataset['embeddings'].shape[1],  # 768 dimensions
            128,  # M parameter (number of connections per layer)
            faiss.METRIC_INNER_PRODUCT  # Similarity metric
        )
        self.index.add(embeddings)
        
        # Save new files (only for base)
        self.dataset.save_to_disk(base_dataset_path)
        faiss.write_index(self.index, base_index_path)
        
        logger.info(f"Created new index ({self.index.ntotal} vectors) and dataset ({len(self.dataset)} entries)")
    
    def embed(self, documents, accelerator):
        input_ids = self.tokenizer(
            documents["title"], 
            documents["text"], 
            truncation=True, 
            padding="max_length", 
            max_length=self.max_doc_len, 
            return_tensors="pt"
        )["input_ids"]
        with torch.no_grad(), accelerator.autocast():
            embeddings = self.context_encoder(input_ids.to(self.context_encoder.device), return_dict=True).pooler_output
        return {"embeddings": embeddings.detach().cpu().numpy()}

    def process_retriever_docs(self, retriever_docs, accelerator):
        new_docs = []
        for doc in retriever_docs:
            text = ' '.join(doc['paras'])
            for passage in split_text(text, n=100): # Split into ~100 word chunks
                new_docs.append({
                    "title": doc['index'].split('#')[0].split('/')[-1],
                    "text": passage,
                })
        
        new_dataset = Dataset.from_dict({"title": [d["title"] for d in new_docs], "text": [d["text"] for d in new_docs]})
        new_dataset = new_dataset.map(
            lambda batch: self.embed(batch, accelerator),
            batched=True,
            batch_size=16,
            desc="Computing embeddings"
        )
        
        # Concatenate datasets
        self.dataset = concatenate_datasets([self.dataset, new_dataset])
        embeddings = np.array(self.dataset['embeddings'])
        
        # Check if index already has base dataset before adding the concatenated dataset, reset
        if self.index:
            self.index.reset()
        self.index.add(embeddings)
        
        # Done before uploading, ensuring uploaded items are consistent
        assert self.index.ntotal == len(self.dataset), (
            f"Size mismatch in saved base dataset and index, "
            f"index size: {self.index.ntotal} vs dataset size: {len(self.dataset)}"
        )
        logger.info(f"Updated dataset size: {len(self.dataset)}, Updated index size: {self.index.ntotal}")

    # TODO: Combine with reusing dataset + index to prevent excess caching or mem usage in experiments/
    def load_from_hub_and_instantiate(self, dataset_path, index_path, config):
        """
        Load dataset and index from Hub.
        
        Args:
            dataset_path: Local path where dataset should be saved 
            index_path: Local path where index should be saved
            config: Configuration object
        """
        # Ensure directories exist
        os.makedirs(os.path.dirname(dataset_path), exist_ok=True)

        # List all files in repo
        hub_files = list_repo_files(self.repo_id)
        
        # Get dataset files using pattern matching
        dataset_name = os.path.basename(dataset_path)
        dataset_files = [f for f in hub_files if f"{dataset_path}/" in f]

        # Download each dataset file to dir level of file executing this
        # downloaded file retains dir structure of Hub
        for file in dataset_files:
            hf_hub_download(
                repo_id=self.repo_id,
                filename=file,
                local_dir=''
            )
        
        # Load dataset
        self.dataset = Dataset.load_from_disk(dataset_path)
        logger.info(f"Total document chunks in datastore: {len(self.dataset)}")
        
        # Download and load index 
        index_file = hf_hub_download(
            repo_id=self.repo_id,
            filename=index_path,
            local_dir=''
        )
        self.index = faiss.read_index(index_file)
        
        return RagRetriever.from_pretrained(
            config.model.retriever.base_model,
            index_name="custom",
            passages_path=dataset_path,
            index_path=index_path
        )

        