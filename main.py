# env imports
import json
from tqdm.auto import tqdm
import os
import wandb
os.environ["WANDB_API_KEY"] = "YOUR_WANDB_KEY"
# Log in to WandB
wandb.login()
# Log in to huggingface hub
from huggingface_hub import login
login(
  token="YOUR_HUGGINGFACE_TOKEN"
)
import pickle
import subprocess

# ML imports
import torch
import faiss
import numpy as np
import pandas as pd

import transformers
from transformers import (
    logging, pipeline,
    AutoTokenizer, MistralForCausalLM, AutoModelForSeq2SeqLM, AutoModelForCausalLM,
    DPRQuestionEncoder, DPRQuestionEncoderTokenizerFast, DPRContextEncoder, DPRReader,
    RagRetriever, RagSequenceForGeneration, RagConfig, 
    DataCollatorForSeq2Seq, DataCollatorForLanguageModeling
)
from accelerate import Accelerator
logging.set_verbosity(logging.CRITICAL)
from transformers.models.rag.retrieval_rag import CustomHFIndex

# file imports
from src.update_retriever import CustomRetriever
from src.update_generator import load_or_finetune, load_base_model
from src.process_documents import load_or_score_docs
from src.dataloader import sft_dataloader
from src.evals import SetOperation, get_filtered_questions, evaluate_model
from src.utils import load_json, load_experiment_config
from src.visualization import MetricsVisualizer

# HF import
from trl import setup_chat_format

from datasets import load_from_disk, Features, Value, Sequence

import re
from collections import Counter

# visulalization related imports
import matplotlib.pyplot as plt
from matplotlib_venn import venn2

import seaborn as sns
import random
from statistics import mean
from fuzzywuzzy import fuzz
import sklearn
from sklearn.metrics import confusion_matrix, roc_auc_score

# misc
from datetime import datetime
import pytz
from dateutil.parser import parse
import warnings
warnings.filterwarnings("ignore")

import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import get_linear_schedule_with_warmup
import bitsandbytes as bnb
from typing import List, Tuple, Dict

from src.utils import setup_logger, log_structured_data
logger = setup_logger(enable_logging=True)

from box import Box
import yaml
# os.environ["CUDA_VISIBLE_DEVICES"] = "3"

def main():

    # Load YAML
    with open("configs/config.yml") as file:
        config = Box(yaml.safe_load(file))

    retriever_docs, uft_docs, scores, weights = load_or_score_docs(config)
    model = load_or_finetune(retriever_docs, uft_docs, config)
    
    human_annotated_train = load_json(config.dataset.human_train)
    human_annotated_test = load_json(config.dataset.test)

    index_sets = {
        'retriever': {doc['index'] for doc in retriever_docs},
        'uft': {doc['index'] for doc in uft_docs},
        'human_annotated_train': {doc['index'] for doc in human_annotated_train},
        # 'human_annotated_test': {doc['index'] for doc in human_annotated_test}
    }

    questions = get_filtered_questions(
        human_annotated_docs=human_annotated_train,#human_annotated_test,
        index_sets=index_sets,
        include_sets=['human_annotated_train'],
        # exclude_sets=['uft'],
        set_operation=SetOperation.UNION
    )
    logger.info(f"Number of questions: {len(questions)}, number of docs: {len(set(x['index'] for x in questions))}")
    save_dir = os.path.join("results", "all-human-train", *config.model.experiment.values())
    os.makedirs(save_dir, exist_ok=True)
    evaluate_model(model, questions, f"{save_dir}/result.pkl")
    
    # Populate results with appropriate keys to compare any two models given their .pkl
    results = {}
    with open(f"{save_dir}/result.pkl", 'rb') as f:
        results["SFT"] = pickle.load(f)

    metrics_vis = MetricsVisualizer(
        results=results,
        output_dir=save_dir,
        )
    metrics_vis.plot_all()
    
if __name__ == '__main__':
    main()