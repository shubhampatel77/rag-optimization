import json
import os
import random
import pickle
from datetime import datetime
import pytz
import numpy as np
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from datasets import Dataset, load_from_disk
from huggingface_hub import hf_hub_download, upload_file
import tempfile
# from visualization import plot_scores
from .utils import parse_date, load_json, setup_logger
logger = setup_logger(enable_logging=True)


def calculate_retriever_score(doc, current_time, existing_docs_sample, feature_dim, weights):
    recency_weight, relevance_weight, uniqueness_weight = weights
    # Recency score
    times = []
    for question in doc['questions']:
        start_date = parse_date(question[0][0])
        end_date = parse_date(question[0][1])
        if start_date and end_date:
            mean_timestamp = np.mean([start_date.timestamp(), end_date.timestamp()])
            times.append(mean_timestamp)
    
    if not times:
        return 0, [0, 0, 0]  # Return 0 if no valid dates found
    
    avg_timestamp = np.mean(times)
    avg_datetime = datetime.fromtimestamp(avg_timestamp, tz=pytz.UTC)
    time_diff_years = (current_time - avg_datetime).days / 365.25
    recency_score = 10 / (1 + time_diff_years)
    
    # Relevance score
    relevance_score = min(1, len(doc['questions']) / 10)
    
    # Uniqueness score
    doc_text = ' '.join(doc['paras'])
    uniqueness_score = calculate_lightweight_uniqueness(doc_text, feature_dim, existing_docs_sample)
    
    # Combine scores
    #.4,.3,.3 # 0,.5,.5
    score = recency_weight * recency_score + relevance_weight * relevance_score + uniqueness_weight * uniqueness_score
    scores = [recency_score, relevance_score, uniqueness_score]
    return score, scores

def calculate_lightweight_uniqueness(doc_text, feature_dim, existing_docs_sample):
    sample_size = len(existing_docs_sample)
    existing_texts = existing_docs_sample['text']
    
    vectorizer = CountVectorizer(max_features=feature_dim).fit(existing_texts + [doc_text])
    doc_vector = vectorizer.transform([doc_text])
    existing_vectors = vectorizer.transform(existing_texts)
    
    similarities = cosine_similarity(doc_vector, existing_vectors)
    return 1 - np.max(similarities)

def calculate_finetuning_score(doc, current_time, weights):
    temporal_weight, diversity_weight = weights
    # Temporal complexity score
    time_ranges = [q[0][0] for q in doc['questions'] if q[0]] + [q[0][1] for q in doc['questions'] if q[0]]
    temporal_complexity = len(set(time_ranges)) / len(doc['questions']) if doc['questions'] else 0
    
    # Answer diversity score
    unique_answers = set(a['answer'] for q in doc['questions'] for a in q[1])
    answer_diversity = len(unique_answers) / len(doc['questions']) if doc['questions'] else 0
    
    # Combine scores
    # .5, .5
    score = temporal_weight * temporal_complexity + diversity_weight * answer_diversity
    scores = [temporal_complexity, answer_diversity]
    return score, scores 

def process_documents(documents, existing_docs, retriever_weights, fine_tune_weights, model_config):
    current_time = datetime.now(pytz.UTC)
    scores = {
        doc["index"]: {
            "retriever": {
                "total": 0,
                "recency": 0,
                "relevance": 0,
                "uniqueness": 0
            },
            "uft": {
                "total": 0,
                "temporal_complexity": 0,
                "answer_diversity": 0
            }
        }
        for doc in documents
    }
    seed = model_config.retriever.sampling.seed
    existing_docs_sample_size = model_config.retriever.sampling.docs_sample_size
    feature_dim = model_config.retriever.sampling.feature_dim

    existing_docs_sample = existing_docs.shuffle(seed=seed).select(range(existing_docs_sample_size))
    
    for doc in tqdm(documents, desc="Calculating scores"):
        rs, rs_list = calculate_retriever_score(doc, current_time, existing_docs_sample, feature_dim, retriever_weights)
        fs, fs_list = calculate_finetuning_score(doc, current_time, fine_tune_weights)
        
        scores[doc["index"]]["retriever"]["total"] = rs
        scores[doc["index"]]["retriever"]["recency"] = rs_list[0]
        scores[doc["index"]]["retriever"]["relevance"] = rs_list[1]
        scores[doc["index"]]["retriever"]["uniqueness"] = rs_list[2]
        
        scores[doc["index"]]["uft"]["total"] = fs
        scores[doc["index"]]["uft"]["temporal_complexity"] = fs_list[0]
        scores[doc["index"]]["uft"]["answer_diversity"] = fs_list[1]

    # Calculate thresholds based on the score distributions
    rs_threshold = np.percentile([scores[doc["index"]]["retriever"]["total"] for doc in documents], 70)  # Top 30% go to memory
    fs_threshold = np.percentile([scores[doc["index"]]["uft"]["total"] for doc in documents], 70)  # Top 30% go to fine-tuning

    retriever_docs = []
    uft_docs = []

    for doc in documents:
        rs = scores[doc["index"]]["retriever"]["total"]
        fs = scores[doc["index"]]["uft"]["total"]
        if rs > rs_threshold and fs > fs_threshold:
            retriever_docs.append(doc)
            uft_docs.append(doc)
        elif rs > rs_threshold:
            retriever_docs.append(doc)
        elif fs > fs_threshold:
            uft_docs.append(doc)
        

    return retriever_docs, uft_docs, scores

def random_split(documents, seed, retriever_size=1068, uft_size=1024):
    random.seed(seed)
    all_indices = list(range(len(documents)))
    
    retriever_indices = set(random.sample(all_indices, retriever_size))
    uft_indices = set(random.sample(all_indices, uft_size))
    
    retriever_docs = [documents[i] for i in retriever_indices]
    uft_docs = [documents[i] for i in uft_indices]
    
    return retriever_docs, uft_docs, None

# experiment path is uptill the contents of the model dir + json is reached
def load_or_score_docs(config):
    repo_id = config.repo_id
    experiment_path = os.path.join("experiments", *config.model.experiment.values())
    docs_save_path = os.path.join(experiment_path, "processed_docs_scores.pkl")
    
    # Try to load from Hub first
    try:
        local_path = hf_hub_download(repo_id=repo_id, filename=docs_save_path)
        with open(local_path, 'rb') as f:
            logger.info(f"Downloading {docs_save_path} from the hub...")
            return pickle.load(f)
        
    except:
        logger.error(f"Could not load pre-processed docs from the Hub. Processing docs now...\n")

    train_dataset = load_json(config.dataset.train)
    r_weights = None
    uft_weights = None
    if config.model.experiment.scoring == "custom-selection":
        if config.model.experiment.variant != 'random':
            r_weights = config.model.scoring.retriever
            uft_weights = config.model.scoring.uft
            existing_docs = Dataset.load_from_disk(config.dataset.base.data)
            retriever_docs, uft_docs, scores = process_documents(train_dataset, existing_docs, r_weights, uft_weights, config.model)
        else:
            retriever_docs, uft_docs, scores = random_split(train_dataset, config.seed)
            
    elif config.model.experiment.scoring == "all-for-finetuning":
        retriever_docs = []
        uft_docs = train_dataset
        scores = None
    elif config.model.experiment.scoring == "all-in-memory":
        retriever_docs = train_dataset
        uft_docs = []
        scores = None
    
    result = (retriever_docs, uft_docs, scores, (r_weights, uft_weights))
    
    # Save locally first
    with tempfile.NamedTemporaryFile(mode='wb', delete=False) as temp_file:
        pickle.dump(result, temp_file)
    
    # Upload to Hub
    upload_file(
        path_or_fileobj=temp_file.name,
        path_in_repo=docs_save_path,
        repo_id=repo_id,
    )
    
    # Clean up temporary file
    os.unlink(temp_file.name)
    
    return result