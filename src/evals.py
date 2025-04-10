from enum import Enum
from typing import Any, Dict, List, Set, Optional
from tqdm.auto import tqdm
from fuzzywuzzy import fuzz
import pickle

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModel
from .dataloader import InferenceDataset

from enum import Enum
from typing import Dict, List, Set, Tuple, Optional

class SetOperation(Enum):
    UNION = "union"
    INTERSECTION = "intersection"
    NONE = "none"

# NOTE: play with index_sets + include_sets to get an idea
def get_filtered_questions(
    human_annotated_docs: List[dict],
    index_sets: Dict[str, Set[str]],  # dataset_name -> set of doc indices
    include_sets: List[str],  # Names of sets to include
    exclude_sets: Optional[List[str]] = None,  # Names of sets to exclude
    set_operation: SetOperation = SetOperation.UNION
) -> List[dict]:
    """
    Get filtered questions from human-annotated docs based on membership in other sets.
    
    Args:
        human_annotated_docs: List of docs containing QA pairs
        index_sets: Dict mapping dataset name to set of doc indices for all sets from which to derive questions
        include_sets: List of dataset names to include, take intersection of each with human_annotated_docs
        exclude_sets: Optional list of sets of indices to remove
        set_operation: Set operation to perform on intersections from include_sets
    
    Returns:
        List of filtered question dicts
    """
    # Get selected doc indices based on set operation
    selected_sets = [index_sets[name] for name in include_sets]
    if not selected_sets:
        return []
    
    universe_set = set(doc['index'] for doc in human_annotated_docs)
    intersections = [universe_set.intersection(selected_set) for selected_set in selected_sets]
    print(len(set.union(*selected_sets)))
    if set_operation == SetOperation.INTERSECTION:
        target_indices = set.intersection(*intersections)
    elif set_operation == SetOperation.UNION:
        target_indices = set.union(*intersections)
    else:
        target_indices = selected_sets[0]
    
    # Apply exclusions
    if exclude_sets:
        excluded_indices =  set.union(*(index_sets[name] for name in exclude_sets))
        target_indices -= excluded_indices
    '''
    target_indices = set_operation(
        intersection of each selected_set (depends on index_sets and include_sets) with human_annotated_docs
        ) - excluded_indices
    '''
    # Extract questions from matching human-annotated docs
    filtered_questions = []
    for doc in human_annotated_docs:
        if doc['index'] in target_indices:
            questions = [
                {
                    "index": doc['index'],
                    "question": q[0],
                    "answer": q[1][0]['answer'],
                    "context": doc['paras']
                }
                for q in doc['questions']
                if q[1][0]['answer']  # Skip questions without answers
            ]
            filtered_questions.extend(questions)
            
    return filtered_questions


def evaluate_model(model, questions, output_path, batch_size=32, max_seq_len=64, max_new_tokens=128):
    dataset = InferenceDataset(questions, model.retriever.question_encoder_tokenizer)
    dataloader = DataLoader(
        dataset, 
        batch_size=batch_size,
        collate_fn=lambda x: {
            'index': [d['index'] for d in x],
            'input_text': [d['input_text'] for d in x],
            'input_ids': torch.stack([d['input_ids'] for d in x]).to(model.device),
            'attention_mask': torch.stack([d['attention_mask'] for d in x]).to(model.device),
            'reference_text': [d['reference_text'] for d in x],
            'reference_context': [d['reference_context'] for d in x]
        }
    )
    
    results = {}  # doc_index -> list of question results
    for batch in tqdm(dataloader, desc="Inference"):
        outputs = model.generate(
            batch['input_ids'],
            batch['attention_mask'],
            max_new_tokens=max_new_tokens,
            do_sample=False
        )
        
        for idx, output in enumerate(outputs):
            doc_idx = batch['index'][idx]
            if doc_idx not in results:
                results[doc_idx] = []
                
            # Clean generated text
            generated_text = output['generated_text'].split('Answer:')[-1]
            
            metrics = calculate_metrics(
                batch['input_text'][idx],
                generated_text,
                batch['reference_text'][idx],
                output['retrieved_context'],
                batch['reference_context'][idx],
                model.generator_tokenizer,
                model.accelerator,
            )
            
            results[doc_idx].append({
                'question': batch['input_text'][idx],
                'full_generated_text': output['generated_text'],
                'generated_text': generated_text,
                'reference_text': batch['reference_text'][idx],
                'retrieved_context': output['retrieved_context'],
                'reference_context': batch['reference_context'][idx],
                'metrics': metrics
            })
    
    with open(output_path, "wb") as f:
        pickle.dump(results, f)
        
def mean_pooling(model_output, attention_mask):
    """Mean pooling for sentence embeddings."""
    token_embeddings = model_output[0]
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

def get_embeddings(texts, model, tokenizer, accelerator):
    """Get embeddings using HF model."""
    inputs = tokenizer(texts, padding=True, truncation=True, return_tensors="pt").to(accelerator.device)
    with torch.no_grad() and accelerator.autocast():
        outputs = model(**inputs)
    embeddings = mean_pooling(outputs, inputs['attention_mask'])
    return F.normalize(embeddings, p=2, dim=1)

def calculate_metrics(
    question: str,
    generated_answer: str,
    reference_answer: str,
    retrieved_context: str,
    reference_context: List[str],
    tokenizer,
    accelerator,
) -> Dict[str, float]:
    """Enhanced metrics calculation."""
    
    def clean_text(text: str) -> str:
        """Clean text by removing special tokens and roles."""
        # Remove special tokens
        special_tokens = ['<|im_start|>', '<|im_end|>', '</s>', '<pad>']
        for token in special_tokens:
            text = text.replace(token, '')
        
        # Remove roles and clean
        text = ' '.join([t for t in text.split() 
                        if t not in ['system', 'user', 'assistant']])
        return text.lower().strip()
    
    # Clean all texts
    question = clean_text(question)
    generated_answer = clean_text(generated_answer)
    reference_answer = clean_text(reference_answer)
    retrieved_context = clean_text(retrieved_context)
    reference_context = [clean_text(ctx) for ctx in reference_context]
    
    # Load model for semantic similarity (cached after first load)
    model_name = "sentence-transformers/all-mpnet-base-v2"  
    embed_model = AutoModel.from_pretrained(model_name, load_in_8bit=True)
    embed_tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Calculate semantic similarity
    embeddings = get_embeddings(
        [generated_answer, reference_answer],
        embed_model,
        embed_tokenizer,
        accelerator
    )
    semantic_sim = torch.cosine_similarity(embeddings[0], embeddings[1], dim=0).item()
    
    # Get token metrics
    metrics = {
        # 'semantic_similarity': semantic_sim,
        # 'length_ratio': len(generated_answer.split()) / max(len(reference_answer.split()), 1),
        # 'has_answer': not generated_answer.lower().startswith("i don't know"),
        **calculate_token_metrics(
            question,
            generated_answer, 
            reference_answer,
            retrieved_context,
            reference_context,
            semantic_sim,
            embed_model,
            embed_tokenizer,
            accelerator,
            tokenizer
        )
    }
    # metrics = {
    #     **calculate_token_metrics(
    #         question,
    #         generated_answer,
    #         reference_answer,
    #         retrieved_context,
    #         reference_context,
    #         tokenizer
    #     ),
    #     'length_ratio': len(generated_answer.split()) / max(len(reference_answer.split()), 1),
    #     'has_answer': not generated_answer.lower().startswith("i don't know")
    # }
    
    return metrics


def calculate_token_metrics(
    question: str,
    generated_answer: str,
    reference_answer: str,
    retrieved_context: str,
    reference_context: List[str],
    semantic_sim: float,
    embed_model,
    embed_tokenizer,
    accelerator,
    tokenizer,
    similarity_threshold: float = 0.5
) -> Dict[str, float]:
    """
    Calculate evaluation metrics using semantic similarity between tokens.
    
    Args:
        question: Input question
        generated_answer: Model generated answer
        reference_answer: Ground truth answer
        retrieved_context: Retrieved context used for generation
        reference_context: List of ground truth context paragraphs
        semantic_sim: Pre-computed semantic similarity between full answers
        embed_model: Model for computing token embeddings
        embed_tokenizer: Tokenizer for embedding model
        accelerator: Accelerator for device management
        tokenizer: Main tokenizer for getting token sets
        similarity_threshold: Threshold for considering tokens semantically similar
        
    Returns:
        Dict containing evaluation metrics
    """
    def get_token_set_with_embeddings(text: str) -> Tuple[set, torch.Tensor]:
        """Get unique tokens and their embeddings."""
        tokens = tokenizer.encode(text, add_special_tokens=False)
        unique_tokens = set(tokenizer.decode(t) for t in tokens)
        
        # Get embeddings for each unique token
        token_embeddings = get_embeddings(
            list(unique_tokens),
            embed_model,
            embed_tokenizer, 
            accelerator
        )
        return unique_tokens, token_embeddings

    def compute_semantic_overlap(embeddings1, embeddings2):
        similarity_matrix = torch.mm(embeddings1, embeddings2.t())
        # Take maximum match per token in either direction
        matches = max(
            similarity_matrix.max(dim=1)[0].mean(),
            similarity_matrix.max(dim=0)[0].mean()
        )
        return matches.item()

    # Get token sets and embeddings
    gen_tokens, gen_embeddings = get_token_set_with_embeddings(generated_answer)
    ref_tokens, ref_embeddings = get_token_set_with_embeddings(reference_answer)
    ctx_tokens, ctx_embeddings = get_token_set_with_embeddings(retrieved_context)
    
    # Get embeddings for full document context
    doc_tokens = set().union(*[tokenizer.encode(ctx, add_special_tokens=False) for ctx in reference_context])
    doc_tokens = set(tokenizer.decode(t) for t in doc_tokens)
    doc_embeddings = get_embeddings(list(doc_tokens), embed_model, embed_tokenizer, accelerator)

    # Calculate semantic overlap
    common_count = compute_semantic_overlap(gen_embeddings, ref_embeddings)
    
    # Calculate metrics
    precision = common_count / max(len(gen_tokens), 1)
    recall = common_count / max(len(ref_tokens), 1)
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    # Use precomputed semantic similarity as accuracy
    accuracy = semantic_sim
    
    # Calculate context usage and faithfulness using semantic similarity
    context_overlap = compute_semantic_overlap(gen_embeddings, ctx_embeddings)
    doc_overlap = compute_semantic_overlap(gen_embeddings, doc_embeddings)
    
    context_usage = context_overlap / max(len(gen_tokens), 1)
    faithfulness = doc_overlap / max(len(gen_tokens), 1)

    return {
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'accuracy': accuracy,
        'context_usage': context_usage,
        'faithfulness': faithfulness
    }

# def calculate_token_metrics(
#     question: str,
#     generated_answer: str,
#     reference_answer: str,
#     retrieved_context: str,
#     reference_context: List[str],
#     semantic_sim,
#     tokenizer
# ) -> Dict[str, float]:
#     """Calculate token-based evaluation metrics."""
    
#     def get_token_set(text: str) -> set:
#         """Get unique tokens after proper tokenization and decoding."""
#         # Tokenize and decode to handle subwords properly
#         tokens = tokenizer.encode(text, add_special_tokens=False)
#         return set(tokenizer.decode(t) for t in tokens)
    
#     # Get token sets
#     gen_tokens = get_token_set(generated_answer)
#     ref_tokens = get_token_set(reference_answer)
#     ctx_tokens = get_token_set(retrieved_context)
#     doc_tokens = set().union(*[get_token_set(ctx) for ctx in reference_context])
#     q_tokens = get_token_set(question)
    
#     # Calculate metrics
#     common_tokens = gen_tokens.intersection(ref_tokens)
#     precision = len(common_tokens) / max(len(gen_tokens), 1)
#     recall = len(common_tokens) / max(len(ref_tokens), 1)
#     f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
#     # String similarity using fuzzy matching
#     accuracy = fuzz.token_sort_ratio(generated_answer, reference_answer) / 100
    
#     # Context usage (from retrieved vs. full document context)
#     context_usage = len(gen_tokens.intersection(ctx_tokens)) / max(len(gen_tokens), 1)
#     faithfulness = len(gen_tokens.intersection(doc_tokens)) / max(len(gen_tokens), 1)
    
#     return {
#         'precision': precision,
#         'recall': recall,
#         'f1_score': f1,
#         'accuracy': accuracy,
#         'context_usage': context_usage,
#         'faithfulness': faithfulness,
#         # NOTE: no need since already calculating length ratio (see above)
#         # 'answer_length': len(generated_answer.split())
#     }