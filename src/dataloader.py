import os
import torch
from transformers import AutoTokenizer, DataCollatorForLanguageModeling, DPRQuestionEncoderTokenizerFast
from torch.utils.data import DataLoader, Dataset
import nltk
import io
import contextlib
with contextlib.redirect_stdout(io.StringIO()):
    nltk.download('punkt', quiet=True)
    nltk.download('punkt_tab', quiet=True)
    
from tqdm.auto import tqdm
from typing import Dict, List, Tuple, Union

from datasets import Dataset as Dataset_hf
from .utils import setup_logger
logger = setup_logger(enable_logging=True)


# uses datasets.Dataset ie from huggingface
def uft_dataloader(
    documents, 
    tokenizer, 
    train_batch_size, 
    val_batch_size, 
    max_seq_len=2048, 
    stride=1024, 
    validation_split=0.0, 
    seed=42
):
    def chunk_document(document, max_seq_len, stride, doc_index):
        text = ' '.join(document['paras'])
        sentences = nltk.sent_tokenize(text)
        chunks = []
        current_chunk_sentences = []
        current_length = 1  # Start with 1 to account for BOS token
        next_sentence_index = 0

        while next_sentence_index < len(sentences):
            sentence = sentences[next_sentence_index]
            sentence_length = len(tokenizer(sentence, add_special_tokens=False).input_ids)

            if current_length + sentence_length <= max_seq_len - 1:
                current_chunk_sentences.append(sentence)
                current_length += sentence_length
                next_sentence_index += 1
            else:
                if len(current_chunk_sentences) == 0:
                    print(f"Warning: Sentence {next_sentence_index} in document {doc_index} is too long, truncating.")
                    truncated_input_ids = tokenizer(sentence, add_special_tokens=False, truncation=True, max_length=max_seq_len-1).input_ids[0]
                    truncated_sentence = tokenizer.decode(truncated_input_ids)
                    chunk_text = truncated_sentence + tokenizer.eos_token
                    chunk_encoded = tokenizer(chunk_text, truncation=True, max_length=max_seq_len, padding="max_length", return_tensors="pt")
                    chunks.append({
                        'input_ids': chunk_encoded['input_ids'],
                        'attention_mask': chunk_encoded['attention_mask']
                    })
                    next_sentence_index += 1
                    current_chunk_sentences = []
                    current_length = 1
                else:
                    chunk_text = ' '.join(current_chunk_sentences) + tokenizer.eos_token
                    chunk_encoded = tokenizer(chunk_text, truncation=True, max_length=max_seq_len, padding="max_length", return_tensors="pt")
                    chunks.append({
                        'input_ids': chunk_encoded['input_ids'],
                        'attention_mask': chunk_encoded['attention_mask']
                    })
                    overlap_sentences = []
                    overlap_length = 1
                    for sent in reversed(current_chunk_sentences):
                        sent_length = len(tokenizer(sent, add_special_tokens=False).input_ids)
                        if overlap_length + sent_length > stride:
                            break
                        overlap_sentences.insert(0, sent)
                        overlap_length += sent_length

                    current_chunk_sentences = overlap_sentences
                    current_length = overlap_length

                    if current_length + sentence_length <= max_seq_len - 1:
                        current_chunk_sentences.append(sentence)
                        current_length += sentence_length
                        next_sentence_index += 1
                    else:
                        current_chunk_sentences = []
                        current_length = 1

        if current_chunk_sentences:
            last_chunk_text = ' '.join(current_chunk_sentences) + tokenizer.eos_token
            last_chunk_encoded = tokenizer(last_chunk_text, truncation=True, max_length=max_seq_len, padding="max_length", return_tensors="pt")
            chunks.append({
                'input_ids': last_chunk_encoded['input_ids'],
                'attention_mask': last_chunk_encoded['attention_mask']
            })
        return chunks

    all_chunks = []
    for doc_index, doc in enumerate(tqdm(documents, desc="Chunking documents")):
        chunks = chunk_document(doc, max_seq_len, stride, doc_index)
        all_chunks.extend(chunks)

    logger.info(f"Total number of chunks: {len(all_chunks)}")

    dataset = Dataset_hf.from_dict({
        "input_ids": [chunk['input_ids'].squeeze().tolist() for chunk in all_chunks],
        "attention_mask": [chunk['attention_mask'].squeeze().tolist() for chunk in all_chunks]
    })

    def prepare_chunk(examples):
        examples["labels"] = examples["input_ids"].copy()
        return examples

    dataset = dataset.map(prepare_chunk, batched=True)

    if validation_split > 0:
        dataset = dataset.train_test_split(test_size=validation_split, seed=seed)
        train_dataset, val_dataset = dataset['train'], dataset['test']
        logger.info(f"Validation dataset size: {len(val_dataset)}")
    else:
        train_dataset = dataset
        val_dataset = []
    logger.info(f"Train dataset size: {len(train_dataset)}")
    
    # Seed torch and generator for deterministic shuffling
    torch.manual_seed(seed)
    generator = torch.Generator().manual_seed(seed)
    
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    train_dataloader = DataLoader(
        train_dataset, 
        batch_size=train_batch_size, 
        shuffle=True, 
        collate_fn=data_collator, 
        generator=generator
    )
    
    if val_dataset:
        val_dataloader = DataLoader(
            val_dataset, 
            batch_size=val_batch_size,
            shuffle=True, 
            collate_fn=data_collator,
            generator=generator
        )
        logger.info(f"Number of batches in validation dataloader: {len(val_dataloader)}")
    else:
        val_dataloader = []
    logger.info(f"Number of batches in train dataloader: {len(train_dataloader)}")
    
    return train_dataloader, val_dataloader


# uses from torch.utils.data.Dataset not the hf Dataset
def sft_dataloader(
    train_data: List[Dict],
    test_data: List[Dict],
    question_encoder_tokenizer: DPRQuestionEncoderTokenizerFast,
    generator_tokenizer,
    train_batch_size: int = 2,
    val_batch_size: int = 2,
    max_seq_len: int = 64,
    validation_split: float = 0.1,
    seed: int = 42,
    # TODO: add this feature, check what to split, test or train, only after inital exps
    return_test: bool = False
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create dataloaders for supervised fine-tuning.
    
    Args:
        train_data: Training data
        test_data: Test data for validation/testing split
        question_encoder_tokenizer: DPR tokenizer for questions
        generator_tokenizer: Generator tokenizer for answers
        train_batch_size: Batch size for training
        val_batch_size: Batch size for validation
        max_seq_len: Maximum sequence length
        validation_split: Fraction of test data to use for validation
        seed: Random seed
    """

    # Create datasets
    train_dataset = SFTDataset(train_data, question_encoder_tokenizer, max_seq_len)
    test_dataset = SFTDataset(test_data, question_encoder_tokenizer, max_seq_len)
    
    # Split test data into validation and test sets
    val_size = int(len(test_dataset) * validation_split)
    test_size = len(test_dataset) - val_size
    
    # Set random seed for reproducibility
    torch.manual_seed(seed)
    generator = torch.Generator().manual_seed(seed)
    val_dataset, test_dataset = torch.utils.data.random_split(
        test_dataset, 
        [val_size, test_size],
        generator=generator
    )
    
    def collate_fn(batch):
        """
        Custom collate function that handles both questions and answers.
        Questions are encoded with DPR tokenizer, answers remain as strings.
        """
        # Stack input tensors
        input_ids = torch.stack([item['input_ids'] for item in batch])
        attention_masks = torch.stack([item['attention_mask'] for item in batch])
        labels = [item['labels'] for item in batch]  # Keep as strings
        
        return {
            'input_ids': input_ids,
            'attention_mask': attention_masks,
            'labels': labels
        }

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=train_batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        pin_memory=True,
        generator=generator
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=val_batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        pin_memory=True,
        generator=generator
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=val_batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        pin_memory=True,
        generator=generator
    )
    
    # Log dataset sizes
    logger.info(
        "\nSFT dataset:"
        f"\n Training samples: {len(train_dataset)}"
        f"\n Validation samples: {len(val_dataset)}"
        f"\n Test samples: {len(test_dataset)}"
        f"\n Training batches: {len(train_loader)}"
        f"\n Validation batches: {len(val_loader)}\n"
    )
    
    return train_loader, val_loader, test_loader

# uses from torch.utils.data.Dataset not the hf Dataset
class SFTDataset(Dataset):
    def __init__(
        self, 
        data: List[Dict], 
        tokenizer: DPRQuestionEncoderTokenizerFast,
        max_seq_len: int,
    ):
        """
        Initialize SFT dataset.
        
        Args:
            data: List of documents with questions and answers
            tokenizer: DPR tokenizer for questions
            max_seq_len: Maximum sequence length
        """
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.questions, self.answers = self._process_data(data)
        
        # Validate data
        assert len(self.questions) == len(self.answers), \
            "Mismatch between questions and answers length"
        
        # Verify max sequence length
        max_question_len = max(
            len(self.tokenizer(q, add_special_tokens=True).input_ids) 
            for q in self.questions
        )
        if max_question_len > max_seq_len:
            logger.warning(
                f"Max question length ({max_question_len}) exceeds max_seq_len ({max_seq_len}). "
                "Some questions will be truncated."
            )

    def _process_data(self, data: List[Dict]) -> Tuple[List[str], List[str]]:
        """Process raw data into questions and answers."""
        questions, answers = [], []
        for doc in data:
            for q in doc["questions"]:
                if q[1][0]["answer"]:  # Check for valid answer
                    questions.append(q[0])
                    answers.append(q[1][0]["answer"])
        return questions, answers

    def __len__(self) -> int:
        """Return dataset size."""
        return len(self.questions)

    def __getitem__(self, idx: int) -> Dict[str, Union[torch.Tensor, str]]:
        """Get dataset item with error handling."""
        if not 0 <= idx < len(self):
            raise IndexError(f"Index {idx} out of range for dataset of size {len(self)}")
            
        questions = self.questions[idx]
        answers = self.answers[idx]

        question_encoding = self.tokenizer(
            questions,
            max_length=self.max_seq_len,
            padding="max_length",
            truncation=True,  # Enable truncation
            return_tensors="pt"
        )

        return {
            "input_ids": question_encoding["input_ids"].squeeze(),
            "attention_mask": question_encoding["attention_mask"].squeeze(),
            "labels": answers
        }
        
        
class InferenceDataset(Dataset):
    def __init__(self, questions, tokenizer, max_seq_len=64):
        self.questions = questions
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
    
    def __len__(self):
        return len(self.questions)
    
    def __getitem__(self, idx):
        tokenizer_input = self.tokenizer(
                self.questions[idx]['question'],
                truncation=True,
                padding='max_length',
                max_length=self.max_seq_len,
                return_tensors='pt'
            )

        return {
            'index': self.questions[idx]['index'],
            'input_text':  self.questions[idx]['question'],
            'input_ids': tokenizer_input.input_ids.squeeze(),
            'attention_mask': tokenizer_input.attention_mask.squeeze(),
            'reference_text': self.questions[idx]['answer'],
            'reference_context': self.questions[idx]['context']
        }
