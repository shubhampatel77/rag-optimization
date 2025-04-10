import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import (
    AutoTokenizer, DPRQuestionEncoder, DPRQuestionEncoderTokenizer,
    RagRetriever, AutoModelForCausalLM, get_linear_schedule_with_warmup
)
from torch.utils.data import DataLoader, Dataset
import bitsandbytes as bnb
from accelerate import Accelerator
from tqdm.auto import tqdm
from typing import List, Tuple, Dict, Union, Optional
from trl import setup_chat_format
from huggingface_hub import HfApi, Repository
from .utils import setup_logger
logger = setup_logger(enable_logging=True)

class RAGSequence(nn.Module):
    """
    RAG implementation supporting decoder-only LLMs with joint optimization.
    
    This model implements a novel probability marginalization approach for combining retriever and generator log probabilities.
    It follows the formulation: p_θ(y | z_k, x) ≈ p_θ(y | f(z_k, x)), where y is the output sequence, z_k is the kth 
    retrieved document latent, x is the input sequence, and f() represents the prompt augmentation function (system prompt, 
    context, instructions, etc.) for the generator LLM.
    
    Args:
        question_encoder (`DPRQuestionEncoder`):
            Dense Passage Retrieval (DPR) question encoder for converting queries into vector representations.
        retriever (`RagRetriever`):
            Document retriever component for fetching relevant document chunks from the knowledge base.
        generator_tokenizer (`AutoTokenizer`):
            Tokenizer for the decoder-only language model.
        generator (`AutoModelForCausalLM`):
            Decoder-only language model used for generating responses.
        accelerator (`Accelerator`):
            Accelerator instance for device management and mixed precision training.
        generator_max_seq_len_train (`int`, *optional*, defaults to 512):
            Maximum sequence length for generator used during training.
        question_max_seq_len_inference (`int`, *optional*, defaults to 64):
            Maximum sequence length for question encoding during inference.
        generator_max_seq_len_inference (`int`, *optional*, defaults to 1024):
            Maximum sequence length for generator inputs during inference.
    """
    
    def __init__(
        self, 
        question_encoder: DPRQuestionEncoder, 
        retriever: RagRetriever, 
        generator_tokenizer: AutoTokenizer, 
        generator: AutoModelForCausalLM, 
        accelerator: Accelerator, 
        generator_max_seq_len_train: int = 512,
        question_max_seq_len_inference: int = 64,
        generator_max_seq_len_inference: int = 1024
    ):
    
        super().__init__()
        self.question_encoder = question_encoder
        self.retriever = retriever
        self.generator_tokenizer = generator_tokenizer
        self.generator = generator
        
        self.generator_max_seq_len_train = generator_max_seq_len_train
        self.question_max_seq_len_inference = question_max_seq_len_inference
        self.generator_max_seq_len_inference = generator_max_seq_len_inference
        
        self.accelerator = accelerator
        self.device = accelerator.device
    
    def generate_with_text(
        self, 
        inputs: Union[str, List[str]], 
        do_retrieval: bool = False, 
        num_docs: int = 5,
        skip_special_tokens: bool = True,
        instruction: Optional[str] = None,
        system_prompt: Optional[str] = None,
        generator_max_length: int = 128,
        **kwargs
    ) -> List[Dict[str, str]]:
        """
        Generate text responses from raw string inputs with optional document retrieval.
        
        This method wraps tokenization and generation into a single function call for convenience,
        handling both RAG and direct generation modes.
        
        Args:
            inputs (`Union[str, List[str]]`):
                A single input string or a list of input strings representing questions or prompts.
            do_retrieval (`bool`, *optional*, defaults to False):
                If True, performs retrieval-augmented generation (RAG). If False, only the generator is used.
            num_docs (`int`, *optional*, defaults to 5):
                Number of documents to retrieve per query if `do_retrieval=True`.
            skip_special_tokens (`bool`, *optional*, defaults to True):
                Whether to skip special tokens in the generated output.
            instruction (`str`, *optional*):
                Optional instruction string for guiding generation, included in the prompt if provided.
            system_prompt (`str`, *optional*):
                Optional system message used in chat-style prompts.
            generator_max_length (`int`, *optional*, defaults to 128):
                Maximum length (in tokens) of the input passed to the generator.
            **kwargs:
                Additional keyword arguments passed to `generate()`.
                
        Returns:
            `List[Dict[str, str]]`: A list of dictionaries, each containing:
                - `generated_text`: the generated response string
                - `retrieved_context` (optional): retrieved context if `do_retrieval=True`
        """
        
        if isinstance(inputs, str):
            inputs = [inputs]

        if do_retrieval:
            encoded_inputs = self.retriever.question_encoder_tokenizer(
                inputs,
                return_tensors='pt',
                truncation=True,
                padding='max_length',
                max_length=self.question_max_seq_len_inference
            )
            encoded_inputs = {
                k: v.to(self.device)
                for k, v in encoded_inputs.items()
                if k in ['input_ids', 'attention_mask']
            }
        else:
            encoded_inputs = self.generator_tokenizer(
                inputs,
                return_tensors="pt",
                truncation=True,
                padding='max_length',
                max_length=generator_max_length,
            ).to(self.device)
        
        return self.generate(**encoded_inputs, do_retrieval=do_retrieval, **kwargs)
    
    def generate(
        self, 
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        do_retrieval: bool,
        num_docs: int = 5, 
        skip_special_tokens: bool = True,
        instruction: Optional[str] = None,
        system_prompt: Optional[str] = None,
        num_return_sequences: int = 1, 
        max_new_tokens: int = 128, 
        no_repeat_ngram_size: int = 3, 
        do_sample: bool = False, 
        temperature: float = 1.0,
        top_p: float = 1.0,
        top_k: int = 50,
        **kwargs
    ) -> List[Dict[str, str]]:   
        """
        Perform batched text generation with optional document retrieval.
        
        This function supports both standard generation and Retrieval-Augmented Generation (RAG)
        depending on the `do_retrieval` flag. It is designed for inference and supports a range
        of decoding parameters.
        
        Args:
            input_ids (`torch.Tensor`):
                Tokenized input IDs of shape `(batch_size, self.question_max_seq_len_inference)`.
            attention_mask (`torch.Tensor`):
                Attention mask of shape `(batch_size, self.question_max_seq_len_inference)`.
            do_retrieval (`bool`):
                If True, performs document retrieval and prompt augmentation (RAG mode).
                If False, uses only the generator for generation.
            num_docs (`int`, *optional*, defaults to 5):
                Number of documents to retrieve per query.
            skip_special_tokens (`bool`, *optional*, defaults to True):
                Whether to skip special tokens in the final generated output.
            instruction (`str`, *optional*):
                Optional instruction string to include in the prompt.
            system_prompt (`str`, *optional*):
                Optional system prompt string for the chat format.
            num_return_sequences (`int`, *optional*, defaults to 1):
                Number of sequences to generate per input.
            max_new_tokens (`int`, *optional*, defaults to 128):
                Maximum number of tokens to generate per output.
            no_repeat_ngram_size (`int`, *optional*, defaults to 3):
                Size of n-grams that should not be repeated.
            do_sample (`bool`, *optional*, defaults to False):
                Whether to sample from the logits distribution (vs. greedy decoding).
            temperature (`float`, *optional*, defaults to 1.0):
                Sampling temperature for softmax distribution.
            top_p (`float`, *optional*, defaults to 1.0):
                Nucleus sampling: cumulative probability cutoff.
            top_k (`int`, *optional*, defaults to 50):
                Top-k sampling: limits sampling to top-k logits.
            **kwargs:
                Additional keyword arguments for `generate()` from `AutoModelForCausalLM`.
                
        Returns:
            `List[Dict[str, str]]`: A list of dictionaries, each containing:
                - `generated_text`: the full generated output
                - `retrieved_context` (optional): the context used for RAG if `do_retrieval=True`
        """
        
        if do_retrieval:
            # Returns top num_docs documents
            retrieved_doc_ids, retriever_scores = self.retrieve(
                input_ids, 
                attention_mask,
                num_docs=num_docs
            )
            
            contexts = []
            for b in range(input_ids.shape[0]):
                batch_contexts = []
                for d in range(retrieved_doc_ids.shape[1]):
                    doc = self.retriever.index.dataset[int(retrieved_doc_ids[b,d].item())]
                    batch_contexts.append(f"Title: {doc['title']}\nText: {doc['text']}")
                contexts.append("\n\n".join(batch_contexts))
                
            if instruction is None:
                instruction = (
                    "Use the given context only if relevant to answer the question.\n"
                    "Otherwise, if you don't know, just say \"I don't know.\"\n"
                )
        else:
            if instruction is None:
                instruction = (
                   "Answer the question to the best of your ability.\n"
                    "If you don't know, just say \"I don't know.\"\n"
                )
                
        if system_prompt is None:
            system_prompt = "You are a friendly chatbot who always responds precisely."
            
        # Handle batch of questions
        batch_prompts = []

        for i in range(input_ids.shape[0]):
            if do_retrieval:
                question = self.retriever.question_encoder_tokenizer.decode(
                    input_ids[i], 
                    skip_special_tokens=True
                )
                user_prompt = (
                    f"{instruction}\n"
                    f"Context:\n{contexts[i]}\n"  # TODO: ability to provide custom context
                    f"Question: {question}\n"
                    "Answer:"
                )
            else:
                # directly use text input and pass to generator
                question = self.generator_tokenizer.decode(
                    input_ids[i], 
                    skip_special_tokens=True
                )
                user_prompt = (
                    f"{question}\n"
                    "Answer:"
                )

            
            message = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]
            
            # no max_length, since not much variation in it, since context is ~ 100 words
            # and question is < 64 so 64 + 5*100 + 100 (say from instruction + system prompt) < 1024
            prompt = self.generator_tokenizer.apply_chat_template(
                message,
                tokenize=True,
                add_generation_prompt=True,
                truncation=True,
                padding='max_length',  # see above why
                padding_side="left",  # "Since LLMs are not trained to continue from pad tokens, your input needs to be left-padded."
                max_length=self.generator_max_seq_len_inference,  # TODO: create class variable
                return_tensors="pt"
            )
            batch_prompts.append(prompt)

        # Stack all prompts into batch
        generator_input_ids = torch.cat(batch_prompts, dim=0).to(self.device) 
        
        generator_attention_mask = (generator_input_ids != self.generator_tokenizer.pad_token_id).long().to(self.device)
    
        generation_kwargs = {
            'pad_token_id': self.generator_tokenizer.pad_token_id,
            'num_return_sequences': num_return_sequences,
            'max_new_tokens': max_new_tokens,
            'no_repeat_ngram_size': no_repeat_ngram_size,
            'do_sample': do_sample,
            'temperature': temperature,
            'top_p': top_p,
            'top_k': top_k,
            **kwargs
        }
        
        with self.accelerator.autocast():
            outputs = self.generator.generate(
                input_ids=generator_input_ids,
                attention_mask=generator_attention_mask, 
                **generation_kwargs
            )
            
        responses = self.generator_tokenizer.batch_decode(
            outputs, 
            skip_special_tokens=skip_special_tokens
        )
        
        batch_outputs = []
        for i, resp in enumerate(responses):
            output = {"generated_text": resp}
            if do_retrieval:
                output["retrieved_context"] = contexts[i]
            batch_outputs.append(output)
        
        return batch_outputs
        
    def prepare_for_sft(self) -> None:
        """
        Prepare the model for supervised finetuning (SFT).
        
        This method sets up the chat format and ensures a custom padding token is added to the 
        generator tokenizer. The model's embedding matrix vocabulary size is resized accordingly.
        """
        
        if self.generator_tokenizer.chat_template:
            self.generator_tokenizer.chat_template = None
            
        pad_token = self.generator_tokenizer.pad_token
        vocab_size = self.generator.config.vocab_size
        embedding_dim_before = self.generator.get_input_embeddings().weight.shape
        
        # setup_chat_format() modifies pad token to <|im_end|> so add back pad_token again
        # also resizes embedding matrix vocab_size
        logger.info("Running setup_chat_format()...")
        self.generator, self.generator_tokenizer = setup_chat_format(self.generator, self.generator_tokenizer)
        self.generator_tokenizer.pad_token = pad_token
        
        embedding_dim_after = self.generator.get_input_embeddings().weight.shape
        logger.info(f"Embedding matrix before: {embedding_dim_before}, after setup_chat_format(): {embedding_dim_after}")
        
        # SEE Llama tokenizers and their training strategy (right padding for training, left for inference)
        
    def add_pad_token(self, pad_token: str) -> None:
        """
        Add a custom pad token to the generator tokenizer.
        
        Args:
            pad_token (`str`):
                Token to use as padding token.
        """

        vocab_size = self.generator.config.vocab_size
        
        self.generator_tokenizer.add_tokens([pad_token])
        self.generator_tokenizer.pad_token = pad_token
        # resize for pad_token
        logger.info("Resizing model's embedding matrix vocab_size...")
        self.generator.resize_token_embeddings(len(self.generator_tokenizer))
        logger.info(f"Resized from {vocab_size} to {len(self.generator_tokenizer)}")

    def forward(
        self, 
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor, 
        answer_texts: List[str], 
        debug_print: bool = False
    ) -> Optional[torch.Tensor]:
        """
        Forward pass through RAGSequence model.
        
        Retrieves documents, generates sequences and computes loss.
        
        Args:
            input_ids (`torch.Tensor`):
                Tensor of shape `(batch_size, max_seq_len)` containing tokenized input IDs using 
                DPRQuestionEncoderTokenizer where max_seq_len is the maximum sequence length for 
                DPRQuestionEncoder during training.
            attention_mask (`torch.Tensor`):
                Tensor of shape `(batch_size, max_seq_len)` for question encoder.
            answer_texts (`List[str]`):
                List of answer strings of length `batch_size`.
            debug_print (`bool`, *optional*, defaults to False):
                Whether to print debug information.
                
        Returns:
            `torch.Tensor`, *optional*: Loss tensor if labels are provided, else None.
        """
        
        # Retrieve documents
        retrieved_doc_ids, retriever_scores = self.retrieve(input_ids, attention_mask)
        
        # Generate sequences for each retrieved document
        logits, labels = self.generate_sequences(
            input_ids, retrieved_doc_ids, answer_texts, debug_print=debug_print
        )
        
        # Compute loss if labels are provided
        loss = None
        if labels is not None:
            loss = self.compute_loss(retriever_scores, logits, labels, debug_print=debug_print)
        return loss
    
    def retrieve(
        self, 
        input_ids: torch.Tensor, 
        attention_mask: torch.Tensor, 
        num_docs: int = 5
    ) -> Tuple[torch.Tensor, torch.Tensor]:  
        """
        Retrieve top documents using Dense Passage Retrieval (DPR).
        
        Args:
            input_ids (`torch.Tensor`):
                Tensor of shape `(batch_size, max_seq_len)` containing tokenized input IDs using 
                DPRQuestionEncoderTokenizer where max_seq_len is the maximum sequence length for 
                DPRQuestionEncoder during training.
            attention_mask (`torch.Tensor`):
                Tensor of shape `(batch_size, max_seq_len)` for question encoder.
            num_docs (`int`, *optional*, defaults to 5):
                Number of documents to retrieve.
                
        Returns:
            `Tuple[torch.Tensor, torch.Tensor]`:
                - retrieved_doc_ids: Tensor of shape `(batch_size, num_docs)` containing retrieved document indices.
                - retriever_scores: Tensor of shape `(batch_size, num_docs)` containing retriever scores.
        """
        
        question_hidden_states = self.question_encoder(input_ids=input_ids, attention_mask=attention_mask)[0]
        docs_dict = self.retriever(
            input_ids.cpu().numpy(),
            question_hidden_states.cpu().detach().numpy(),
            n_docs=num_docs,
            return_tensors="pt"
        )
        retrieved_doc_ids = docs_dict['doc_ids'].to(self.device) # batch_size x num_docs

        # docs_dict['retrieved_doc_embeds'].shape = batch_size x num_docs x embedding_dim (768)
        
        retriever_scores = torch.bmm(
            question_hidden_states.cpu().detach().unsqueeze(1), docs_dict["retrieved_doc_embeds"].float().transpose(1, 2)
        ).squeeze(1).to(self.device) # batch size x num_docs
        
        return retrieved_doc_ids, retriever_scores

    def generate_sequences(
        self, 
        input_ids: torch.Tensor, 
        retrieved_doc_ids: torch.Tensor, 
        answer_texts: List[str],
        debug_print: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Construct input prompts and target labels for the generator.
        
        Args:
            input_ids (`torch.Tensor`):
                Tensor of shape `(batch_size, max_seq_len)` containing tokenized input IDs using
                DPRQuestionEncoderTokenizer where max_seq_len is the maximum sequence length for 
                DPRQuestionEncoder during training.
            retrieved_doc_ids (`torch.Tensor`):
                Tensor of shape `(batch_size, num_docs)` containing retrieved document indices.
            answer_texts (`List[str]`):
                List of answer strings of length `batch_size`.
            debug_print (`bool`, *optional*, defaults to False):
                Whether to print debug information.
                
        Returns:
            `Tuple[torch.Tensor, torch.Tensor]`:
                - logits: Tensor of shape `(batch_size * num_docs, self.generator_max_seq_len_train, vocab_size)` for generator.
                - labels: Tensor of shape `(batch_size * num_docs, self.generator_max_seq_len_train)` for generator.
        """
    
        batch_size, num_docs = retrieved_doc_ids.shape
        total_samples = batch_size * num_docs
        generator_input_ids = torch.zeros((total_samples, self.generator_max_seq_len_train), dtype=torch.long).to(self.device)

        for i in range(batch_size):
            question = self.retriever.question_encoder_tokenizer.decode(input_ids[i], skip_special_tokens=True)
            for j in range(num_docs):
                doc_id = retrieved_doc_ids[i, j]
                doc = self.retriever.index.dataset[int(doc_id.item())]
                context = f"Title: {doc['title']}\nText: {doc['text']}"
                
                prompt = (
                        "Use the given context only if relevant to answer the question.\n"
                        "If you don't know or if the answer is not in the context, just say \"I don't know.\"\n"
                        f"Context:\n{context}\n"
                        f"Question: {question}\n"
                        f"Answer: {answer_texts[i]}"
                    )
                message = [
                    {"role": "system", "content": "You are a friendly chatbot who always responds precisely."},
                    {"role": "user", "content": prompt}
                ]
                
                # NOTE:
                # 279 is the max prompt length for num_docs = 5, 421 for 10, so generator_tokenizer's 
                # max seq length 512 balances both RAM and undue truncation
                generator_input_ids[i * num_docs + j] = self.generator_tokenizer.apply_chat_template(
                    message,
                    tokenize=True,
                    return_tensors="pt",
                    truncation=True,
                    padding="max_length",
                    max_length=self.generator_max_seq_len_train
                )

        generator_attention_mask = (generator_input_ids != self.generator_tokenizer.pad_token_id).long()

        labels = torch.full_like(generator_input_ids, fill_value=-100)
        answer_token = self.generator_tokenizer.encode("Answer:", add_special_tokens=False)[-1]
        
        for i in range(total_samples):
            answer_start = (generator_input_ids[i] == answer_token).nonzero(as_tuple=True)[0]
            if len(answer_start) > 0:
                # +1 to move past "Answer:"
                answer_start = answer_start[-1].item() + 1
                labels[i, answer_start:] = generator_input_ids[i, answer_start:]

        labels = labels.masked_fill(generator_attention_mask == 0, -100)
        
        # Debug with prints      
        if debug_print:
            logger.info(
                f"\n  Generator Input IDs Shape: {generator_input_ids.shape}"
                f"\n  Generator Attention Mask Shape: {generator_attention_mask.shape}"
                f"\n  Labels Shape: {labels.shape}"
            )
            
            # Decode one sequence for verification (just the first one in batch_size * num_docs)
            for skip_special_tokens in [False, True]:
                decoded_prompt = self.generator_tokenizer.decode(generator_input_ids[0], skip_special_tokens=skip_special_tokens)
                decoded_label = self.generator_tokenizer.decode(
                    labels[0][labels[0] != -100], 
                    skip_special_tokens=skip_special_tokens
                )
                logger.info(
                    f"======= DECODED (INPUT, TARGET) TOKENS, with skip_special_tokens: {skip_special_tokens} =======\n"
                    f"Decoded Prompt: {decoded_prompt}\n"
                    f"Decoded Label: {decoded_label}\n"
                )
                
        # Move to the appropriate device
        generator_input_ids = generator_input_ids.to(self.device)
        generator_attention_mask = generator_attention_mask.to(self.device)
        labels = labels.to(self.device)
        
        # Forward pass through generator, passing labels for loss calculation
        outputs = self.generator(input_ids=generator_input_ids, attention_mask=generator_attention_mask, labels=labels)
        
        # Print the model's loss for debugging
        if debug_print:
            logger.info(f"Model Loss (from src code): {outputs.loss}")
        
        return outputs.logits, labels

    def compute_loss(
        self, 
        retriever_scores: torch.Tensor, 
        logits: torch.Tensor, 
        labels: torch.Tensor,
        debug_print: bool = False
    ) -> torch.Tensor:
        """
        Compute loss for joint optimization of retriever and generator components.
        
        The generator loss is calculated using cross-entropy over token logits, while the 
        retriever loss is implicitly computed through marginalization over retrieved contexts.
        
        Args:
            retriever_scores (`torch.Tensor`):
                Tensor of shape `(batch_size, num_docs)` representing the relevance scores 
                assigned by the retriever to each document per query.
            logits (`torch.Tensor`):
                Tensor of shape `(batch_size * num_docs, self.generator_max_seq_len_train, vocab_size)` representing 
                the generator's predicted token distributions.
            labels (`torch.Tensor`):
                Tensor of shape `(batch_size * num_docs, self.generator_max_seq_len_train)` containing the token-level 
                ground truth, where ignored positions are typically marked with -100.
            debug_print (`bool`, *optional*, defaults to False):
                Whether to print internal stats for debugging.
                
        Returns:
            `torch.Tensor`: A scalar tensor representing the combined loss value.
        """
        
        vocab_size = logits.shape[-1]
        batch_size, num_docs = retriever_scores.shape

        # Shift logits and labels
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        
        # Reshape tensors
        shift_logits = shift_logits.view(batch_size, num_docs, -1, vocab_size)
        shift_labels = shift_labels.view(batch_size, num_docs, -1)

        # Compute log probabilities
        log_probs = F.log_softmax(shift_logits, dim=-1)

        # Create mask for non-ignored tokens
        mask = (shift_labels != -100)
        
        # Clamp labels to valid range, to prevent CUDA assertion triggers
        # since dim_idx = 0 to vocab_size - 1, and labels should span that
        # hence ignore -100 indexed log_probs
        shift_labels = torch.clamp(shift_labels, 0, vocab_size - 1)

        # Gather target log probabilities
        target_log_probs = log_probs.gather(dim=-1, index=shift_labels.unsqueeze(-1)).squeeze(-1)

        # Apply mask and compute sequence log probabilities
        masked_log_probs = target_log_probs * mask.float()
        seq_log_probs = masked_log_probs.sum(-1)
        
        if debug_print:
            logger.info(f"seq_log_probs:\n{seq_log_probs}")

        # Compute generator loss, see notes and torch.CrossEntropy() docs
        # N spans batch_size, num_docs and self.generator_max_seq_len_train,
        # so avg not by N but by N - |labels == -100| = mask.sum()
        gen_loss = -masked_log_probs.sum()/mask.sum()
        if debug_print:
            logger.info(f"Calculated loss: {gen_loss}")

        # Compute retriever log probabilities
        retriever_log_probs = F.log_softmax(retriever_scores, dim=-1)
        print(retriever_log_probs)
        # Combine sequence and retriever log probabilities
        total_log_probs = seq_log_probs + retriever_log_probs

        # Marginalize over documents
        marginalized_log_probs = torch.logsumexp(total_log_probs, dim=1)

        # Compute final loss (negative log likelihood)
        loss = -marginalized_log_probs.mean()
        if debug_print:
            print(f"Final computed loss: {loss}")

        return loss
    
    # Other cleanup options like accelerate.clear(), gc.collect(), torch.cuda.ipc_collect()
    # empty CUDA cache, del model don't seem to work, see 03/15 TODO point 3.
    def to_cpu(self) -> None:
        """
        Move all model components to CPU.
        
        This is useful for post-training cleanup and memory management.
        """

        self.question_encoder.to('cpu')
        self.generator.to('cpu')
        if hasattr(self, 'retriever') and hasattr(self.retriever, 'to'):
            self.retriever.to('cpu')
