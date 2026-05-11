import numpy as np
import faiss
import torch
from transformers import (
    DPRContextEncoder,
    DPRContextEncoderTokenizer,
    DPRQuestionEncoder,
    DPRQuestionEncoderTokenizer,
)

class DPRRetriever:
    def __init__(
        self,
        context_encoder_name,
        question_encoder_name,
        max_context_length=256,
    ):
        self.max_context_length = max_context_length

        self.context_tokenizer = DPRContextEncoderTokenizer.from_pretrained(
            context_encoder_name
        )
        self.context_encoder = DPRContextEncoder.from_pretrained(
            context_encoder_name
        )

        self.question_tokenizer = DPRQuestionEncoderTokenizer.from_pretrained(
            question_encoder_name
        )
        self.question_encoder = DPRQuestionEncoder.from_pretrained(
            question_encoder_name
        )

        self.index = None
        self.paragraphs = None

    def encode_contexts(self, paragraphs):
        embeddings = []

        for paragraph in paragraphs:
            inputs = self.context_tokenizer(
                paragraph,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self.max_context_length,
            )

            with torch.no_grad():
                outputs = self.context_encoder(**inputs)

            embeddings.append(outputs.pooler_output)

        embeddings = torch.cat(embeddings).detach().numpy(). # Convert list of numpy arrays into a single numpy array
        return embeddings.astype("float32")
    
    def build_index(self, paragraphs):

        self.paragraphs = paragraphs
        context_embeddings = self.encode_contexts(paragraphs)

        embedding_dim = context_embeddings.shape[1]  # This should match the dimension of your embeddings
        # Create a FAISS index for the embeddings
        self.index = faiss.IndexFlatL2(embedding_dim)
        self.index.add(context_embeddings) # Add the context embeddings to the index

    def search(self, question, top_k=5):
        """
        Searches for the most relevant contexts to a given question.

        Returns:
        tuple: Distances and indices of the top k relevant contexts.
        """
        if self.index is None:
            raise ValueError("FAISS index has not been built yet.")

        question_inputs = self.question_tokenizer(
            question,
            return_tensors="pt",
            padding=True,
            truncation=True,
        )

        with torch.no_grad():
            question_embedding = self.question_encoder(
                **question_inputs
            ).pooler_output

        question_embedding = question_embedding.detach().numpy().astype("float32")

        distances, indices = self.index.search(question_embedding, top_k)

        retrieved_contexts = [self.paragraphs[idx] for idx in indices[0]]

        return retrieved_contexts, distances[0]





