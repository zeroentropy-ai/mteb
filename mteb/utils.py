import os
import numpy as np
import torch
from mteb.types import PromptType, BatchedInput
from mteb.models.abs_encoder import AbsEncoder
from torch.utils.data import DataLoader
from mteb.abstasks.task_metadata import TaskMetadata
from pydantic import BaseModel, Field, AliasChoices
from typing import Any
from pathlib import Path

Array = np.ndarray | torch.Tensor

ROOT = f"{Path(__file__).resolve().parent.parent}"

class ZeroModel(AbsEncoder):
    def encode(
        self,
        inputs: DataLoader[BatchedInput],
        task_metadata: TaskMetadata,
        hf_split: str,
        hf_subset: str,
        prompt_type: PromptType | None = None,
        **kwargs,
    ) -> np.ndarray:
        sentences = [text for batch in inputs for text in batch["text"]]

        return np.zeros(
            (len(sentences), 768),
            dtype=np.float32
        )
    
    def similarity(
        self,
        embeddings1: Array,
        embeddings2: Array,
        **kwargs,
    ) -> Array:
        return np.zeros((len(embeddings1), len(embeddings2)), dtype=np.float32)

    def similarity_pairwise(
        self,
        embeddings1: Array,
        embeddings2: Array,
        **kwargs,
    ) -> Array:
        return np.zeros((len(embeddings1),), dtype=np.float32)