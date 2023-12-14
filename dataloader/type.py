from pydantic import BaseModel, Field
from typing import List
import numpy as np


class SarsaModel(BaseModel):
    state: List[float]
    reward: float
    next_state: List[float]
    done: bool
    action: List[float]
    truncated: bool


# class SarsaModel(BaseModel):
#     state: np.ndarray
#     reward: float
#     next_state: np.ndarray
#     done: bool
#     action: np.ndarray
#     truncated: bool
