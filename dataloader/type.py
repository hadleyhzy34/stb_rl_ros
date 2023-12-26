from pydantic import BaseModel, Field
from typing import List, Optional
import numpy as np


class SarsaModel(BaseModel):
    state: List[List[float]]
    goal: List[float]
    reward: float
    next_state: List[List[float]]
    next_state_goal: List[float]
    done: bool
    action: List[float]
    truncated: bool
    state_path_x: Optional[List[float]] = None
    state_path_y: Optional[List[float]] = None
    next_state_path_x: Optional[List[float]] = None
    next_state_path_y: Optional[List[float]] = None


# class SarsaModel(BaseModel):
#     state: np.ndarray
#     reward: float
#     next_state: np.ndarray
#     done: bool
#     action: np.ndarray
#     truncated: bool
