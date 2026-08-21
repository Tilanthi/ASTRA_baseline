# Copyright 2026 Glenn J. White
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Sensorimotor System - Interface for embodied interaction with the world

This system provides the bridge between abstract reasoning and physical interaction,
enabling the system to learn through sensorimotor experience.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
import time
import logging
from enum import Enum

# Virtual simulation dependencies (would be real hardware in production)
try:
    import pygame
    import pymunk  # Physics engine
    PHYSICS_AVAILABLE = True
    logging.info("Physics engines (pygame, pymunk) successfully loaded")
except ImportError:
    PHYSICS_AVAILABLE = False
    # Only show warning if explicitly needed, not on general import
    pass


class ModalityType(Enum):
    """Types of sensory modalities"""
    VISION = "vision"
    AUDIO = "audio"
    TOUCH = "touch"
    PROPRIOCEPTION = "proprioception"
    CHEMICAL = "chemical"
    TEMPERATURE = "temperature"


@dataclass
class SensoryInput:
    """Structured sensory input from the environment"""
    modality: ModalityType
    data: np.ndarray
    timestamp: float
    confidence: float = 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class MotorCommand:
    """Motor command for physical interaction"""
    action_type: str
    parameters: Dict[str, Any]
    target: Optional[Tuple[float, float, float]] = None
    duration: float = 1.0
    force: float = 1.0


@dataclass
class WorldAction:
    """Complete action combining perception, decision, and motor execution"""
    action_id: str
    perception: List[SensoryInput]
    decision: Dict[str, Any]
    motor_commands: List[MotorCommand]
    goal: str
    confidence: float = 1.0


@dataclass
class ActionResult:
    """Result of executing an action in the world"""
    success: bool
    sensory_feedback: List[SensoryInput]
    physical_changes: Dict[str, Any]
    error_message: Optional[str] = None
    execution_time: float = 0.0


@dataclass
class Experience:
    """Complete experience record for learning"""
    action: WorldAction
    result: ActionResult
    timestamp: float
    context: Dict[str, Any]
    success: bool = True
