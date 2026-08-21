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
Consciousness Simulator for STAR-Learn V2.5

This module implements metacognitive consciousness simulation:
1. Self-awareness of reasoning processes
2. Introspection and self-reflection
3. Metacognitive monitoring
4. Attention control and focus
5. Theory of Mind (reasoning about others)
6. Qualia simulation (subjective experience)
7. Stream of consciousness processing
8. Meta-reasoning about reasoning

This is a FRONTIER AGI CAPABILITY - simulating conscious awareness
is key to human-level reasoning and understanding.

Version: 2.5.0
Date: 2026-03-16
"""

import numpy as np
from typing import Dict, List, Optional, Any, Tuple, Set
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime
import time


class ConsciousState(Enum):
    """States of consciousness"""
    AWAKE = "awake"
    FOCUSED = "focused"
    MEDITATING = "meditating"
    DREAMING = "dreaming"
    FLOW = "flow"
    CONFUSED = "confused"
    AWARE = "aware"


class MentalProcess(Enum):
    """Types of mental processes"""
    PERCEPTION = "perception"
    ATTENTION = "attention"
    MEMORY = "memory"
    REASONING = "reasoning"
    IMAGINATION = "imagination"
    EMOTION = "emotion"
    INTENTION = "intention"
    METACOGNITION = "metacognition"


class AttentionMode(Enum):
    """Modes of attention"""
    FOCUSED = "focused"  # Concentrated on one thing
    DIVIDED = "divided"  # Split across multiple things
    SUSTAINED = "sustained"  # Maintained over time
    SELECTIVE = "selective"  # Filtering relevant info
    ALTERNATING = "alternating"  # Switching between tasks
    MINDFUL = "mindful"  # Present-moment awareness


@dataclass
class Thought:
    """A discrete unit of thought"""
    content: str
    process_type: MentalProcess
    confidence: float = 0.5
    emotional_valence: float = 0.0  # -1 to 1
    importance: float = 0.5
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    associations: List[str] = field(default_factory=list)
