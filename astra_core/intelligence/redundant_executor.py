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
Redundant Execution for Fault Tolerance (Priority 5)
====================================================

Executes 2+ identical agents in parallel for reliability.

Problem Solved:
- APIs time out, models crash, networks drop
- Single point of failure causes entire operation to fail
- Long-tail latency causes unpredictable user experience

Benefits:
- 33% reliability improvement (60% → 80% success rate in article)
- Eliminates long-tail latency events
- Zero additional latency (parallel execution)
- Graceful degradation under adverse conditions

Based on: "Building the 14 Key Pillars of Agentic AI" - Pillar 9

Example Use:
    executor = RedundantExecutor(num_copies=2)
    result = executor.execute(unreliable_api_call, "user_data")
    # If first copy fails or times out, second copy provides result
    # Success: 80% vs 60% for single execution
"""

import time
import random
from dataclasses import dataclass, field
from typing import Callable, Any, List, Dict, Optional, TypeVar
from concurrent.futures import ThreadPoolExecutor, as_completed, Future
from enum import Enum


T = TypeVar('T')


class ExecutionStrategy(Enum):
    """Strategy for redundant execution."""
    FIRST_SUCCESS = "first_success"      # Return first successful result
    MAJORITY_VOTE = "majority_vote"      # Wait for majority, return most common
    ALL_SUCCESS = "all_success"          # Wait for all, verify consistency


@dataclass
class ExecutionResult:
    """Result from redundant execution."""
    success: bool
    result: Optional[Any]
    error: Optional[str]
    execution_time: float
    attempts: int
    successful_attempts: int
    failed_attempts: int
    all_results: List[Any] = field(default_factory=list)
    all_errors: List[str] = field(default_factory=list)
