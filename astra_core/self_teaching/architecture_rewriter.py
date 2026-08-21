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
Architecture Rewriter for Autocatalytic Self-Compiler

Performs safe mutations on cognitive architecture with validation,
sandboxing, and rollback capabilities.

Version: 4.0.0
Date: 2026-03-17
"""

from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum
import ast
import copy


class MutationType(Enum):
    """Types of architecture mutations"""
    ADD_MODULE = "add_module"              # Add a new module
    REMOVE_MODULE = "remove_module"        # Remove an existing module
    MODIFY_CONNECTION = "modify_connection"  # Change connection between modules
    ADJUST_PARAMETER = "adjust_parameter"   # Adjust a parameter value
    RESTRUCTURE_HIERARCHY = "restructure"   # Restructure module hierarchy
    OPTIMIZE_FLOW = "optimize_flow"        # Optimize data flow


@dataclass
class Mutation:
    """A proposed mutation to the architecture"""
    mutation_type: MutationType
    target_module: str
    description: str
    changes: Dict[str, Any]
    expected_impact: float  # -1.0 to 1.0
    confidence: float
    safety_score: float
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RewriteResult:
    """Result of an architecture rewrite"""
    success: bool
    applied_mutations: List[Mutation]
    validation_passed: bool
    new_version_id: str
    performance_delta: float = 0.0
    error_message: str = ""
    rollback_available: bool = False


@dataclass
class ValidationResult:
    """Result of architecture validation"""
    is_valid: bool
    safety_score: float
    issues: List[str]
    warnings: List[str]
    estimated_performance: float
