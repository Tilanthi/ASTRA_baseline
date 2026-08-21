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
Atomic Commit Workflow Module
==============================

Implements GSD's atomic commit pattern with structured git workflow.

Every task completion results in:
- Atomic commit with clear message
- SUMMARY.md documenting outcomes
- STATE.md update with decisions and position

Based on: https://github.com/glittercowboy/get-shit-done
"""

from __future__ import annotations
import os
import subprocess
import json
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Callable
from enum import Enum
from datetime import datetime

try:
    from .xml_task_formatting import XMLTask, TaskStatus
except ImportError:
    # Define minimal types if import fails
    class XMLTask:
        id: str = ""
        name: str = ""
        status: str = "pending"

    class TaskStatus:
        PENDING = "pending"
        COMPLETED = "completed"
        FAILED = "failed"


class CommitStrategy(Enum):
    """Commit strategy options."""
    ATOMIC = "atomic"           # One commit per task
    BATCH = "batch"             # Batch multiple tasks
    SQUASH = "squash"           # Squash into single commit
    INTERACTIVE = "interactive" # Interactive git rebase


@dataclass
class TaskCompletion:
    """
    Result of task completion with commit info.

    Attributes:
        task: The completed task
        success: Whether task completed successfully
        commit_hash: Git commit hash if committed
        commit_message: Commit message used
        files_changed: List of files that were changed
        summary: Summary of what was done
        issues: Any issues encountered
        timestamp: Completion timestamp
    """
    task: XMLTask
    success: bool
    commit_hash: str = ""
    commit_message: str = ""
    files_changed: List[str] = field(default_factory=list)
    summary: str = ""
    issues: List[str] = field(default_factory=list)
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())
