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
