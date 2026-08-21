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
Brownfield Codebase Analyzer
=============================

Implements GSD's parallel codebase mapping for existing projects.

Spawns parallel agents to analyze codebase and creates documentation:
- STACK.md - Languages, frameworks, dependencies
- ARCHITECTURE.md - Patterns, layers, data flow
- STRUCTURE.md - Directory layout, where things live
- CONVENTIONS.md - Code style, naming patterns
- TESTING.md - Test framework, patterns
- INTEGRATIONS.md - External services, APIs
- CONCERNS.md - Tech debt, known issues, fragile areas

Based on: https://github.com/glittercowboy/get-shit-done
"""

from __future__ import annotations
import os
import re
import ast
import json
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Set, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import Counter, defaultdict


@dataclass
class StackDocument:
    """Technology stack documentation."""
    languages: Dict[str, int] = field(default_factory=dict)  # language -> file count
    frameworks: List[str] = field(default_factory=list)
    dependencies: Dict[str, List[str]] = field(default_factory=dict)  # file -> imports
    build_tools: List[str] = field(default_factory=list)
    package_managers: List[str] = field(default_factory=list)

    def to_markdown(self) -> str:
        """Generate STACK.md content."""
        lines = [
            "# Technology Stack",
            "",
            "## Languages",
            ""
        ]

        for lang, count in sorted(self.languages.items(), key=lambda x: -x[1]):
            lines.append(f"- {lang}: {count} files")

        lines.extend([
            "",
            "## Frameworks",
            ""
        ])

        for fw in self.frameworks:
            lines.append(f"- {fw}")

        if self.build_tools:
            lines.extend(["", "## Build Tools", ""])
            for tool in self.build_tools:
                lines.append(f"- {tool}")

        if self.package_managers:
            lines.extend(["", "## Package Managers", ""])
            for pm in self.package_managers:
                lines.append(f"- {pm}")

        lines.extend(["", "## Key Dependencies", ""])

        # Aggregate top dependencies
        dep_counter = Counter()
        for deps in self.dependencies.values():
            dep_counter.update(deps)

        for dep, count in dep_counter.most_common(20):
            lines.append(f"- {dep} (used in {count} files)")

        return "\n".join(lines)


@dataclass
class ArchitectureDocument:
    """Architecture documentation."""
    patterns: List[str] = field(default_factory=list)
    layers: List[str] = field(default_factory=list)
    data_flow: List[str] = field(default_factory=list)
    components: Dict[str, List[str]] = field(default_factory=dict)

    def to_markdown(self) -> str:
        """Generate ARCHITECTURE.md content."""
        lines = [
            "# Architecture",
            "",
            "## Design Patterns",
            ""
        ]

        for pattern in self.patterns:
            lines.append(f"- {pattern}")

        lines.extend(["", "## System Layers", ""])

        for layer in self.layers:
            lines.append(f"- {layer}")
