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
Tool Integration: External APIs and Computation Tools

Provides access to external knowledge sources and computation capabilities:
- Wikipedia API for factual context (free, unlimited)
- arXiv API for research/academic questions (free, unlimited)
- MathTool for symbolic computation (SymPy-based)
- Python executor for safe numerical computation

Expected gain: +5-8% accuracy

Date: 2025-12-10
Version: 38.0
"""

import re
import json
import urllib.request
import urllib.parse
import urllib.error
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
import xml.etree.ElementTree as ET
import math


@dataclass
class ToolResult:
    """Result from a tool query"""
    tool: str
    query: str
    success: bool
    result: Any
    confidence: float
    source_url: Optional[str] = None
    error_message: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __str__(self):
        if self.success:
            result_preview = str(self.result)[:100] + "..." if len(str(self.result)) > 100 else str(self.result)
            return f"ToolResult({self.tool}: {result_preview}, confidence={self.confidence:.2f})"
        return f"ToolResult({self.tool}: FAILED - {self.error_message})"


class WikipediaAPI:
    """
    Wikipedia API for factual context.

    Free, unlimited - always query for factual context.
    Uses the Wikipedia REST API v1.
    """

    BASE_URL = "https://en.wikipedia.org/api/rest_v1"
    SEARCH_URL = "https://en.wikipedia.org/w/api.php"
