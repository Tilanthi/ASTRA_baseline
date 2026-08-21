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
Alert Processing Module

Real-time transient alert stream handling for ZTF, Rubin/LSST,
and other time-domain surveys.

Date: 2025-12-15
"""

import numpy as np
from typing import List, Dict, Optional, Any, Tuple, Callable
from dataclasses import dataclass, field
from enum import Enum
from abc import ABC, abstractmethod
import warnings
from datetime import datetime
import json


class AlertType(Enum):
    """Types of transient alerts"""
    ZTF = "ztf"
    RUBIN = "rubin"
    TESS = "tess"
    FERMI = "fermi"
    SWIFT = "swift"
    GENERIC = "generic"


class AlertPriority(Enum):
    """Alert follow-up priority levels"""
    CRITICAL = 5  # Immediate follow-up required
    HIGH = 4
    MEDIUM = 3
    LOW = 2
    ROUTINE = 1


@dataclass
class Alert:
    """Base alert structure"""
    alert_id: str
    timestamp: datetime
    ra: float  # degrees
    dec: float  # degrees
    magnitude: float
    mag_error: float
    filter_band: str
    alert_type: AlertType
    raw_data: Dict[str, Any] = field(default_factory=dict)
    classification: str = "unknown"
    priority: AlertPriority = AlertPriority.ROUTINE


@dataclass
class LightCurvePoint:
    """Single photometric measurement"""
    mjd: float
    mag: float
    mag_err: float
    band: str
    flux: float = None
    flux_err: float = None
    is_detection: bool = True


@dataclass
class ProcessedAlert:
    """Processed alert with derived properties"""
    alert: Alert
    light_curve: List[LightCurvePoint]
    cross_matches: Dict[str, Any]
    classification: str
    classification_score: float
    priority_score: float
    follow_up_recommended: bool
    notes: List[str] = field(default_factory=list)


# =============================================================================
# ALERT STREAM PROCESSOR
# =============================================================================

class AlertStreamProcessor:
    """
    Base class for processing transient alert streams.

    Handles connection, filtering, and dispatch to handlers.
    """

    def __init__(self, filters: List[Callable] = None):
        """
        Initialize stream processor.

        Args:
            filters: List of filter functions
        """
        self.filters = filters or []
        self.handlers: Dict[AlertType, 'AlertHandler'] = {}
        self.processed_count = 0
        self.passed_count = 0
        self.alert_buffer: List[Alert] = []
        self.max_buffer_size = 10000

    def register_handler(self, alert_type: AlertType, handler: 'AlertHandler'):
        """
        Register a handler for specific alert type.

        Args:
            alert_type: Type of alerts to handle
            handler: Handler instance
        """
        self.handlers[alert_type] = handler

    def add_filter(self, filter_func: Callable[[Alert], bool]):
        """
        Add a filter function.

        Args:
            filter_func: Function returning True for alerts to keep
        """
        self.filters.append(filter_func)

    def process_alert(self, alert_data: Dict[str, Any],
                      alert_type: AlertType) -> Optional[ProcessedAlert]:
        """
        Process a single alert packet.

        Args:
            alert_data: Raw alert data
            alert_type: Type of alert

        Returns:
            ProcessedAlert if passes filters, None otherwise
        """
        self.processed_count += 1

        # Get appropriate handler
        handler = self.handlers.get(alert_type)
        if handler is None:
            handler = self.handlers.get(AlertType.GENERIC)

        if handler is None:
            warnings.warn(f"No handler for alert type: {alert_type}")
            return None

        # Parse alert
        alert = handler.parse_alert(alert_data)

        # Apply filters
        for filt in self.filters:
            if not filt(alert):
                return None

        self.passed_count += 1

        # Process with handler
        processed = handler.process(alert)

        # Buffer for batch operations
        self._add_to_buffer(processed.alert)

        return processed

    def _add_to_buffer(self, alert: Alert):
        """Add alert to buffer, maintaining max size"""
        self.alert_buffer.append(alert)
        if len(self.alert_buffer) > self.max_buffer_size:
            self.alert_buffer = self.alert_buffer[-self.max_buffer_size:]

    def get_statistics(self) -> Dict[str, Any]:
        """Get processing statistics"""
        return {
            'processed': self.processed_count,
            'passed_filters': self.passed_count,
            'pass_rate': self.passed_count / max(self.processed_count, 1),
            'buffer_size': len(self.alert_buffer),
        }


# =============================================================================
# ALERT HANDLERS
# =============================================================================

class AlertHandler(ABC):
    """Base class for alert handlers"""
