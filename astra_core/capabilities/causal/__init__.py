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
Causal Capabilities

This module contains causal inference and causal discovery capabilities.
"""

# Re-export for backward compatibility
import warnings

# Import from new locations
try:
    from .causal_engine import *
    from .fci_discovery import *
    from .explainable_causal import *
    from .temporal_causal import *
    from .universal_causal import *
except ImportError as e:
    warnings.warn(f"Some causal capabilities could not be imported: {e}")

__all__ = [
    # Causal inference
    'CausalEngine',
    'FCIDiscovery',
    'ExplainableCausal',
    'TemporalCausal',
    'UniversalCausal',
]
