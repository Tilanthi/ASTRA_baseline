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
V7.0 Autonomous Research Scientist

Full autonomous research cycle: question generation -> hypothesis formulation ->
experiment design -> execution -> prediction -> analysis -> theory revision ->
publication.

The canonical dataclasses and enums live in `.types` (that is the module
`v7_autonomous_scientist` itself imports from); the engines live in `.engines`.
Both surfaces are re-exported here so that `astra_core/__init__.py` can expose
them at package level.
"""

from .v7_autonomous_scientist import (
    V7AutonomousScientist,
    create_v7_scientist,
)

from .types import (
    # Cycle / question
    ResearchCycle,
    ResearchQuestion,
    QuestionType,
    QuestionImportance,
    # Hypothesis
    Hypothesis,
    HypothesisType,
    HypothesisStatus,
    # Experiment
    Experiment,
    ExperimentType,
    DesignParameters,
    DataSource,
    ExecutionResult,
    # Prediction / analysis
    PredictionType,
    PredictionConfidence,
    AnalysisType,
    CausalInferenceResult,
    # Theory / publication
    RevisionType,
    TheoryStatus,
    PaperStructure,
    FigureType,
    ResearchResult,
    Publication,
)

from .engines import (
    QuestionGenerator,
    HypothesisFormulator,
    ExperimentDesigner,
    ExperimentExecutor,
    PredictionEngine,
    AnalysisEngine,
    TheoryRevisionEngine,
    PublicationEngine,
)

__all__ = [
    # Main system
    'V7AutonomousScientist',
    'create_v7_scientist',
    # Types
    'ResearchCycle',
    'ResearchQuestion',
    'QuestionType',
    'QuestionImportance',
    'Hypothesis',
    'HypothesisType',
    'HypothesisStatus',
    'Experiment',
    'ExperimentType',
    'DesignParameters',
    'DataSource',
    'ExecutionResult',
    'PredictionType',
    'PredictionConfidence',
    'AnalysisType',
    'CausalInferenceResult',
    'RevisionType',
    'TheoryStatus',
    'PaperStructure',
    'FigureType',
    'ResearchResult',
    'Publication',
    # Engines
    'QuestionGenerator',
    'HypothesisFormulator',
    'ExperimentDesigner',
    'ExperimentExecutor',
    'PredictionEngine',
    'AnalysisEngine',
    'TheoryRevisionEngine',
    'PublicationEngine',
]
