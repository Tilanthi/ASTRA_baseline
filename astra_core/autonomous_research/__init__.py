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
