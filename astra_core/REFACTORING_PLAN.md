# ASTRA astra_core Refactoring Plan

**Objective**: Remove Vxxx version numbering from astra_core to create a coherent, integrated structure that is professional and not confusing for users.

**Date**: April 2026
**Status**: Planning Phase

---

## 1. Current State Analysis

### 1.1 Versioned Directories (Root Level)

| Directory | Purpose | Files |
|-----------|---------|-------|
| `v4_revolutionary/` | V4.0 revolutionary capabilities | MCE, ASC, CRN, MMOL |
| `v5_discovery/` | V5.0 discovery system | (mostly empty) |
| `v7_autonomous_research/` | V7.0 autonomous research scientist | Question/Hypothesis/Experiment engines |
| `v100/` | V100 simulation & validation system | Core, simulation, validation |
| `core_legacy/` | Legacy core systems | v36-v94 versioned subdirs |

### 1.2 Versioned Capability Files (36 files)

**V50 Series** (Discovery Enhancement):
- v50_abstraction_learning.py
- v50_adversarial_debate.py
- v50_causal_engine.py
- v50_meta_learner.py
- v50_program_synthesis.py
- v50_world_simulator.py

**V60 Series** (Cognitive/Persistent Memory):
- v60_active_inference.py
- v60_active_knowledge.py
- v60_cognitive_agent.py
- v60_cognitive_self_modification.py
- v60_grounded_representations.py
- v60_persistent_memory.py
- v60_predictive_world_models.py

**V70 Series** (Meta-Scientific/Emergent):
- v70_algorithmic_discovery.py
- v70_analogical_transfer.py
- v70_emergent_computation.py
- v70_hypothesis_generator.py
- v70_meta_scientific.py
- v70_predictive_geometry.py
- v70_synthetic_intelligence.py
- v70_temporal_hierarchy.py
- v70_universal_causal.py

**V95-V98 Series** (Advanced Capabilities):
- v95_semantic_grounding.py
- v95_semantic_grounding_demo.py
- v95_integration.py
- v96_discovery_provenance.py
- v97_knowledge_isolation.py
- v98_fci_causal_discovery.py

**V100-V108 Series** (Latest Capabilities):
- v101_temporal_causal.py
- v102_counterfactual_engine.py
- v103_multimodal_evidence.py
- v104_adversarial_discovery.py
- v105_meta_discovery.py
- v106_explainable_causal.py
- v107_discovery_triage.py
- v108_streaming_discovery.py

### 1.3 Versioned Core Legacy Systems (17 versions)

v36, v37, v38, v39, v40, v41, v42, v43, v50, v80, v90, v91, v92, v93, v94

### 1.4 Versioned Test Directories

- `tests/v4/` - V4.0 integration tests
- `tests/v5/` - V5.0 capability tests

### 1.5 Files with Versioned Imports

86 files contain imports from Vxxx modules, including:
- Main __init__.py
- reasoning/__init__.py
- theoretical_discovery/__init__.py
- Various test files
- Core system files

---

## 2. Target Structure

### 2.1 Proposed Directory Organization

```
astra_core/
├── autonomous_research/          # V7.0 (was v7_autonomous_research)
│   ├── scientist.py              # Main autonomous scientist
│   ├── engines/
│   │   ├── question_generator.py
│   │   ├── hypothesis_formulator.py
│   │   ├── experiment_designer.py
│   │   ├── experiment_executor.py
│   │   ├── prediction_engine.py
│   │   ├── analysis_engine.py
│   │   ├── theory_revision.py
│   │   └── publication_engine.py
│   └── architecture/
│       ├── global_coherence.py
│       ├── hierarchical_understanding.py
│       └── continuous_learning.py
│
├── capabilities/                 # Unified capabilities (no version numbers)
│   ├── causal/                   # Causal inference capabilities
│   │   ├── causal_engine.py      # v50_causal_engine
│   │   ├── fci_discovery.py      # v98_fci_causal_discovery
│   │   ├── explainable_causal.py # v106_explainable_causal
│   │   ├── temporal_causal.py    # v101_temporal_causal
│   │   └── universal_causal.py   # v70_universal_causal
│   │
│   ├── discovery/                 # Discovery capabilities
│   │   ├── meta_discovery.py     # v105_meta_discovery
│   │   ├── adversarial_discovery.py # v104_adversarial_discovery
│   │   ├── algorithmic_discovery.py # v70_algorithmic_discovery
│   │   ├── discovery_provenance.py # v96_discovery_provenance
│   │   ├── discovery_triage.py   # v107_discovery_triage
│   │   └── streaming_discovery.py # v108_streaming_discovery
│   │
│   ├── learning/                  # Learning capabilities
│   │   ├── meta_learner.py       # v50_meta_learner
│   │   ├── abstraction_learning.py # v50_abstraction_learning
│   │   ├── adversarial_debate.py  # v50_adversarial_debate
│   │   ├── semantic_grounding.py # v95_semantic_grounding
│   │   └── knowledge_isolation.py # v97_knowledge_isolation
│   │
│   ├── cognitive/                 # Cognitive capabilities
│   │   ├── cognitive_agent.py    # v60_cognitive_agent
│   │   ├── active_inference.py   # v60_active_inference
│   │   ├── cognitive_self_modification.py # v60_cognitive_self_modification
│   │   ├── grounded_representations.py # v60_grounded_representations
│   │   └── predictive_world_models.py # v60_predictive_world_models
│   │
│   ├── memory/                    # Memory capabilities
│   │   └── persistent_memory.py   # v60_persistent_memory
│   │
│   ├── metacognitive/             # Meta-cognitive capabilities
│   │   ├── meta_scientific.py    # v70_meta_scientific
│   │   ├── synthetic_intelligence.py # v70_synthetic_intelligence
│   │   ├── emergent_computation.py # v70_emergent_computation
│   │   ├── hypothesis_generator.py # v70_hypothesis_generator
│   │   ├── temporal_hierarchy.py # v70_temporal_hierarchy
│   │   └── predictive_geometry.py # v70_predictive_geometry
│   │
│   ├── synthesis/                 # Synthesis capabilities
│   │   ├── program_synthesis.py  # v50_program_synthesis
│   │   ├── world_simulator.py    # v50_world_simulator
│   │   ├── analogical_transfer.py # v70_analogical_transfer
│   │   └── counterfactual_engine.py # v102_counterfactual_engine
│   │
│   └── multimodal/                # Multimodal capabilities
│       ├── multimodal_evidence.py # v103_multimodal_evidence
│       └── active_knowledge.py   # v60_active_knowledge
│
├── revolutionary/                # V4.0 capabilities (was v4_revolutionary)
│   ├── meta_context_engine.py    # MCE
│   ├── autocatalytic_compiler.py # ASC
│   ├── cognitive_relativity.py   # CRN
│   └── multi_mind_orchestration.py # MMOL
│
├── simulation/                   # V100 simulation system (was v100)
│   ├── universe_simulator.py
│   ├── validation.py
│   └── competition.py
│
├── integration/                  # Integrated capabilities (from v95)
│   ├── integration.py
│   └── semantic_grounding_demo.py
│
├── core/                         # Main core system (no versioning)
│   ├── unified.py                 # Main system interface
│   └── scientific_discovery.py   # V105 capability
│
├── legacy/                       # Legacy core systems (was core_legacy)
│   └── systems/                  # All versioned systems archived
│       ├── v36_system.py
│       ├── v37_system.py
│       # ... etc
│       └── v94_system.py
│
├── tests/                        # Unified tests
│   ├── test_autonomous_research/ # V7.0 tests
│   ├── test_capabilities/        # Capability tests
│   ├── test_revolutionary/        # V4.0 tests
│   └── test_integration/          # Integration tests
│
└── [all other existing directories]
    ├── domains/                   # (unchanged - already good)
    ├── reasoning/                # (unchanged)
    ├── physics/                  # (unchanged)
    ├── memory/                   # (unchanged)
    ├── metacognitive/            # (unchanged)
    # ... etc
```

### 2.2 File Naming Conventions

**New Naming Convention**:
- Remove Vxxx prefix from all filenames
- Use descriptive names that indicate functionality
- Group related capabilities in subdirectories

**Examples**:
- `v50_causal_engine.py` → `capabilities/causal/causal_engine.py`
- `v70_meta_scientific.py` → `capabilities/metacognitive/meta_scientific.py`
- `v7_autonomous_scientist.py` → `autonomous_research/scientist.py`

---

## 3. Migration Strategy

### 3.1 Phase 1: Preparation (LOW RISK)

1. **Backup current codebase**
   ```bash
   git stash save "Pre-refactoring backup"
   git checkout -b refactor/remove-version-numbers
   ```

2. **Create compatibility layer**
   - Create stub files with old names that import from new locations
   - This allows gradual migration without breaking existing code

3. **Document all changes**
   - Create migration guide for users
   - Document API changes (if any)

### 3.2 Phase 2: Restructure Capabilities (MEDIUM RISK)

1. **Create new directory structure under capabilities/**
   ```bash
   mkdir -p capabilities/{causal,discovery,learning,cognitive,memory,metacognitive,synthesis,multimodal}
   ```

2. **Move and rename files**
   ```bash
   # Example moves
   mv capabilities/v50_causal_engine.py capabilities/causal/causal_engine.py
   mv capabilities/v70_meta_scientific.py capabilities/metacognitive/meta_scientific.py
   # ... etc
   ```

3. **Create __init__.py files for each subdirectory**
   - Export renamed classes/functions
   - Maintain backward compatibility imports

### 3.3 Phase 3: Restructure Major Systems (MEDIUM RISK)

1. **Rename major directories**
   ```bash
   mv v7_autonomous_research autonomous_research
   mv v4_revolutionary revolutionary
   mv v100 simulation
   ```

2. **Update all internal imports**
   - Update imports within moved files
   - Update imports in other files that reference these

3. **Create compatibility stubs in old locations**
   - Import from new locations
   - Deprecation warnings

### 3.4 Phase 4: Handle Legacy Systems (LOW RISK)

1. **Archive core_legacy versions**
   ```bash
   mv core_legacy legacy
   cd legacy
   mkdir systems
   mv v* systems/
   ```

2. **Create README in legacy/** explaining archived systems

### 3.5 Phase 5: Update Main Entry Points (HIGH RISK)

1. **Update astra_core/__init__.py**
   - Remove versioned imports
   - Import from new locations
   - Maintain backward compatibility

2. **Update core/__init__.py**

3. **Update reasoning/__init__.py**

### 3.6 Phase 6: Update Tests (MEDIUM RISK)

1. **Reorganize test directories**
   ```bash
   mv tests/v4 tests/test_revolutionary
   mv tests/v5 tests/test_discovery
   mkdir tests/test_autonomous_research
   ```

2. **Update test imports**

3. **Run all tests to verify functionality**

### 3.7 Phase 7: Clean Up (LOW RISK)

1. **Remove compatibility stubs after verification**
2. **Update documentation**
3. **Create migration guide for contributors**

---

## 4. Risk Assessment

### 4.1 High Risk Areas

**Main Entry Points** (`astra_core/__init__.py`, `core/__init__.py`):
- **Risk**: Breaking all user code that imports from astra_core
- **Mitigation**: Maintain backward compatibility through re-exports

**Dependencies in 86+ files**:
- **Risk**: Breaking internal imports
- **Mitigation**: Automated find-and-replace with verification

**Test Files**:
- **Risk**: Tests failing due to import errors
- **Mitigation**: Update tests before main code, run frequently

### 4.2 Medium Risk Areas

**Capability Reorganization**:
- **Risk**: Breaking imports from specific capabilities
- **Mitigation**: Subdirectory __init__.py files with re-exports

**Major Directory Renames**:
- **Risk**: Breaking imports from these directories
- **Mitigation**: Compatibility stubs with deprecation warnings

### 4.3 Low Risk Areas

**Legacy System Archival**:
- **Risk**: Minimal - these are rarely used
- **Mitigation**: Document archival location

**File Naming Changes**:
- **Risk**: Low - mostly internal
- **Mitigation**: Proper import updates

---

## 5. Testing Strategy

### 5.1 Pre-Refactoring Testing

1. **Run existing test suite**
   ```bash
   python -m pytest astra_core/tests/ -v
   ```

2. **Create baseline test results**
   - Document current test pass rate
   - Identify any pre-existing failures

3. **Test main imports**
   ```python
   import astra_core
   from astra_core import create_stan_system
   from astra_core.v7_autonomous_research import create_v7_scientist
   ```

### 5.2 Incremental Testing During Refactoring

After each phase:
1. Run test suite
2. Test main imports
3. Test specific capabilities being refactored
4. Fix any failures before proceeding

### 5.3 Post-Refactoring Testing

1. **Full test suite**
   ```bash
   python -m pytest astra_core/tests/ -v --tb=short
   ```

2. **Import tests**
   ```python
   # Test all major entry points still work
   import astra_core
   from astra_core import create_stan_system
   from astra_core.autonomous_research import create_scientist  # NEW
   from astra_core.capabilities.causal import causal_engine  # NEW
   ```

3. **Backward compatibility tests**
   ```python
   # Old imports should still work (with deprecation warnings)
   from astra_core.v7_autonomous_research import create_v7_scientist  # OLD
   from astra_core.capabilities.v50_causal_engine import CausalEngine  # OLD
   ```

4. **Integration tests**
   - Test V7.0 autonomous research cycle
   - Test V4.0 revolutionary capabilities
   - Test V100 simulation system

---

## 6. Implementation Timeline

### Week 1: Planning & Preparation
- Day 1-2: Detailed planning (this document)
- Day 3-4: Backup and test baseline
- Day 5: Create compatibility layer framework

### Week 2: Capabilities Refactoring
- Day 1-2: Phase 2.1-2.2 (Create structure, move files)
- Day 3-4: Phase 2.3 (Update imports, create __init__.py)
- Day 5: Testing and fixes

### Week 3: Major Systems Refactoring
- Day 1-2: Phase 3.1-3.2 (Rename directories, update imports)
- Day 3-4: Phase 3.3 (Compatibility stubs)
- Day 5: Testing and fixes

### Week 4: Legacy & Entry Points
- Day 1-2: Phase 4 (Archive legacy)
- Day 3-4: Phase 5 (Update entry points)
- Day 5: Testing and fixes

### Week 5: Tests & Cleanup
- Day 1-2: Phase 6 (Reorganize tests)
- Day 3-4: Phase 7 (Cleanup, remove compatibility stubs)
- Day 5: Final testing and documentation

---

## 7. Rollback Plan

If refactoring encounters critical issues:

1. **Immediate Rollback**:
   ```bash
   git checkout main
   git branch -D refactor/remove-version-numbers
   ```

2. **Partial Rollback**:
   - Revert specific commits
   - Keep successful changes

3. **Fallback Strategy**:
   - Keep versioned names as aliases
   - Document known issues
   - Plan smaller, incremental refactoring

---

## 8. Success Criteria

Refactoring will be considered successful when:

1. ✅ No Vxxx version numbers remain in directory or file names
2. ✅ All tests pass with ≥95% success rate
3. ✅ Backward compatibility maintained through deprecation period
4. ✅ Documentation updated with new structure
5. ✅ Code imports work cleanly without version confusion
6. ✅ Git history is clean and well-documented

---

## 9. Communication Plan

### 9.1 For Users

- Migration guide with code examples
- Deprecation notices in old imports
- New API documentation

### 9.2 For Contributors

- New contribution guidelines
- Directory structure documentation
- Code organization principles

### 9.3 For Developers

- Technical implementation details
- Import migration guide
- Testing guidelines

---

## 10. Next Steps

1. **Review and approve this plan** with stakeholders
2. **Create detailed task list** from this plan
3. **Begin Phase 1: Preparation**
4. **Execute refactoring incrementally**
5. **Monitor and test at each step**
6. **Document lessons learned**

---

**Document Version**: 1.0
**Last Updated**: 2026-04-03
**Status**: Ready for Review and Approval
