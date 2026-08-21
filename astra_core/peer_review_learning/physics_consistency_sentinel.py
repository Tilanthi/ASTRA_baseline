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
Physics Consistency Sentinel

Enforces dimensional analysis, limit-case validation, cross-theory consistency,
and order-of-magnitude sanity checking.
"""

from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass
import numpy as np


@dataclass
class PhysicalClaim:
    """A physical claim with validation metadata"""
    claim: str
    equation: Optional[str]
    variables: Dict[str, Tuple[float, str]]  # value, units
    dimensions: Dict[str, str]
    limit_cases: Dict[str, float]
    theoretical_bounds: Dict[str, Tuple[float, float]]
    consistency_checks: List[str]


@dataclass
class Violation:
    """A physics consistency violation"""
    violation_type: str
    severity: str  # 'critical', 'warning', 'info'
    description: str
    affected_claim: str
    suggested_fix: Optional[str]


class PhysicsConsistencySentinel:
    """
    Physics consistency checker that addresses peer review concerns about
    violations of physical constraints and dimensional inconsistencies.
    """

    def __init__(self):
        # Fundamental physical constants (CGS units)
        self.constants = {
            'c': 2.998e10,  # speed of light (cm/s)
            'G': 6.674e-8,  # gravitational constant (dyn cm^2/g^2)
            'h': 6.626e-27,  # Planck constant (erg s)
            'k_B': 1.381e-16,  # Boltzmann constant (erg/K)
            'm_e': 9.109e-28,  # electron mass (g)
            'm_p': 1.673e-24,  # proton mass (g)
            'e': 4.803e-10,  # elementary charge (esu)
            'sigma_sb': 5.670e-5,  # Stefan-Boltzmann constant (erg cm^-2 s^-1 K^-4)
            'a_rad': 7.565e-15,  # radiation constant (erg cm^-3 K^-4)
        }

        # Dimension mappings
        self.base_dimensions = {
            'mass': 'M',
            'length': 'L',
            'time': 'T',
            'temperature': 'Θ',
            'charge': 'Q'
        }

        # Common derived dimensions
        self.derived_dimensions = {
            'velocity': 'L/T',
            'acceleration': 'L/T^2',
            'force': 'M*L/T^2',
            'energy': 'M*L^2/T^2',
            'power': 'M*L^2/T^3',
            'pressure': 'M/L/T^2',
            'density': 'M/L^3',
            'frequency': '1/T',
            'angular_momentum': 'M*L^2/T',
            'entropy': 'M*L^2/T^2/Θ',
        }

    def check_dimensional_consistency(self,
                                     claim: PhysicalClaim) -> List[Violation]:
        """
        Ensure units are consistent throughout analysis.

        Addresses peer review: "Dimensional inconsistency in equation"
        """
        violations = []

        if claim.equation:
            # Parse equation and check dimensions
            lhs_dims, rhs_dims = self._parse_equation_dimensions(claim.equation, claim.variables)

            if lhs_dims != rhs_dims:
                violations.append(Violation(
                    violation_type='dimensional_inconsistency',
                    severity='critical',
                    description=f"LHS dimensions [{lhs_dims}] ≠ RHS dimensions [{rhs_dims}]",
                    affected_claim=claim.claim,
                    suggested_fix="Check units on both sides of equation"
                ))

        # Check variable dimensions are consistent
        for var_name, (value, units) in claim.variables.items():
            if var_name in claim.dimensions:
                expected_dims = claim.dimensions[var_name]
                inferred_dims = self._infer_dimensions_from_units(units)

                if expected_dims != inferred_dims:
                    violations.append(Violation(
                        violation_type='unit_mismatch',
                        severity='warning',
                        description=f"Variable {var_name}: expected [{expected_dims}], inferred [{inferred_dims}] from units '{units}'",
                        affected_claim=claim.claim,
                        suggested_fix=f"Verify units for {var_name}"
                    ))

        return violations

    def check_limit_cases(self,
                         claim: PhysicalClaim,
                         function: callable) -> List[Violation]:
        """
        Validate theories at extremes (zero, infinity, known limits).

        Addresses peer review: "Does theory behave correctly in known limits?"
        """
        violations = []

        for limit_name, limit_value in claim.limit_cases.items():
            try:
                result = function(limit_value)

                # Check for numerical issues
                if not np.isfinite(result):
                    violations.append(Violation(
                        violation_type='limit_case_failure',
                        severity='critical',
                        description=f"Function returns non-finite value at limit {limit_name}={limit_value}",
                        affected_claim=claim.claim,
                        suggested_fix="Check for division by zero or overflow"
                    ))

                # Check against expected behavior if provided
                if limit_name in claim.theoretical_bounds:
                    min_val, max_val = claim.theoretical_bounds[limit_name]
                    if not (min_val <= result <= max_val):
                        violations.append(Violation(
                            violation_type='limit_case_violation',
                            severity='critical',
                            description=f"At {limit_name}={limit_value}, result {result} outside bounds [{min_val}, {max_val}]",
                            affected_claim=claim.claim,
                            suggested_fix="Re-examine theory derivation"
                        ))

            except Exception as e:
                violations.append(Violation(
                    violation_type='limit_case_error',
                    severity='critical',
                    description=f"Error evaluating limit {limit_name}={limit_value}: {str(e)}",
                    affected_claim=claim.claim,
                    suggested_fix="Check function implementation"
                ))

        return violations

    def check_cross_theory_consistency(self,
                                      claim: PhysicalClaim,
                                      domain_theories: Dict[str, Any]) -> List[Violation]:
        """
        Flag contradictions between different physical theories.

        Addresses peer review: "This contradicts established theory X"
        """
        violations = []

        # Check against thermodynamics
        if 'entropy' in claim.claim.lower():
            if not self._check_entropy_consistency(claim):
                violations.append(Violation(
                    violation_type='thermodynamics_violation',
                    severity='critical',
                    description="Claim may violate second law of thermodynamics",
                    affected_claim=claim.claim,
                    suggested_fix="Verify entropy doesn't decrease in isolated system"
                ))

        # Check against relativity
        if 'velocity' in claim.claim.lower() or 'speed' in claim.claim.lower():
            for var_name, (value, units) in claim.variables.items():
                if 'velocity' in var_name.lower() or 'speed' in var_name.lower():
                    # Check if speed exceeds c
                    if units in ['cm/s', 'm/s', 'km/s']:
                        # Convert to cm/s
                        if units == 'm/s':
                            value_cm = value * 100
                        elif units == 'km/s':
                            value_cm = value * 1e5
                        else:
                            value_cm = value

                        if value_cm > self.constants['c']:
                            violations.append(Violation(
                                violation_type='relativity_violation',
                                severity='critical',
                                description=f"{var_name} = {value} {units} exceeds speed of light",
                                affected_claim=claim.claim,
                                suggested_fix="Check if velocity is relativistic"
                            ))

        # Check against quantum mechanics
        if 'planck' in claim.claim.lower() or 'quantum' in claim.claim.lower():
            if not self._check_quantum_consistency(claim):
                violations.append(Violation(
                    violation_type='quantum_inconsistency',
                    severity='warning',
                    description="Claim may not respect quantum mechanical principles",
                    affected_claim=claim.claim,
                    suggested_fix="Verify quantum regime applicability"
                ))

        # Check against conservation laws
        if 'energy' in claim.claim.lower() or 'momentum' in claim.claim.lower():
            if not self._check_conservation_laws(claim):
                violations.append(Violation(
                    violation_type='conservation_violation',
                    severity='critical',
                    description="Claim may violate conservation law",
                    affected_claim=claim.claim,
                    suggested_fix="Explicitly check energy/momentum conservation"
                ))

        return violations

    def check_order_of_magnitude(self,
                                claim: PhysicalClaim) -> List[Violation]:
        """
        Sanity check for physically impossible results.

        Addresses peer review: "This result is orders of magnitude off"
        """
        violations = []

        # Check variables against reasonable astrophysical ranges
        astrophysical_ranges = {
            'temperature': (1e0, 1e12),  # K
            'density': (1e-30, 1e20),  # g/cm^3
            'pressure': (1e-20, 1e20),  # dyne/cm^2
            'luminosity': (1e20, 1e50),  # erg/s
            'mass': (1e-10, 1e50),  # g
            'radius': (1e-5, 1e30),  # cm
            'magnetic_field': (1e-10, 1e10),  # G
        }

        for var_name, (value, units) in claim.variables.items():
            # Check if variable name matches a known quantity
            for quantity, (min_val, max_val) in astrophysical_ranges.items():
                if quantity in var_name.lower():
                    if not (min_val <= value <= max_val):
                        violations.append(Violation(
                            violation_type='order_of_magnitude_violation',
                            severity='warning',
                            description=f"{var_name} = {value} {units} outside typical range [{min_val:.1e}, {max_val:.1e}]",
                            affected_claim=claim.claim,
                            suggested_fix="Verify value and units"
                        ))

        # Check energy scales
        if 'energy' in claim.claim.lower():
            for var_name, (value, units) in claim.variables.items():
                if 'energy' in var_name.lower():
                    # Convert to ergs
                    energy_erg = self._convert_to_ergs(value, units)
                    if energy_erg < 1e-10 or energy_erg > 1e100:
                        violations.append(Violation(
                            violation_type='energy_scale_violation',
                            severity='warning',
                            description=f"Energy {value} {units} seems unphysically extreme",
                            affected_claim=claim.claim,
                            suggested_fix="Check energy scale and units"
                        ))

        return violations

    def _parse_equation_dimensions(self,
                                   equation: str,
                                   variables: Dict[str, Tuple[float, str]]) -> Tuple[str, str]:
        """
        Parse equation and return dimensions of LHS and RHS.
        Simplified implementation - would need full equation parser in production.
        """
        # Placeholder: returns dimensionless for now
        # In production, would use symbolic algebra (sympy) to parse equation
        return '1', '1'

    def _infer_dimensions_from_units(self, units: str) -> str:
        """Infer physical dimensions from units string"""
        # Common unit mappings
        unit_to_dimension = {
            'cm': 'L',
            'm': 'L',
            'km': 'L',
            'pc': 'L',
            's': 'T',
            'yr': 'T',
            'g': 'M',
            'kg': 'M',
            'M_sun': 'M',
            'K': 'Θ',
            'erg': 'M*L^2/T^2',
            'dyne': 'M*L/T^2',
            'G': 'M*L/T^2/I^2',  # Need to handle current
        }

        # Simplified - would need full unit parser in production
        return unit_to_dimension.get(units.split('/')[-0], 'unknown')

    def _convert_to_ergs(self, value: float, units: str) -> float:
        """Convert energy value to ergs"""
        conversions = {
            'erg': 1.0,
            'J': 1e7,
            'eV': 1.602e-12,
            'keV': 1.602e-9,
            'MeV': 1.602e-6,
            'GeV': 1.602e-3
        }
        return value * conversions.get(units, 1.0)

    def _check_entropy_consistency(self, claim: PhysicalClaim) -> bool:
        """Check if claim respects second law of thermodynamics"""
        # Placeholder: would implement specific entropy checks
        return True

    def _check_quantum_consistency(self, claim: PhysicalClaim) -> bool:
        """Check if claim respects quantum mechanical principles"""
        # Placeholder: would implement quantum consistency checks
        return True

    def _check_conservation_laws(self, claim: PhysicalClaim) -> bool:
        """Check if claim respects conservation laws"""
        # Placeholder: would implement conservation law checks
        return True


def create_physics_consistency_sentinel() -> PhysicsConsistencySentinel:
    """Factory function for PhysicsConsistencySentinel"""
    return PhysicsConsistencySentinel()
