"""
Statistical Defense Framework

Implements power analysis, robustness testing, effect size validation,
and multiple comparisons tracking to address statistical rigor concerns.
"""

from typing import List, Dict, Any, Tuple, Optional, Union
from dataclasses import dataclass
import numpy as np
from scipy import stats
from scipy.stats import power, mannwhitneyu, ks_2samp, bootstrap


@dataclass
class StatisticalClaim:
    """A statistical claim with defense metadata"""
    claim: str
    test_statistic: float
    p_value: float
    effect_size: float
    confidence_interval: Tuple[float, float]
    sample_size: int
    power: float
    robustness_checks: Dict[str, Any]
    multiple_testing_correction: Optional[str]
    assumptions: List[str]


@dataclass
class PowerAnalysis:
    """Statistical power analysis results"""
    required_sample_size: int
    achieved_power: float
    effect_size: float
    alpha: float
    beta: float
    recommendations: List[str]


@dataclass
class RobustnessResult:
    """Result of robustness/sensitivity analysis"""
    test_type: str
    original_result: float
    perturbed_results: List[float]
    sensitivity_score: float
    outliers_influential: bool
    recommended_analysis: str


class StatisticalDefenseFramework:
    """
    Statistical rigor framework that addresses peer review concerns about
    sample size, effect size, robustness, and multiple testing.
    """

    def __init__(self):
        self.effect_size_measures = ['cohens_d', 'r', 'odds_ratio', 'relative_risk']
        self.robustness_tests = [
            'outlier_removal',
            'alternative_test',
            'bootstrap',
            'jackknife',
            'parametric_sensitivity'
        ]
        self.multiple_testing_methods = [
            'bonferroni',
            'holm_bonferroni',
            'benjamini_hochberg',
            'benjamini_yekutieli'
        ]

    def calculate_required_sample_size(self,
                                      effect_size: float,
                                      alpha: float = 0.05,
                                      power: float = 0.8,
                                      test_type: str = 'two_sample') -> PowerAnalysis:
        """
        Calculate required sample size before conducting test.

        Addresses peer review: "Is your sample size adequate?"
        """
        # Calculate sample size for given effect, alpha, power
        if test_type == 'two_sample':
            # Two-sample t-test
            required_n = power.tt_ind_solve_power(
                effect_size=effect_size,
                alpha=alpha,
                power=power
            )
            # Per group
            required_n = int(np.ceil(required_n * 2))
        else:
            # Default approximation
            required_n = int(np.ceil(16 / (effect_size ** 2)))

        # Calculate achieved power with current sample
        achieved_power = power.tt_ind_solve_power(
            effect_size=effect_size,
            nobs=required_n,
            alpha=alpha
        )

        # Generate recommendations
        recommendations = []
        if required_n > 1000:
            recommendations.append("Consider alternative designs with smaller required sample")
        if effect_size < 0.2:
            recommendations.append("Effect size is small - consider if result is practically significant")
        if achieved_power < 0.8:
            recommendations.append(f"Current sample gives {achieved_power:.2f} power - below recommended 0.8")

        return PowerAnalysis(
            required_sample_size=required_n,
            achieved_power=achieved_power,
            effect_size=effect_size,
            alpha=alpha,
            beta=1.0 - achieved_power,
            recommendations=recommendations
        )

    def test_robustness(self,
                       data: np.ndarray,
                       test_result: float,
                       test_function: callable,
                       n_bootstrap: int = 1000) -> List[RobustnessResult]:
        """
        Test sensitivity to outliers, methods, and assumptions.

        Addresses peer review: "How robust are your results to..."
        """
        results = []

        # Test 1: Outlier removal
        outlier_result = self._test_outlier_sensitivity(data, test_function)
        results.append(outlier_result)

        # Test 2: Bootstrap resampling
        bootstrap_result = self._test_bootstrap_robustness(
            data, test_function, n_bootstrap
        )
        results.append(bootstrap_result)

        # Test 3: Alternative test methods
        alt_result = self._test_alternative_methods(data, test_result)
        results.append(alt_result)

        # Test 4: Parameter sensitivity
        param_result = self._test_parameter_sensitivity(data, test_function)
        results.append(param_result)

        return results

    def validate_effect_size(self,
                           effect_size: float,
                           domain: str,
                           context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Distinguish statistical from practical significance.

        Addresses peer review: "Is this effect practically meaningful?"
        """
        validation = {}

        # Categorize effect size (Cohen's conventions)
        if domain == 'astrophysics':
            # Astrophysics often deals with large effects
            if abs(effect_size) > 1.0:
                magnitude = "very_large"
                practical = True
            elif abs(effect_size) > 0.5:
                magnitude = "large"
                practical = True
            elif abs(effect_size) > 0.2:
                magnitude = "moderate"
                practical = True
            elif abs(effect_size) > 0.1:
                magnitude = "small"
                practical = None  # Context-dependent
            else:
                magnitude = "very_small"
                practical = False
        else:
            # Standard Cohen's conventions
            if abs(effect_size) > 0.8:
                magnitude = "large"
                practical = True
            elif abs(effect_size) > 0.5:
                magnitude = "medium"
                practical = True
            elif abs(effect_size) > 0.2:
                magnitude = "small"
                practical = None
            else:
                magnitude = "negligible"
                practical = False

        validation['magnitude'] = magnitude
        validation['practically_significant'] = practical

        # Compare to domain benchmarks
        if 'benchmark_effects' in context:
            benchmarks = context['benchmark_effects']
            percentile = self._compute_effect_percentile(effect_size, benchmarks)
            validation['percentile_vs_benchmarks'] = percentile
            validation['above_median'] = percentile > 50

        # Confidence interval for effect size
        if 'effect_ci' in context:
            ci_low, ci_high = context['effect_ci']
            validation['effect_ci'] = (ci_low, ci_high)
            validation['ci_excludes_zero'] = (ci_low * ci_high > 0)

        return validation

    def track_multiple_comparisons(self,
                                  p_values: List[float],
                                  claims: List[str],
                                  method: str = 'benjamini_hochberg') -> Dict[str, Any]:
        """
        Track all implicit tests and correct for multiple comparisons.

        Addresses peer review: "Have you corrected for multiple testing?"
        """
        n_tests = len(p_values)

        # Apply correction
        if method == 'bonferroni':
            rejected, adjusted_p, _, _ = stats.multipletests(
                p_values, alpha=0.05, method='bonferroni'
            )
        elif method == 'holm_bonferroni':
            rejected, adjusted_p, _, _ = stats.multipletests(
                p_values, alpha=0.05, method='holm'
            )
        elif method == 'benjamini_hochberg':
            rejected, adjusted_p, _, _ = stats.multipletests(
                p_values, alpha=0.05, method='fdr_bh'
            )
        elif method == 'benjamini_yekutieli':
            rejected, adjusted_p, _, _ = stats.multipletests(
                p_values, alpha=0.05, method='fdr_by'
            )
        else:
            rejected, adjusted_p = [], []

        # Compile results
        results = {
            'n_tests': n_tests,
            'correction_method': method,
            'n_significant_uncorrected': sum(p < 0.05 for p in p_values),
            'n_significant_corrected': sum(rejected),
            'adjusted_p_values': adjusted_p,
            'claims': claims
        }

        # Flag problematic tests
        results['lost_significance'] = [
            claim for claim, p, adj, orig in zip(claims, p_values, adjusted_p, rejected)
            if p < 0.05 and not orig
        ]

        # Recommendation
        if results['n_significant_uncorrected'] > results['n_significant_corrected']:
            results['recommendation'] = (
                f"{method} correction reduced significant findings from "
                f"{results['n_significant_uncorrected']} to {results['n_significant_corrected']}"
            )
        else:
            results['recommendation'] = "Results robust to multiple testing correction"

        return results

    def check_assumptions(self,
                         test_type: str,
                         data: Union[np.ndarray, Tuple[np.ndarray, ...]],
                         alpha: float = 0.05) -> Dict[str, Any]:
        """
        Check statistical assumptions and warn of violations.

        Addresses peer review: "Does your data meet test assumptions?"
        """
        assumptions_checked = {}

        if test_type in ['t_test', 'anova']:
            # Normality test
            if isinstance(data, np.ndarray):
                _, normality_p = stats.shapiro(data[:5000])  # Shapiro limited to 5000
            else:
                # Test each group
                normality_p = [stats.shapiro(group[:5000])[1] for group in data]

            assumptions_checked['normality'] = {
                'assumed': True,
                'p_value': normality_p if isinstance(normality_p, float) else normality_p,
                'met': normality_p > alpha if isinstance(normality_p, float)
                else all(p > alpha for p in normality_p),
                'warning': normality_p < alpha if isinstance(normality_p, float)
                else any(p < alpha for p in normality_p)
            }

            # Homogeneity of variance
            if isinstance(data, tuple) and len(data) == 2:
                _, levene_p = stats.levene(data[0], data[1])
                assumptions_checked['homogeneity_variance'] = {
                    'assumed': True,
                    'p_value': levene_p,
                    'met': levene_p > alpha,
                    'warning': levene_p < alpha
                }

        elif test_type == 'mann_whitney':
            assumptions_checked['ordinal'] = {
                'assumed': False,
                'met': True,
                'note': 'No distribution assumptions required'
            }

        # Independence check
        assumptions_checked['independence'] = {
            'assumed': True,
            'met': None,  # Cannot be tested from data alone
            'note': 'Must be ensured by study design'
        }

        return assumptions_checked

    def _test_outlier_sensitivity(self,
                                  data: np.ndarray,
                                  test_function: callable) -> RobustnessResult:
        """Test if results change with outlier removal"""
        # Identify outliers using IQR method
        Q1, Q3 = np.percentile(data, [25, 75])
        IQR = Q3 - Q1
        outlier_mask = (data < Q1 - 1.5*IQR) | (data > Q3 + 1.5*IQR)
        n_outliers = np.sum(outlier_mask)

        # Run test with and without outliers
        original_result = test_function(data)
        cleaned_data = data[~outlier_mask]
        perturbed_result = test_function(cleaned_data) if len(cleaned_data) > 10 else original_result

        # Calculate sensitivity
        relative_change = abs(perturbed_result - original_result) / (abs(original_result) + 1e-10)

        return RobustnessResult(
            test_type='outlier_removal',
            original_result=original_result,
            perturbed_results=[perturbed_result],
            sensitivity_score=relative_change,
            outliers_influential=relative_change > 0.1,
            recommended_analysis='robust_method' if relative_change > 0.1 else 'standard'
        )

    def _test_bootstrap_robustness(self,
                                   data: np.ndarray,
                                   test_function: callable,
                                   n_bootstrap: int) -> RobustnessResult:
        """Test stability using bootstrap resampling"""
        original_result = test_function(data)
        bootstrapped_results = []

        for _ in range(n_bootstrap):
            resample = np.random.choice(data, size=len(data), replace=True)
            bootstrapped_results.append(test_function(resample))

        # Calculate variability
        bootstrapped_results = np.array(bootstrapped_results)
        variability = np.std(bootstrapped_results) / (abs(original_result) + 1e-10)

        return RobustnessResult(
            test_type='bootstrap',
            original_result=original_result,
            perturbed_results=bootstrapped_results.tolist(),
            sensitivity_score=variability,
            outliers_influential=variability > 0.2,
            recommended_analysis='report_bootstrap_ci'
        )

    def _test_alternative_methods(self,
                                  data: np.ndarray,
                                  original_result: float) -> RobustnessResult:
        """Test using alternative statistical methods"""
        # If data is two groups, compare parametric vs non-parametric
        if isinstance(data, tuple) and len(data) == 2:
            group1, group2 = data

            # Parametric (assuming original)
            parametric_result = original_result

            # Non-parametric alternative
            stat, p_value = mannwhitneyu(group1, group2)
            nonparametric_result = p_value

            # Alternative non-parametric
            stat_ks, p_ks = ks_2samp(group1, group2)
            ks_result = p_ks

            return RobustnessResult(
                test_type='alternative_methods',
                original_result=parametric_result,
                perturbed_results=[nonparametric_result, ks_result],
                sensitivity_score=0.0,  # Compute based on consistency
                outliers_influential=False,
                recommended_analysis='consistent' if abs(nonparametric_result - parametric_result) < 0.1
                else 'report_both'
            )

        return RobustnessResult(
            test_type='alternative_methods',
            original_result=original_result,
            perturbed_results=[],
            sensitivity_score=0.0,
            outliers_influential=False,
            recommended_analysis='na'
        )

    def _test_parameter_sensitivity(self,
                                    data: np.ndarray,
                                    test_function: callable) -> RobustnessResult:
        """Test sensitivity to analysis parameters"""
        results = []

        # Vary bin sizes if relevant
        # Vary smoothing parameters
        # Vary threshold values

        # Placeholder implementation
        return RobustnessResult(
            test_type='parameter_sensitivity',
            original_result=0.0,
            perturbed_results=[],
            sensitivity_score=0.0,
            outliers_influential=False,
            recommended_analysis='na'
        )

    def _compute_effect_percentile(self,
                                   effect_size: float,
                                   benchmarks: List[float]) -> float:
        """Compute percentile rank vs benchmark effects"""
        percentile = stats.percentileofscore(benchmarks, effect_size)
        return percentile


def create_statistical_defense_framework() -> StatisticalDefenseFramework:
    """Factory function for StatisticalDefenseFramework"""
    return StatisticalDefenseFramework()
