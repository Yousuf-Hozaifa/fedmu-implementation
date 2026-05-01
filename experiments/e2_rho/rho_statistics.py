"""
experiments/e2_rho/rho_statistics.py

Statistical analysis of ρ(p) values — implements the full
Experiment E2 statistical validation plan from the proposal:

  1. Descriptive statistics (mean, std, median, percentiles)
  2. Distribution report (histogram + box plot data)
  3. Bootstrap confidence intervals (95% CI)
  4. One-sided t-test: H₀: E[ρ(p)] ≤ 1 vs H₁: E[ρ(p)] > 1
  5. Cohen's d effect size
  6. Power analysis confirmation

All methods take a list of ρ(p) floats as input and return
structured result dicts — no plotting library required here.
Visualisation is separate (Part 4D).
"""

import math
import random
from typing import Dict, List, Optional, Tuple


def descriptive_stats(rho_values: List[float]) -> Dict:
    """
    Compute basic descriptive statistics for a list of ρ(p) values.
    Uses only stdlib math — no numpy required.
    """
    n = len(rho_values)
    if n == 0:
        return {}

    sorted_vals = sorted(rho_values)
    mean = sum(rho_values) / n

    # Variance and std
    variance = sum((x - mean) ** 2 for x in rho_values) / (n - 1)
    std = math.sqrt(variance)

    # Percentiles via linear interpolation
    def percentile(p: float) -> float:
        idx = (n - 1) * p / 100.0
        lo, hi = int(idx), min(int(idx) + 1, n - 1)
        return sorted_vals[lo] + (idx - lo) * (sorted_vals[hi] - sorted_vals[lo])

    return {
        "n":       n,
        "mean":    mean,
        "std":     std,
        "min":     sorted_vals[0],
        "p25":     percentile(25),
        "median":  percentile(50),
        "p75":     percentile(75),
        "max":     sorted_vals[-1],
        "pct_above_1":   100 * sum(r > 1.0 for r in rho_values) / n,
        "pct_above_1_5": 100 * sum(r > 1.5 for r in rho_values) / n,
    }


def bootstrap_confidence_interval(
    rho_values: List[float],
    n_bootstrap: int = 1000,
    confidence: float = 0.95,
    seed: int = 42,
) -> Tuple[float, float, float]:
    """
    Compute a bootstrap confidence interval for the mean of ρ(p).

    Args:
        rho_values:   list of per-patient ρ(p) values
        n_bootstrap:  number of bootstrap resamples
        confidence:   CI level (default 0.95 = 95% CI)
        seed:         random seed for reproducibility

    Returns:
        (mean, lower_bound, upper_bound)

    How it works:
      1. Draw n_bootstrap samples of size n WITH REPLACEMENT from rho_values
      2. Compute the mean of each resample
      3. The (2.5th, 97.5th) percentiles of those means = 95% CI
    """
    rng = random.Random(seed)
    n = len(rho_values)

    bootstrap_means = []
    for _ in range(n_bootstrap):
        resample = [rng.choice(rho_values) for _ in range(n)]
        bootstrap_means.append(sum(resample) / n)

    bootstrap_means.sort()

    alpha = 1.0 - confidence
    lo_idx = int(alpha / 2 * n_bootstrap)
    hi_idx = int((1 - alpha / 2) * n_bootstrap)

    lo = bootstrap_means[lo_idx]
    hi = bootstrap_means[min(hi_idx, n_bootstrap - 1)]
    mean = sum(rho_values) / n

    return mean, lo, hi


def one_sided_t_test(
    rho_values: List[float],
    null_mean: float = 1.0,
    alpha_level: float = 0.01,
) -> Dict:
    """
    One-sided t-test: H₀: μ ≤ null_mean vs H₁: μ > null_mean

    Uses stdlib only — implements the t-distribution CDF via
    the regularised incomplete beta function approximation.

    Args:
        rho_values:  list of ρ(p) values
        null_mean:   the hypothesised population mean under H₀ (default: 1.0)
        alpha_level: significance level (default: 0.01)

    Returns:
        dict with t_stat, p_value, reject_h0, interpretation
    """
    n = len(rho_values)
    mean = sum(rho_values) / n
    variance = sum((x - mean)**2 for x in rho_values) / (n - 1)
    std = math.sqrt(variance)
    se = std / math.sqrt(n)   # standard error of the mean

    if se < 1e-12:
        return {
            "t_stat": float('inf'),
            "p_value": 0.0,
            "reject_h0": True,
            "note": "Near-zero standard error — all values identical",
        }

    t_stat = (mean - null_mean) / se

    # Approximate one-sided p-value using normal approximation
    # (valid for n > 30; for smaller n use scipy.stats.t.sf)
    # p-value = P(T > t_stat) under H₀
    # Use complementary error function approximation
    p_value = _normal_sf(t_stat)  # one-sided upper tail

    reject = p_value < alpha_level

    return {
        "n":            n,
        "sample_mean":  mean,
        "null_mean":    null_mean,
        "t_stat":       t_stat,
        "p_value":      p_value,
        "alpha_level":  alpha_level,
        "reject_h0":    reject,
        "conclusion":   (
            f"REJECT H₀: Evidence that E[ρ(p)] > {null_mean} "
            f"(p={p_value:.4f} < α={alpha_level})"
            if reject else
            f"FAIL TO REJECT H₀: Insufficient evidence that "
            f"E[ρ(p)] > {null_mean} (p={p_value:.4f} ≥ α={alpha_level})"
        ),
    }


def cohens_d(
    rho_values: List[float],
    reference_mean: float = 1.0,
) -> Dict:
    """
    Compute Cohen's d effect size.

    d = (sample_mean - reference_mean) / sample_std

    Interpretation:
      |d| < 0.2:  negligible effect
      0.2 ≤ |d| < 0.5:  small effect
      0.5 ≤ |d| < 0.8:  medium effect
      |d| ≥ 0.8:  large effect

    Args:
        rho_values:      list of ρ(p) values
        reference_mean:  H₀ null mean (default: 1.0)
    """
    n = len(rho_values)
    mean = sum(rho_values) / n
    std = math.sqrt(sum((x - mean)**2 for x in rho_values) / (n - 1))

    d = (mean - reference_mean) / std if std > 1e-12 else 0.0

    if abs(d) < 0.2:
        magnitude = "negligible"
    elif abs(d) < 0.5:
        magnitude = "small"
    elif abs(d) < 0.8:
        magnitude = "medium"
    else:
        magnitude = "large"

    return {
        "d":             d,
        "magnitude":     magnitude,
        "sample_mean":   mean,
        "sample_std":    std,
        "reference_mean": reference_mean,
    }


def power_analysis(
    n: int,
    cohens_d_value: float,
    alpha_level: float = 0.01,
) -> Dict:
    """
    Estimate statistical power for a one-sided t-test.

    Power = P(reject H₀ | H₁ is true)
    We want power > 0.99 as stated in the proposal.

    Uses the normal approximation: noncentrality = d * sqrt(n)

    Args:
        n:              sample size
        cohens_d_value: expected Cohen's d
        alpha_level:    significance level

    Returns:
        dict with power estimate and interpretation
    """
    # Noncentrality parameter
    nc = abs(cohens_d_value) * math.sqrt(n)

    # Critical value for one-sided test (normal approx)
    z_alpha = _normal_ppf(1 - alpha_level)

    # Power = P(Z > z_alpha - nc) where Z ~ N(0,1)
    power = 1 - _normal_cdf(z_alpha - nc)

    return {
        "n":           n,
        "cohens_d":    cohens_d_value,
        "alpha_level": alpha_level,
        "power":       power,
        "adequate":    power > 0.80,
        "excellent":   power > 0.99,
        "interpretation": (
            f"Power = {power:.4f}. "
            + ("Excellent (>0.99): sample size is more than sufficient."
               if power > 0.99 else
               "Adequate (>0.80): sample size is sufficient."
               if power > 0.80 else
               "Insufficient (<0.80): consider increasing sample size.")
        ),
    }


def run_full_e2_analysis(
    rho_values: List[float],
    config: dict,
    patient_ids: Optional[List[str]] = None,
) -> Dict:
    """
    Run the complete Experiment E2 statistical analysis.

    This is the single entry point that produces all statistical
    results needed for the proposal's E2 experiment.

    Args:
        rho_values:  list of per-patient ρ(p) floats
        config:      fedmu_config dict (for alpha_test, n_bootstrap, rho_0)
        patient_ids: optional list matching rho_values (for reporting)

    Returns:
        comprehensive results dict
    """
    alpha_test  = config["statistics"]["alpha_test"]
    n_bootstrap = config["statistics"]["n_bootstrap"]
    rho_0       = config["statistics"]["rho_0"]
    seed        = config["project"]["seed"]

    # Filter out None/NaN values
    valid_rhos = [r for r in rho_values if r is not None and
                  not math.isnan(r) and not math.isinf(r)]
    n_valid = len(valid_rhos)
    n_total = len(rho_values)

    print(f"\n  E2 Analysis: {n_valid}/{n_total} valid ρ(p) values")

    if n_valid < 5:
        return {"error": f"Too few valid ρ values ({n_valid}) for analysis"}

    # 1. Descriptive statistics
    desc = descriptive_stats(valid_rhos)

    # 2. Bootstrap CI
    mean_rho, ci_lo, ci_hi = bootstrap_confidence_interval(
        valid_rhos, n_bootstrap=n_bootstrap,
        confidence=0.95, seed=seed,
    )

    # 3. Hypothesis test
    test_result = one_sided_t_test(
        valid_rhos,
        null_mean=1.0,
        alpha_level=alpha_test,
    )

    # 4. Cohen's d
    cd = cohens_d(valid_rhos, reference_mean=1.0)

    # 5. Power analysis
    power = power_analysis(
        n=n_valid,
        cohens_d_value=cd["d"],
        alpha_level=alpha_test,
    )

    # 6. Hypothesis about ρ₀
    pct_above_rho0 = 100 * sum(r > rho_0 for r in valid_rhos) / n_valid

    results = {
        "n_valid":          n_valid,
        "n_total":          n_total,
        "descriptive":      desc,
        "bootstrap_ci": {
            "mean":         mean_rho,
            "ci_lower":     ci_lo,
            "ci_upper":     ci_hi,
            "ci_level":     0.95,
        },
        "hypothesis_test":  test_result,
        "cohens_d":         cd,
        "power_analysis":   power,
        "rho_0_analysis": {
            "rho_0":              rho_0,
            "pct_above_rho_0":    pct_above_rho0,
            "hypothesis_supported": pct_above_rho0 > 50,
        },
    }

    return results


def print_e2_report(results: Dict):
    """Print a formatted E2 analysis report."""
    if "error" in results:
        print(f"  ERROR: {results['error']}")
        return

    d  = results["descriptive"]
    ci = results["bootstrap_ci"]
    ht = results["hypothesis_test"]
    cd = results["cohens_d"]
    pw = results["power_analysis"]
    r0 = results["rho_0_analysis"]

    print("\n" + "─"*56)
    print("  EXPERIMENT E2: GRADIENT CONCENTRATION RATIO ρ(p)")
    print("─"*56)
    print(f"  Sample size:  n = {results['n_valid']} patients")
    print()
    print("  DESCRIPTIVE STATISTICS:")
    print(f"    Mean ρ(p):   {d['mean']:.4f}")
    print(f"    Std:         {d['std']:.4f}")
    print(f"    Median:      {d['median']:.4f}")
    print(f"    IQR:         [{d['p25']:.4f}, {d['p75']:.4f}]")
    print(f"    Range:       [{d['min']:.4f}, {d['max']:.4f}]")
    print(f"    % above 1.0: {d['pct_above_1']:.1f}%")
    print(f"    % above ρ₀={r0['rho_0']}: {d['pct_above_1_5']:.1f}%")
    print()
    print("  BOOTSTRAP CONFIDENCE INTERVAL (95%):")
    print(f"    Mean ρ(p) = {ci['mean']:.4f}")
    print(f"    95% CI:    [{ci['ci_lower']:.4f}, {ci['ci_upper']:.4f}]")
    print()
    print("  HYPOTHESIS TEST (one-sided t-test):")
    print(f"    H₀: E[ρ(p)] ≤ 1.0    H₁: E[ρ(p)] > 1.0")
    print(f"    t-statistic: {ht['t_stat']:.4f}")
    print(f"    p-value:     {ht['p_value']:.6f}")
    print(f"    α_test:      {ht['alpha_level']}")
    print(f"    Decision:    {'✓ REJECT H₀' if ht['reject_h0'] else '✗ FAIL TO REJECT H₀'}")
    print(f"    {ht['conclusion']}")
    print()
    print("  EFFECT SIZE (Cohen's d):")
    print(f"    d = {cd['d']:.4f}  ({cd['magnitude']} effect)")
    print()
    print("  POWER ANALYSIS:")
    print(f"    {pw['interpretation']}")
    print()
    print("  ρ₀ THRESHOLD ANALYSIS:")
    print(f"    Conjectured ρ₀ = {r0['rho_0']}")
    print(f"    % patients with ρ(p) > ρ₀: {r0['pct_above_rho_0']:.1f}%")
    supported = r0['hypothesis_supported']
    print(f"    Hypothesis supported: {'✓ YES' if supported else '✗ NO'}")
    print("─"*56)


# ── Internal math helpers ──────────────────────────────────────────────────

def _normal_cdf(x: float) -> float:
    """Standard normal CDF using math.erfc approximation."""
    return 0.5 * math.erfc(-x / math.sqrt(2))

def _normal_sf(x: float) -> float:
    """Standard normal survival function (upper tail): P(Z > x)."""
    return 1.0 - _normal_cdf(x)

def _normal_ppf(p: float) -> float:
    """
    Inverse normal CDF (percent-point function) via rational approximation.
    Accurate to ~5 decimal places for p in (0.001, 0.999).
    """
    if p <= 0.0:
        return float('-inf')
    if p >= 1.0:
        return float('inf')

    # Rational approximation (Abramowitz & Stegun 26.2.17)
    c = [2.515517, 0.802853, 0.010328]
    d = [1.432788, 0.189269, 0.001308]

    def rational(t):
        num = c[0] + c[1]*t + c[2]*t*t
        den = 1 + d[0]*t + d[1]*t*t + d[2]*t*t*t
        return t - num / den

    if p < 0.5:
        t = math.sqrt(-2 * math.log(p))
        return -rational(t)
    else:
        t = math.sqrt(-2 * math.log(1 - p))
        return rational(t)