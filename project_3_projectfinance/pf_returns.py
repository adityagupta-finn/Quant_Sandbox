"""
pf_returns.py - Project IRR and Equity IRR.

Project IRR treats the whole project cost as if it were funded by one
pool of capital with no debt: -TPC at t=0, then each year's EBITDA less
tax computed with no interest shield at all. Equity IRR is the actual
sponsor return: -equity at t=0, then each year's CFADS less the real debt
service that was sculpted in pf_debt_sizing.py. The gap between the two is
the leverage effect - debt priced below the project's own return, with
its interest tax-deductible on top, gets amplified equity returns above
what the unlevered project earns on its own.

No numpy_financial/scipy dependency in this repo, so IRR is solved twice,
independently, and cross-checked: Newton-Raphson (fast) and bisection
(slow but robust to a bad starting guess). Both operate on the same NPV
function; conventional cash flows here (one negative t=0, all positive
afterward) have a single root, so bisection's monotonicity assumption
holds.
"""

import pf_assumptions as A


def npv(rate, cashflows):
    return sum(cf / (1 + rate) ** t for t, cf in enumerate(cashflows))


def _npv_derivative(rate, cashflows):
    return sum(-t * cf / (1 + rate) ** (t + 1) for t, cf in enumerate(cashflows))


def irr_newton(cashflows, guess=0.1, tolerance=1e-10, max_iterations=200):
    """
    tolerance is relative to the magnitude of the t=0 cash flow, not an
    absolute NPV threshold - cash flows here run in the billions of rupees,
    where float precision alone limits absolute NPV to roughly 1e-7, so an
    absolute 1e-10 threshold could never be satisfied.
    """
    scale = abs(cashflows[0])
    rate = guess
    for _ in range(max_iterations):
        value = npv(rate, cashflows)
        if abs(value) / scale < tolerance:
            return rate
        derivative = _npv_derivative(rate, cashflows)
        rate -= value / derivative
    raise RuntimeError("IRR (Newton) did not converge")


def irr_bisection(cashflows, lo=-0.99, hi=10.0, tolerance=1e-10, max_iterations=200):
    """Same relative-tolerance reasoning as irr_newton."""
    scale = abs(cashflows[0])
    f_lo, f_hi = npv(lo, cashflows), npv(hi, cashflows)
    if f_lo * f_hi > 0:
        raise RuntimeError("IRR (bisection) bracket does not span a sign change")

    for _ in range(max_iterations):
        mid = (lo + hi) / 2
        f_mid = npv(mid, cashflows)
        if abs(f_mid) / scale < tolerance:
            return mid
        if f_lo * f_mid < 0:
            hi = mid
        else:
            lo, f_lo = mid, f_mid
    raise RuntimeError("IRR (bisection) did not converge")


def solve_irr_cross_checked(cashflows, cross_check_tolerance=1e-6):
    irr_a = irr_newton(cashflows)
    irr_b = irr_bisection(cashflows)
    if abs(irr_a - irr_b) > cross_check_tolerance:
        raise RuntimeError(
            f"IRR cross-check mismatch: Newton={irr_a:.6f} vs bisection={irr_b:.6f}"
        )
    return irr_a


def compute_project_irr(total_project_cost, ops_df):
    """Unlevered, post-tax cash flows: -TPC at t=0, no interest shield thereafter."""
    cashflows = [-total_project_cost]
    for _, row in ops_df.iterrows():
        unlevered_tax = max(0.0, (row["ebitda"] - row["depreciation"]) * A.TAX_RATE)
        cashflows.append(row["ebitda"] - unlevered_tax)
    return solve_irr_cross_checked(cashflows), cashflows


def compute_equity_irr(total_project_cost, sanctioned_debt, debt_schedule):
    """
    Levered cash flows: -equity at t=0, then CFADS less actual debt service.
    Equity check is total_project_cost - sanctioned_debt, not construction's
    own equity_amount - see docs/project_3_projectfinance.md ("The equity
    true-up bug") for why those aren't interchangeable.
    """
    equity_amount = total_project_cost - sanctioned_debt
    cashflows = [-equity_amount]
    for _, row in debt_schedule.iterrows():
        cashflows.append(row["cfads"] - row["debt_service"])
    return solve_irr_cross_checked(cashflows), cashflows


def summarize_returns(construction, ops_df, debt_result, verbose=True):
    project_irr, project_cashflows = compute_project_irr(construction["total_project_cost"], ops_df)
    equity_irr, equity_cashflows = compute_equity_irr(
        construction["total_project_cost"], debt_result["sanctioned_debt"], debt_result["schedule"]
    )
    leverage_spread = equity_irr - project_irr

    if verbose:
        print(f"Project IRR (unlevered, post-tax): {project_irr*100:.2f}%")
        print(f"Equity IRR (post-debt-service):     {equity_irr*100:.2f}%")
        print(f"Leverage effect (Equity - Project):  {leverage_spread*100:+.2f} pp")

    return {
        "project_irr": project_irr,
        "equity_irr": equity_irr,
        "leverage_spread": leverage_spread,
        "project_cashflows": project_cashflows,
        "equity_cashflows": equity_cashflows,
    }


if __name__ == "__main__":
    from pf_construction import solve_construction_financing
    from pf_operations import generate_operating_series
    from pf_debt_sizing import solve_debt_size

    construction = solve_construction_financing(verbose=False)
    ops_df = generate_operating_series(construction["total_project_cost"])
    debt_result = solve_debt_size(ops_df, construction["leverage_cap_debt"], verbose=False)
    summarize_returns(construction, ops_df, debt_result)
