"""
pf_construction.py - construction-phase capex drawdown and IDC capitalization.

The debt/equity split of each monthly tranche is NOT fixed at 75:25
directly - it's solved so the *final* debt balance at COD (draws plus all
capitalized IDC) equals exactly 75% of total project cost. IDC depends on
how much of each tranche is debt-funded, so this is a bisection, not a
closed-form split. See docs/project_3_projectfinance.md ("Why bisection
converges here") for the monotonicity argument.
"""

import pandas as pd

import pf_assumptions as A


def simulate_monthly_draws(base_capex_inr, construction_months, debt_draw_fraction, annual_rate):
    """
    Simulate monthly capex draws and IDC capitalization for a given debt
    draw fraction. Draws happen at the start of each month; interest for
    the month accrues on the balance immediately after that month's draw.

    Returns (schedule_df, final_debt_balance, total_idc).
    """
    monthly_rate = annual_rate / 12
    monthly_capex = base_capex_inr / construction_months

    rows = []
    balance = 0.0
    total_idc = 0.0

    for month in range(1, construction_months + 1):
        debt_draw = monthly_capex * debt_draw_fraction
        equity_draw = monthly_capex * (1 - debt_draw_fraction)

        balance += debt_draw
        interest = balance * monthly_rate
        balance += interest
        total_idc += interest

        rows.append({
            "month": month,
            "capex_draw": monthly_capex,
            "debt_draw": debt_draw,
            "equity_draw": equity_draw,
            "interest_accrued": interest,
            "closing_debt_balance": balance,
        })

    return pd.DataFrame(rows), balance, total_idc


def solve_construction_financing(
    capex_per_mw_cr=A.CAPEX_PER_MW_CR,
    construction_months=A.CONSTRUCTION_MONTHS,
    verbose=True,
    tolerance=1e-9,
    max_iterations=100,
):
    """Bisection on the draw fraction; see module docstring for why it converges."""
    base_capex_inr = A.CAPACITY_MW * capex_per_mw_cr * A.CR_TO_INR
    target_debt_ratio = A.DEBT_RATIO
    annual_rate = A.LOAN_INTEREST_RATE

    lo, hi = 0.0, 1.0
    schedule, debt_balance, total_idc = None, None, None

    if verbose:
        print("construction financing: solving debt draw fraction for target leverage "
              f"{target_debt_ratio:.4f}")

    for iteration in range(1, max_iterations + 1):
        mid = (lo + hi) / 2
        schedule, debt_balance, total_idc = simulate_monthly_draws(
            base_capex_inr, construction_months, mid, annual_rate
        )
        tpc = base_capex_inr + total_idc
        leverage = debt_balance / tpc
        error = leverage - target_debt_ratio

        if verbose:
            print(f"  iter {iteration:2d}: debt_fraction={mid:.6f} leverage={leverage:.6f} "
                  f"error={error:+.2e}")

        if abs(error) < tolerance:
            break

        if error > 0:
            hi = mid
        else:
            lo = mid
    else:
        raise RuntimeError(f"construction financing did not converge within {max_iterations} iterations")

    tpc = base_capex_inr + total_idc
    equity_total = tpc - debt_balance

    if verbose:
        print(f"  converged: TPC={tpc/A.CR_TO_INR:.2f} Cr, IDC={total_idc/A.CR_TO_INR:.2f} Cr, "
              f"construction debt={debt_balance/A.CR_TO_INR:.2f} Cr, "
              f"equity={equity_total/A.CR_TO_INR:.2f} Cr")

    return {
        "schedule": schedule,
        "base_capex": base_capex_inr,
        "total_idc": total_idc,
        "total_project_cost": tpc,
        "leverage_cap_debt": debt_balance,
        # only the actual equity check if the leverage cap governs - see
        # docs/project_3_projectfinance.md ("The equity true-up bug") if
        # the DSCR cap binds lower instead.
        "equity_amount": equity_total,
        "iterations": iteration,
    }


if __name__ == "__main__":
    solve_construction_financing()
