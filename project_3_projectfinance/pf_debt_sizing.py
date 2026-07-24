"""
pf_debt_sizing.py - term loan sizing, sculpted repayment, and DSCR.

Two independent debt caps are computed and the lower one governs. The DSCR
cap is the largest loan a flat-1.30x-DSCR sculpted repayment schedule can
fully amortize over the tenor, found by bisection on loan principal - tax
(and therefore CFADS) depends on interest expense, which depends on the
principal, which is exactly what's being solved for, so this can't be
computed in closed form. The leverage cap is 75% of total project cost,
from pf_construction.py. If the leverage cap binds instead, the resulting
DSCR schedule is looser than 1.30x (the deal is over-collateralized on
leverage terms) and the loan amortizes before the tenor ends - both are
valid, reported outcomes, not errors.
"""

import pandas as pd

import pf_assumptions as A


def _year_tax_and_cfads(ebitda, depreciation, interest):
    taxable_income = ebitda - depreciation - interest
    tax = max(0.0, taxable_income * A.TAX_RATE)
    cfads = ebitda - tax
    return tax, cfads


def _terminal_balance(principal, ops_df):
    """
    Forward-simulate the sculpted schedule for a candidate principal with
    no clamping (principal or balance can go negative). Used only to find
    the root for bisection - the unclamped trajectory is a smooth,
    monotonic function of principal, which clamping would break.
    """
    balance = principal
    for _, row in ops_df.iterrows():
        year = int(row["year"])
        interest = balance * A.LOAN_INTEREST_RATE

        if year <= A.MORATORIUM_YEARS:
            continue  # interest-only during moratorium, no principal movement

        _, cfads = _year_tax_and_cfads(row["ebitda"], row["depreciation"], interest)
        target_principal = cfads / A.TARGET_MIN_DSCR - interest
        balance -= target_principal

    return balance


def solve_debt_size(ops_df, leverage_cap_debt, verbose=True, tolerance=1e-6, max_iterations=100):
    """
    Bisection on loan principal so the sculpted schedule's terminal balance
    is exactly zero, then apply the leverage cap and build the final,
    clamped repayment schedule for whichever principal governs.
    """
    lo, hi = 0.0, leverage_cap_debt * 3

    if verbose:
        print(f"debt sizing: solving DSCR-cap principal for target min DSCR {A.TARGET_MIN_DSCR:.2f}x")

    for iteration in range(1, max_iterations + 1):
        mid = (lo + hi) / 2
        terminal = _terminal_balance(mid, ops_df)

        if verbose:
            print(f"  iter {iteration:2d}: principal={mid/A.CR_TO_INR:10.2f} Cr "
                  f"terminal_balance={terminal/A.CR_TO_INR:+10.4f} Cr")

        if abs(terminal) / max(leverage_cap_debt, 1.0) < tolerance:
            break

        # terminal_balance is increasing in principal: too much principal
        # leaves a positive balance at maturity, too little overpays it.
        if terminal > 0:
            hi = mid
        else:
            lo = mid
    else:
        raise RuntimeError(f"debt sizing did not converge within {max_iterations} iterations")

    dscr_cap_debt = mid
    sanctioned_debt = min(dscr_cap_debt, leverage_cap_debt)
    binding_constraint = "dscr" if dscr_cap_debt <= leverage_cap_debt else "leverage"

    if verbose:
        print(f"  DSCR cap: {dscr_cap_debt/A.CR_TO_INR:.2f} Cr | leverage cap: "
              f"{leverage_cap_debt/A.CR_TO_INR:.2f} Cr | binding: {binding_constraint} "
              f"({sanctioned_debt/A.CR_TO_INR:.2f} Cr)")

    schedule = amortize_debt(sanctioned_debt, ops_df)

    return {
        "dscr_cap_debt": dscr_cap_debt,
        "leverage_cap_debt": leverage_cap_debt,
        "sanctioned_debt": sanctioned_debt,
        "binding_constraint": binding_constraint,
        "schedule": schedule,
        # debt_service > 0 excludes years after the loan is already fully
        # repaid (dscr is NaN there), not the moratorium year - year 1 has
        # interest-only debt service (still > 0) and stays in this min, since
        # a covenant breach on interest coverage during moratorium is a real
        # risk a stress scenario could surface, not something to hide.
        "min_dscr": schedule.loc[schedule["debt_service"] > 0, "dscr"].min(),
        "iterations": iteration,
    }


def amortize_debt(principal, ops_df):
    """
    Final, clamped repayment schedule for a fixed principal. Principal is
    capped at the remaining balance and the balance floors at zero - once
    the loan is repaid (which happens before the tenor ends if the
    leverage cap governs at less than the DSCR-cap amount), later years
    carry zero debt service.
    """
    rows = []
    balance = principal

    for _, row in ops_df.iterrows():
        year = int(row["year"])
        opening_balance = balance
        interest = opening_balance * A.LOAN_INTEREST_RATE

        if year <= A.MORATORIUM_YEARS:
            principal_paid = 0.0
            tax, cfads = _year_tax_and_cfads(row["ebitda"], row["depreciation"], interest)
        else:
            tax, cfads = _year_tax_and_cfads(row["ebitda"], row["depreciation"], interest)
            target_principal = cfads / A.TARGET_MIN_DSCR - interest
            principal_paid = min(max(target_principal, 0.0), opening_balance)

        debt_service = interest + principal_paid
        balance = opening_balance - principal_paid
        dscr = cfads / debt_service if debt_service > 0 else float("nan")

        rows.append({
            "year": year,
            "opening_balance": opening_balance,
            "interest": interest,
            "principal": principal_paid,
            "debt_service": debt_service,
            "closing_balance": balance,
            "tax": tax,
            "cfads": cfads,
            "dscr": dscr,
        })

    return pd.DataFrame(rows)


def evaluate_fixed_schedule(base_schedule, ops_df):
    """
    Recompute tax, CFADS, and DSCR against an already-sized schedule -
    interest and debt_service come from base_schedule, not resolved fresh -
    against a different ebitda/depreciation profile from ops_df. Used by
    pf_sensitivity.py; see docs/project_3_projectfinance.md ("The
    sensitivity-freezing design flaw") for why debt isn't resized here.
    """
    rows = []
    for (_, base_row), (_, ops_row) in zip(base_schedule.iterrows(), ops_df.iterrows()):
        interest = base_row["interest"]
        debt_service = base_row["debt_service"]
        tax, cfads = _year_tax_and_cfads(ops_row["ebitda"], ops_row["depreciation"], interest)
        dscr = cfads / debt_service if debt_service > 0 else float("nan")

        rows.append({
            "year": int(ops_row["year"]),
            "interest": interest,
            "debt_service": debt_service,
            "tax": tax,
            "cfads": cfads,
            "dscr": dscr,
        })

    return pd.DataFrame(rows)


if __name__ == "__main__":
    from pf_construction import solve_construction_financing
    from pf_operations import generate_operating_series

    construction = solve_construction_financing(verbose=False)
    ops_df = generate_operating_series(construction["total_project_cost"])
    result = solve_debt_size(ops_df, construction["leverage_cap_debt"])
    print(result["schedule"].to_string(index=False))
    print(f"min DSCR across the schedule: {result['min_dscr']:.4f}")
    print(f"final closing balance: {result['schedule']['closing_balance'].iloc[-1]/A.CR_TO_INR:.6f} Cr")
