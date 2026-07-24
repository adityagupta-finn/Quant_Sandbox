"""
pf_sensitivity.py - one-at-a-time (tornado) sensitivity grid.

The loan is sized once against the base case and that schedule is held
fixed across every scenario rather than resolved per-scenario - see
docs/project_3_projectfinance.md ("The sensitivity-freezing design flaw")
for why. Capex overrun / COD delay scenarios are funded entirely by
additional equity, since the frozen loan doesn't grow with cost.
"""

import pandas as pd

import pf_assumptions as A
from pf_construction import solve_construction_financing
from pf_operations import generate_operating_series
from pf_debt_sizing import solve_debt_size, evaluate_fixed_schedule
from pf_returns import solve_irr_cross_checked


def run_scenario(
    base_schedule,
    base_sanctioned_debt,
    cuf=A.CUF_P90,
    tariff_per_kwh=A.TARIFF_PER_KWH,
    capex_overrun_pct=0.0,
    cod_delay_months=0,
):
    construction = solve_construction_financing(
        capex_per_mw_cr=A.CAPEX_PER_MW_CR * (1 + capex_overrun_pct),
        construction_months=A.CONSTRUCTION_MONTHS + cod_delay_months,
        verbose=False,
    )
    ops_df = generate_operating_series(
        construction["total_project_cost"], cuf=cuf, tariff_per_kwh=tariff_per_kwh
    )
    evaluation = evaluate_fixed_schedule(base_schedule, ops_df)

    equity_amount = construction["total_project_cost"] - base_sanctioned_debt
    equity_cashflows = [-equity_amount] + list(evaluation["cfads"] - evaluation["debt_service"])
    equity_irr = solve_irr_cross_checked(equity_cashflows)

    min_dscr = evaluation.loc[evaluation["debt_service"] > 0, "dscr"].min()

    return {"min_dscr": min_dscr, "equity_irr": equity_irr, "equity_amount_cr": equity_amount / A.CR_TO_INR}


def run_tornado(verbose=True):
    base_construction = solve_construction_financing(verbose=False)
    base_ops = generate_operating_series(base_construction["total_project_cost"])
    base_debt_result = solve_debt_size(base_ops, base_construction["leverage_cap_debt"], verbose=False)
    base_schedule = base_debt_result["schedule"]
    base_sanctioned_debt = base_debt_result["sanctioned_debt"]

    rows = [{"variable": "base case", "level": "base", **run_scenario(base_schedule, base_sanctioned_debt)}]

    for cuf, label in [(0.22, "downside (22%)"), (0.26, "upside (26%)")]:
        rows.append({"variable": "CUF", "level": label, **run_scenario(base_schedule, base_sanctioned_debt, cuf=cuf)})

    rows.append({
        "variable": "tariff", "level": "downside (-5%)",
        **run_scenario(base_schedule, base_sanctioned_debt, tariff_per_kwh=A.TARIFF_PER_KWH * 0.95),
    })

    for overrun, label in [(0.10, "+10%"), (0.20, "+20%")]:
        rows.append({
            "variable": "capex overrun", "level": label,
            **run_scenario(base_schedule, base_sanctioned_debt, capex_overrun_pct=overrun),
        })

    for delay, label in [(3, "+3 months"), (6, "+6 months")]:
        rows.append({
            "variable": "COD delay", "level": label,
            **run_scenario(base_schedule, base_sanctioned_debt, cod_delay_months=delay),
        })

    df = pd.DataFrame(rows)

    if verbose:
        print(df.to_string(index=False))

    return df


if __name__ == "__main__":
    run_tornado()
