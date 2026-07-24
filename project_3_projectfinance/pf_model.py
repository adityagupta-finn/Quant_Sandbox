"""
pf_model.py - orchestrator. Runs the base case end-to-end and prints the
full report: construction financing, debt sizing, the annual cash
flow/DSCR schedule, returns, and the sensitivity tornado.

Usage: $ python pf_model.py
"""

import pandas as pd

import pf_assumptions as A
from pf_construction import solve_construction_financing
from pf_operations import generate_operating_series
from pf_debt_sizing import solve_debt_size
from pf_returns import summarize_returns
from pf_sensitivity import run_tornado


def run_base_case():
    print("200MW solar SPV debt-sizing model - base case")
    print(f"capacity {A.CAPACITY_MW:.0f} MW | CUF (P90) {A.CUF_P90*100:.1f}% | "
          f"tariff INR {A.TARIFF_PER_KWH:.2f}/kWh | capex INR {A.CAPEX_PER_MW_CR:.2f} Cr/MW")
    print()

    print("--- construction financing ---")
    construction = solve_construction_financing(verbose=True)
    print()

    ops_df = generate_operating_series(construction["total_project_cost"])

    print("--- debt sizing ---")
    debt_result = solve_debt_size(ops_df, construction["leverage_cap_debt"], verbose=True)
    print()

    print("--- annual schedule ---")
    schedule = debt_result["schedule"].copy()
    display = pd.DataFrame({
        "year": schedule["year"],
        "ebitda_cr": ops_df["ebitda"] / A.CR_TO_INR,
        "interest_cr": schedule["interest"] / A.CR_TO_INR,
        "principal_cr": schedule["principal"] / A.CR_TO_INR,
        "debt_service_cr": schedule["debt_service"] / A.CR_TO_INR,
        "cfads_cr": schedule["cfads"] / A.CR_TO_INR,
        "dscr": schedule["dscr"],
        "closing_balance_cr": schedule["closing_balance"] / A.CR_TO_INR,
    })
    print(display.to_string(index=False, float_format=lambda x: f"{x:.2f}"))
    print(f"min DSCR: {debt_result['min_dscr']:.4f} | binding constraint: {debt_result['binding_constraint']}")
    print()

    print("--- returns ---")
    returns = summarize_returns(construction, ops_df, debt_result, verbose=True)
    print()

    print("--- sensitivity (tornado) ---")
    tornado = run_tornado(verbose=True)
    print()

    return {
        "construction": construction,
        "ops_df": ops_df,
        "debt_result": debt_result,
        "returns": returns,
        "tornado": tornado,
    }


if __name__ == "__main__":
    run_base_case()
