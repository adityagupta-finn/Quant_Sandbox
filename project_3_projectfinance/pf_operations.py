"""
pf_operations.py - operations-phase annual series, years 1-15 post-COD.

Everything here is debt-independent: generation, revenue, opex, and tax
depreciation depend only on the plant and total project cost, never on how
the project is financed. Interest expense (financing-dependent) is applied
downstream in pf_debt_sizing.py, which combines these series with a
candidate debt schedule to compute tax and CFADS.
"""

import pandas as pd

import pf_assumptions as A


def generate_operating_series(total_project_cost, cuf=A.CUF_P90, tariff_per_kwh=A.TARIFF_PER_KWH):
    """
    Build the year 1..TENOR_YEARS debt-independent series.

    Depreciation uses total_project_cost as the depreciable tax block (the
    whole project, not just the debt-funded portion - depreciation is a tax
    concept tied to asset cost, not financing). WDV carries forward; once
    the block approaches zero the depreciation shield simply tapers off,
    it isn't reset or floored.
    """
    rows = []
    opening_wdv = total_project_cost

    for year in range(1, A.TENOR_YEARS + 1):
        generation_kwh = A.CAPACITY_MW * 1000 * cuf * A.HOURS_PER_YEAR * (1 - A.DEGRADATION_RATE) ** (year - 1)
        revenue = generation_kwh * tariff_per_kwh
        opex = A.CAPACITY_MW * A.OM_COST_PER_MW_LAKH * A.LAKH_TO_INR * (1 + A.OM_ESCALATION) ** (year - 1)
        depreciation = opening_wdv * A.DEPRECIATION_RATE_WDV
        closing_wdv = opening_wdv - depreciation
        ebitda = revenue - opex

        rows.append({
            "year": year,
            "generation_kwh": generation_kwh,
            "revenue": revenue,
            "opex": opex,
            "depreciation": depreciation,
            "closing_wdv": closing_wdv,
            "ebitda": ebitda,
        })

        opening_wdv = closing_wdv

    return pd.DataFrame(rows)


if __name__ == "__main__":
    from pf_construction import solve_construction_financing

    construction = solve_construction_financing(verbose=False)
    df = generate_operating_series(construction["total_project_cost"])
    print(df.to_string(index=False))
