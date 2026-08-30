"""Stretch (c): back-of-envelope insurance economics of a node deployment.

Every input is a config knob (``economics:`` in config.yaml) and every input
is an ILLUSTRATIVE assumption, not data. The point of the module is the
*shape* of the argument insurers hear - deployment cost vs expected annual
loss averted - with each assumption exposed for them to replace:

    expected_annual_loss   = P(ignition) x town_value x loss_fraction
    expected_loss_averted  = expected_annual_loss x loss_reduction_with_warning
    (loss_reduction applies only when the mesh actually gains warning time -
    see the hindcast section for the measured warning-minutes-gained.)

Run: ``python -m emberline.foresight.economics``
"""

from __future__ import annotations

from ..config import load_config
from ..report import save_metrics, update_section


def main() -> None:
    cfg = load_config()
    e = cfg["economics"]
    n = int(e["n_nodes"])
    capex = n * (float(e["node_unit_usd"]) + float(e["install_per_node_usd"]))
    opex = n * float(e["annual_maint_per_node_usd"])
    years = int(e["horizon_years"])
    total_cost = capex + opex * years

    town_value = float(e["town_structures"]) * float(e["avg_structure_value_usd"])
    eal = float(e["annual_ignition_prob"]) * town_value * float(e["loss_fraction_no_warning"])
    averted_annual = eal * float(e["loss_reduction_with_warning"])
    averted_horizon = averted_annual * years
    payback_years = total_cost / averted_annual if averted_annual > 0 else float("inf")
    roi = averted_horizon / total_cost if total_cost > 0 else float("inf")

    m = {"n_nodes": n, "capex_usd": capex, "opex_annual_usd": opex,
         "cost_10y_usd": total_cost, "town_value_usd": town_value,
         "expected_annual_loss_usd": eal, "averted_annual_usd": averted_annual,
         "payback_years": round(payback_years, 3), "roi_10y_x": round(roi, 1),
         "inputs": {k: v for k, v in e.items()}}
    save_metrics("economics", m)
    update_section("economics", f"""
### Deployment economics (stretch — illustrative inputs, all in config.yaml)

Regenerate: `python -m emberline.foresight.economics`. Every number derives
from the `economics:` config block; the assumptions are placeholders an
insurer/county would replace, and the loss-reduction factor only applies when
the mesh actually gains warning time (see hindcast).

| quantity | value |
|---|---|
| deployment ({n} nodes) capex | ${capex:,.0f} |
| 10-year cost (capex + maintenance) | ${total_cost:,.0f} |
| town structure value | ${town_value:,.0f} |
| expected annual wildfire loss (assumed) | ${eal:,.0f} |
| expected annual loss averted by warning | ${averted_annual:,.0f} |
| payback period | {payback_years:.2f} years |
| 10-year averted-loss : cost ratio | {roi:,.0f}x |
""")
    print(f"cost 10y ${total_cost:,.0f} vs averted/yr ${averted_annual:,.0f} "
          f"-> payback {payback_years:.2f} y; wrote metrics/economics.json + REPORT section")


if __name__ == "__main__":
    main()
