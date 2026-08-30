"""Emberline: a fully simulated wildfire early-warning stack.

Modules
-------
worldgen   Procedural terrain, fuel, town, roads, and wind.
firesim    Rothermel-inspired cellular-automata surface fire spread.
surrogate  Neural surrogate (UNet) forecasting fire spread.
sensors    Gaussian-plume driven synthetic sensor streams + confounders.
detect     On-node smoke classifier (1D-CNN vs GBM baseline).
mesh       Discrete-event LoRa-class mesh with escalation ladder.
foresight  Digital-twin assembly, probability cones, evacuation routing, CAP drafts.
demo       Scripted end-to-end terminal demo.
adapters   Stub interfaces for real data sources (USGS, LANDFIRE, OSM).

Everything is synthetic. Nothing here detects real fires; metrics reflect
simulated worlds only.
"""

__version__ = "0.1.0"
