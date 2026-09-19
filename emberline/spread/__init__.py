"""Real-data fire-spread models on NDWS (1 km / daily satellite scale).

This regime is NOT the node-scale simulator's: 1 km pixels at daily cadence
versus 10 m cells at minute cadence. Results here are the real-data
reference point for spread modelling, not a claim about the 10 m system.
Real data only: everything loads through emberline.data (guard enforced),
and the UNet trains from scratch.
"""
