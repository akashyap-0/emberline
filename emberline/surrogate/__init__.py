"""Neural fire-spread surrogate: dataset gen, UNet model, training, eval.

Pipeline: ``python -m emberline.surrogate.data`` (shards) ->
``python -m emberline.surrogate.train`` (checkpoints) ->
``python -m emberline.surrogate.eval`` (metrics -> REPORT.md).
"""
