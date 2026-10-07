"""Coupling this pipeline's equilibria to external transport models.

Proof of concept only. The one coupling here feeds a TokaMaker equilibrium of
CUTE into QLKNN_7_11, the turbulent-transport neural network in Google
DeepMind's open-source ``fusion_surrogates`` library (the model TORAX uses by
default). Its job is to find the roadblocks, not to produce trustworthy
transport numbers. See ``docs/qlknn_poc.md``.

Modules:
    qlknn_inputs  Pure NumPy: equilibrium + assumed profiles -> QLKNN inputs,
                  and a check against the ranges QLKNN was trained on.
    qlknn_runner  Thin wrapper that loads and runs the QLKNN model. Imports
                  ``fusion_surrogates`` lazily so the rest of the repo never
                  needs it.
"""
