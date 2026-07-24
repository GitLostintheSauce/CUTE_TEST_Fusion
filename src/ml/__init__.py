"""Machine-learning surrogate for CUTE equilibrium reconstruction.

This package trains a fast neural-network surrogate that maps magnetic
diagnostic signals to plasma parameters, as an accelerated alternative to
iterative Grad-Shafranov reconstruction.

Modules:
    physics    Analytic circular-loop Green's functions (flux and B field).
    dataset    Reduced-physics forward model and labeled dataset generation.
    mlp        A from-scratch NumPy multilayer perceptron with Adam.
    surrogate  High-level train/predict/save/load API for the surrogate.
    baseline   Classical least-squares inversion, for a fair speed benchmark.
"""
