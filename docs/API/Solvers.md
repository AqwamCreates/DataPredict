# [API Reference](../API.md) - Solvers

| Solver                                | Convergence Speed | Computational Efficiency (Non-Cached) | Computational Efficiency (Cached)* |
|---------------------------------------|-------------------|---------------------------------------|------------------------------------|
| [Gradient](Solvers/Gradient.md)       | Low               | High                                  | Very High                          |
| [GaussNewton](Solvers/GaussNewton.md) | Medium-High       | Medium                                | Medium-High                        |

\* Computational efficiency due to cache is only applicable to linear models like LinearRegression, SupportVectorRegression and SupportVectorMachine.
