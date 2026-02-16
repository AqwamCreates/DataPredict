# [API Reference](../API.md) - Solvers

| Solver                                                    | Convergence Speed | Computational Efficiency (Non-Cached) | Computational Efficiency (Cached)* |
|-----------------------------------------------------------|-------------------|---------------------------------------|------------------------------------|
| [Gradient](Solvers/Gradient.md)                           | Low               | High                                  | Very High                          |
| [ConjugateGradient](Solvers/ConjugateGradient.md)         | Medium            | Medium                                | Medium-High                        |
| [GaussNewton](Solvers/GaussNewton.md)                     | Medium-High       | Medium                                | Very High                          |
| [LevenbergMarquardt](Solvers/LevenbergMarquardt)          | Medium-High       | Medium                                | Very High                          |
| [IterativelyReweighted](Solvers/IterativelyReweighted.md) | Medium            | Medium                                | Medium                             |

\* Computational efficiency due to cache is only applicable to linear models like LinearRegression, SupportVectorRegression and SupportVectorMachine.
