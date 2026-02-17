# [API Reference](../API.md) - Solvers

| Solver                                                    | Convergence Speed | Computational Efficiency (Non-Cached) | Computational Efficiency (Cached)* |
|-----------------------------------------------------------|-------------------|---------------------------------------|------------------------------------|
| [Gradient](Solvers/Gradient.md)                           | Low               | High                                  | Very High                          |
| [ConjugateGradient](Solvers/ConjugateGradient.md)         | Medium            | Medium                                | Medium-High                        |
| [GaussNewton](Solvers/GaussNewton.md)                     | Medium-High       | Medium                                | Very High                          |
| [LevenbergMarquardt](Solvers/LevenbergMarquardt)          | Medium-High       | Medium                                | Very High                          |
| [IterativelyReweighted](Solvers/IterativelyReweighted.md) | Medium            | Medium                                | Medium                             |

\* Computational efficiency due to cache is only applicable to models that has linear expression like LinearRegression, SupportVectorMachine and SupportVectorRegression. However, if the non-linear expression can be converted to a linear expression, then it can be cached as well liked PoissonRegression, NegativeBinomialRegression and GammaRegression.
