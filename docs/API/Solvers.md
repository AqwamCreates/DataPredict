# [API Reference](../API.md) - Solvers

| Solver                                                    | Convergence Speed | Computational Efficiency (Non-Cached) | Computational Efficiency (Cached)* | Use Cases                                                            |
|-----------------------------------------------------------|-------------------|---------------------------------------|------------------------------------|----------------------------------------------------------------------|
| [Gradient](Solvers/Gradient.md)                           | Low               | High                                  | Very High                          | General Applications                                                 |
| [ConjugateGradient](Solvers/ConjugateGradient.md)         | Medium            | Medium                                | Medium-High                        | Large Datasets + Most Values Are Zero (a.k.a. Sparse Data)           |
| [GaussNewton](Solvers/GaussNewton.md)                     | Medium-High       | Medium                                | Very High                          | Small-Medium Datasets + Well-Defined Problems                        |
| [LevenbergMarquardt](Solvers/LevenbergMarquardt)          | Medium-High       | Medium                                | Very High                          | Small-Medium Datasets + Poorly-Defined Problems                      |
| [IterativelyReweighted](Solvers/IterativelyReweighted.md) | Medium            | Medium                                | Medium                             | Individual Datapoints Have Different Contribution To Feature Weights |

\* Computational efficiency due to cache is only applicable to models that has linear expression. In other words, anything that depends on nested of functions as its input (and not on linear transformation itself) will not be cached like neural networks.
