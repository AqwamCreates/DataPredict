# [API Reference](../API.md) - Solvers

| Solver                                                    | Convergence Speed | Computational Efficiency (Non-Cached) | Computational Efficiency (Cached)* | Best For                                                             |
|-----------------------------------------------------------|-------------------|---------------------------------------|------------------------------------|----------------------------------------------------------------------|
| [Gradient](Solvers/Gradient.md)                           | Low               | High                                  | Very High                          | General-Purpose                                                      |
| [ConjugateGradient](Solvers/ConjugateGradient.md)         | Medium            | Medium                                | Medium-High                        | Large Datasets + Most Values Are Zero (a.k.a. Sparse Data)           |
| [GaussNewton](Solvers/GaussNewton.md)                     | Medium-High       | Medium                                | Very High                          | Small-Medium Datasets + Well-Defined Problems                        |
| [LevenbergMarquardt](Solvers/LevenbergMarquardt)          | Medium-High       | Medium                                | Very High                          | Small-Medium Datasets + Poorly-Defined Problems                      |
| [IterativelyReweighted](Solvers/IterativelyReweighted.md) | Medium            | Medium                                | Medium                             | Individual Datapoints Have Different Contribution To Feature Weights |
| [GreedyCoordinate](Solvers/GreedyCoordinate.md)           | Low               | High                                  | Very High                          | Large Datasets + Most Values Are Zero (a.k.a. Sparse Data)           |
| [RandomCoordinate](Solvers/RandomCoordinate.md)           | Low               | Very High                             | Very High                          | Extremely Large Datasets                                             |

\* Computational efficiency due to cache is only applicable to models that has a constant linear expression as inputs:

  * In other words, anything that depends on nested of functions with their own weights applied as its input (regardless if a non-linear function is applied to the input) will not be cached like neural networks.

  * However, even if non-linear transformation is applied, this model would still benefit from caches like BinaryRegression, PoissonRegression, NegativeBinomialRegression and GammaRegression.
