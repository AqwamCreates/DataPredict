# [API Reference](../API.md) - Solvers

| Solver                                                    | Convergence Speed | Computational Efficiency (Non-Cached) | Computational Efficiency (Cached)* | Optimizer Compatibility | Properties | Best For                                                             |
|-----------------------------------------------------------|-------------------|---------------------------------------|------------------------------------|-------------------------|------------|----------------------------------------------------------------------|
| [Gradient](Solvers/Gradient.md)                           | Low               | High                                  | Very High                          | Very High               | 🔰        | General-Purpose                                                      |
| [ConjugateGradient](Solvers/ConjugateGradient.md)         | Medium            | Medium                                | Medium-High                        | High                    | 🛡️ 📈     | Large Datasets + Most Values Are Zero (a.k.a. Sparse Data)          |
| [GaussNewton](Solvers/GaussNewton.md)                     | Medium-High       | Medium                                | Very High                          | Medium                  | 🔢 🎯     | Small-Medium Datasets + Well-Defined Problems                        |
| [LevenbergMarquardt](Solvers/LevenbergMarquardt)          | Medium-High       | Medium                                | Very High                          | Medium                  | 🔢 ⚠️     | Small-Medium Datasets + Poorly-Defined Problems                      |
| [IterativelyReweighted](Solvers/IterativelyReweighted.md) | Medium            | Medium                                | Medium                             | Medium                  | 🔢 🛡️	    | Individual Datapoints Have Different Contribution To Feature Weights |
| [GreedyCoordinate](Solvers/GreedyCoordinate.md)           | Low               | High                                  | Very High                          | Very Low                | 🛡️ 📈     | Large Datasets + Most Values Are Zero (a.k.a. Sparse Data)           |
| [RandomCoordinate](Solvers/RandomCoordinate.md)           | Low               | Very High                             | Very High                          | Very Low                | 🛡️ 📈     | Extremely Large Datasets                                            |

\* Computational efficiency due to cache is only applicable to models that has a constant linear expression as inputs:

  * In other words, anything that depends on nested of functions with their own weights applied as its input (regardless if a non-linear function is applied to the input) will not be cached like neural networks.

  * However, even if non-linear transformation is applied, this model would still benefit from caches like BinaryRegression, PoissonRegression, NegativeBinomialRegression and GammaRegression.

### Legend

| Icon | Name                        | Description                                                        |
|------|-----------------------------|--------------------------------------------------------------------|
| 🔰   | Beginner Solver             | Commonly taught to beginners.                                      |
| 🔢   | Data Constraint             | The number of data cannot be less than the number of features.     |
| 🛡️	   | Noise Resistant            |	Can handle randomness / unclean data.                               |
| 🎯   | Exact Solution              | Finds exact optimum (for linear problems).                         |
| 📈   | Scales Well                 | Handles large datasets.                                            |
| ⚠️   | Double Regularization Issue | Contains a regularization term and may conflict with regularizers. |

### Potential Solver Additions

* Secant method

* Brent's method

* Bisection method (Low priority; slow convergence and slow to compute.) 
