# [API Reference](../API.md) - Solvers

## General Information

| Solver                                                              | Convergence Speed | Computational Efficiency (Non-Cached) | Computational Efficiency (Cached)* | Optimizer Compatibility | Properties | Best For                                                             |
|---------------------------------------------------------------------|-------------------|---------------------------------------|------------------------------------|-------------------------|------------|----------------------------------------------------------------------|
| [Gradient](Solvers/Gradient.md)                                     | Low               | High                                  | Very High                          | Very High               | 🔰 📈     | General-Purpose                                                      |
| [ConjugateGradient](Solvers/ConjugateGradient.md)                   | Medium            | Medium                                | Medium-High                        | High                    | 🛡️ 📈     | Large Datasets + Most Values Are Zero (a.k.a. Sparse Data)           |
| [NonLinearConjugateGradient](Solvers/NonLinearConjugateGradient.md) | Medium            | Medium                                | Medium-High                        | High                    | 🛡️ 📈     | Large Datasets + Most Values Are Zero (a.k.a. Sparse Data)           |
| [GaussNewton](Solvers/GaussNewton.md)                               | Medium-High       | Medium                                | Very High                          | Medium                  | 🔢 🎯     | Small-Medium Datasets + Well-Defined Problems                        |
| [LevenbergMarquardt](Solvers/LevenbergMarquardt)                    | Medium-High       | Medium                                | Very High                          | Medium                  | 🔢 ⚠️     | Small-Medium Datasets + Poorly-Defined Problems                      |
| [IterativelyReweighted](Solvers/IterativelyReweighted.md)           | Medium            | Medium                                | Medium                             | Medium                  | 🔢 🛡️	    | Individual Datapoints Have Different Contribution To Feature Weights |
| [GreedyCoordinate](Solvers/GreedyCoordinate.md)                     | Low               | High                                  | Very High                          | Very Low                | 🛡️ 📈     | Large Datasets + Most Values Are Zero (a.k.a. Sparse Data)           |
| [RandomCoordinate](Solvers/RandomCoordinate.md)                     | Low               | Very High                             | Very High                          | Very Low                | 🛡️ 📈     | Extremely Large Datasets                                             |
| [GaussSeidel](Solvers/GaussSeidel.md)                               | Low               | Medium                                | Medium-High                        | Very Low                | 🔢 💥     | Alternative To Gradient                                              |
| [Jacobi](Solvers/Jacobi.md)                                         | Low               | Medium-Low                            | Medium                             | Very Low                | 🔢 💥     | Alternative To Gradient                                              |

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
| 💥   | Explosion-Prone             | Tend to give huge values.                                          |

## Number Of Cache Operations

> m: Number Of Data n: Number Of Features

| Solver                                                    | Number Of Operations Reduced On Cached | Remaining Number Of Operations On Non-Cache |
|-----------------------------------------------------------|----------------------------------------|---------------------------------------------|
| [Gradient](Solvers/Gradient.md)                           | O(mn)                                  | O(n^2m)                                     |
| [ConjugateGradient](Solvers/ConjugateGradient.md)         | O(n^2m + mn)                           | Huge                                        |
| [GaussNewton](Solvers/GaussNewton.md)                     | O(n^3 + 2(n^2m) + mn)                  | O(n^2m)                                     |
| [LevenbergMarquardt](Solvers/LevenbergMarquardt)          | O(n^3 + 2(n^2m) + 2(mn))               | O(n^2m)                                     |
| [IterativelyReweighted](Solvers/IterativelyReweighted.md) | O(mn)                                  | O(n^2m)                                     |
| [GreedyCoordinate](Solvers/GreedyCoordinate.md)           | O(mn)                                  | O(n^2m)                                     |
| [RandomCoordinate](Solvers/RandomCoordinate.md)           | O(mn)                                  | O(n^2)                                      |

## Convergence Speed And Convergence Cost Save Analysis (When Cached)

| Number Of Iterations | Gradient Cost                  | GaussNewton Cost            | Gradient Converged? | GaussNewton Converged? | Accumulated Gradient's Gain Balance                       |
|----------------------|--------------------------------|-----------------------------|---------------------|------------------------|-----------------------------------------------------------|
| 1                    | O(n^2m + mn)                   | O(n^3 + 2(n^2m) + 2(mn))    | No                  | No                     | +O(n^3 + 2(n^2m) + mn)                                    |
| 20                   | O(mn)                          | O(mn)                       | No                  | Yes                    | +O(n^3 + 2(n^2m) + mn) - 19 * -O(mn) (Convergence Waste)  |
| 100                  | O(mn)                          | O(mn                        | No                  | Yes                    | +O(n^3 + 2(n^2m) + mn) - 99 * -O(mn) (Convergence Waste)  |
| 1000                 | O(mn)                          | O(mn                        | No                  | Yes                    | +O(n^3 + 2(n^2m) + mn) - 999 * -O(mn) (Convergence Waste) |

### Potential Solver Additions

* Secant method

* Brent's method

* Bisection method (Low priority; slow convergence and slow to compute.) 
