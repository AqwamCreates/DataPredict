# [API Reference](../API.md) - Solvers

> M: Number Of Data N: Number Of Features O: Number Of Outputs

| Solver                                | Convergence Speed | Computational Efficiency (Non-Cached)    | Computational Efficiency (Cached)* |
|---------------------------------------|-------------------|------------------------------------------|------------------------------------|
| [Gradient](Solvers/Gradient.md)       | Low               | High, O((M x N) + (M x O))               | Very High, O(M x O)                |
| [GaussNewton](Solvers/GaussNewton.md) | Medium-High       | Medium, O(2(M^2 x N) + O(M^3) + (M x O)) | Very High, O(M x O)                |

\* Computational efficiency due to cache is only applicable to linear models like LinearRegression, SupportVectorRegression and SupportVectorMachine.
