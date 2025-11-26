# Model Parameters Compatibility

This documentation shows you on which algorithms can be switched 

## Full Compatibility

The algorithms under the same group can have its model parameters swapped without additional changes.

### Gradient-Based Models

* LinearRegression
* PoissonRegression
* QuantileLinearRegression
* SupportVectorRegressionGradientVariant
* PassiveAggressiveRegression

* LogisticRegression
* SupportVectorMachineGradientVariant
* PassiveAggressiveClassifier
* OneClassPassiveAggressiveClassifier

### Naive Bayes Models

* All except for CategoricalNaiveBayes

### K-Nearest Neigbours Models

* All

### Mean-Based With Data Points Number Models

* NearestCentroids
* KMeans

### Tabular Reinforcement Learning

* All
