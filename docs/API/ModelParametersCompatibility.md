# Model Parameters Compatibility

This documentation shows you on which algorithms can be switched 

## Full Compatibility

The algorithms under the same group can have its model parameters swapped without additional changes.

### Gradient-Based

* LinearRegression
* PoissonRegression
* QuantileLinearRegression
* SupportVectorRegressionGradientVariant
* PassiveAggressiveRegression

* LogisticRegression
* SupportVectorMachineGradientVariant
* PassiveAggressiveClassifier
* OneClassPassiveAggressiveClassifier

### Naive Bayes

* All except for CategoricalNaiveBayes

### K-Nearest Neigbours

* All

### Mean-Based With Data Points Number

* NearestCentroids
* KMeans

### Deep Reinforcement Learning (Single)

* All

### Deep Reinforcement Learning (Actor-Critic)

* All

### Tabular Reinforcement Learning

* All
