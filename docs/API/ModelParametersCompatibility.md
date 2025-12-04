# Model Parameters Compatibility

This documentation shows you on which algorithms can be switched with others.

However, it assumes that you will ignore the difference in hyperparameters and data distribution that the model trained on.

## Full Compatibility

The algorithms under the same group can have its model parameters swapped without additional changes.

### Closed-Form Linear Regression

* BayesianLinearRegression
* BayesianQuantileLinearRegression

### Gradient-Based

* LinearRegression
* PoissonRegression
* QuantileLinearRegression
* SupportVectorRegressionGradientVariant
* PassiveAggressiveRegressor

* LogisticRegression
* ProbitRegression
* SupportVectorMachineGradientVariant
* PassiveAggressiveClassifier
* OneClassPassiveAggressiveClassifier

### Naive Bayes

* All except for CategoricalNaiveBayes

### K-Nearest Neigbours

* All

### Mean-Based With Data Point Number

* NearestCentroids
* KMeans

### Deep Reinforcement Learning (Single)

* All

### Deep Reinforcement Learning (Actor-Critic)

* All

### Tabular Reinforcement Learning

* All

### Generative Adversarial Network

* All

### Outlier Detection

* All

## Partial Compatibility

The algorithms under the same group can have its model parameters swapped with some changes to the model parameters. Refer to the models' API reference for more information.

### Statistical-Based Clustering

* MeanShift
* ExpectedMaximization
* FuzzyCMeans

### Kalman Filters

* All
