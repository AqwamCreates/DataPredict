# [API Reference](../../API.md) - [Models](../Models.md) - ExpectationMaximization (EM)

ExpectationMaximization is an unsupervised machine learning model that estimates the probability distribution of a dataset and assigns each data point to a cluster based on its most likely probability.

## Stored Model Parameters

Contains a table of matrices.  

* ModelParameters[1]: piMatrix. The rows represent the clusters.

* ModelParameters[2]: meanMatrix. The rows represent the clusters. The columns represent the features.

* ModelParameters[3]: varianceMatrix. The rows represent the clusters. The columns represent the features.

## Constructors

### new()

Create new model object. If any of the arguments are nil, default argument values for that argument will be used.

```
ExpectationMaximization.new(maximumNumberOfIterations: integer, numberOfClusters: integer, epsilon: number): ModelObject
```
#### Parameters

* maximumNumberOfIterations: The maximum number of iterations.

* numberOfClusters: Number of clusters for model to train and predict on. When using default or set to math.huge(), it will find the best number of clusters using Bayesian information criterion.

* epsilon: The value to ensure that Gaussian calculation doesn't reach infinity.

#### Returns:

* Model: The generated model object.

## Functions

### setParameters()

Set model's parameters. When any of the arguments are nil, previous argument values for that argument will be used. 

```
ExpectationMaximization:setParameters(maximumNumberOfIterations: integer, numberOfClusters: integer, epsilon: number)
```

#### Parameters

* maximumNumberOfIterations: The maximum number of iterations. 

* numberOfClusters: Number of clusters for model to train and predict on. When using default or set to math.huge(), it will find the best number of clusters using Bayesian information criterion.

* epsilon: The value to ensure that Gaussian calculation doesn't reach infinity.

### train()

Train the model.

```
ExpectationMaximization:train(featureMatrix: Matrix)
```

#### Parameters:

* featureMatrix: Matrix containing all data.

#### Returns:

* costArray: An array containing cost values.

### predict()

Predict which cluster does it belong to for a given data.

```
ExpectationMaximization:predict(featureMatrix: Matrix): integer, number
```

#### Parameters:

* featureMatrix: Matrix containing data.

#### Returns:

* clusterNumber: The cluster which the data belongs to.

* highestProbabilityVector: The probability (n x 1) matrix of the datapoint belongs to that particular cluster.

## Inherited From

* [IterativeMethodBaseModel](IterativeMethodBaseModel.md)
