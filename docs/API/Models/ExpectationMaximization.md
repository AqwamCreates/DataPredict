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
ExpectationMaximization.new(maximumNumberOfIterations: integer, numberOfClusters: integer, mode: string, useLogProbabilities: boolean, distanceFunction: string, epsilon: number): ModelObject
```
#### Parameters

* maximumNumberOfIterations: The maximum number of iterations.

* numberOfClusters: Number of clusters for model to train and predict on. When using default or set to math.huge(), it will find the best number of clusters using Bayesian information criterion.

* mode: Controls the mode of the model. Available options are:

  * Hybrid (Default)
 
  * Online
 
  * Offline

* useLogProbabilities: Controls whether or not to convert probabilities using the logarithm function for numerical stability [Default: False].

* distanceFunction: The distance function to be used to initialize the centroids. Available options are:

  * Euclidean (Default)
 
  * Manhattan
 
  * Cosine

* epsilon: The value to ensure that Gaussian calculation doesn't reach infinity.

#### Returns:

* Model: The generated model object.

## Functions

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
ExpectationMaximization:predict(featureMatrix: Matrix, returnOriginalOutput: boolean): Matrix, Matrix -OR- Matrix
```

#### Parameters:

* featureMatrix: Matrix containing data.

* returnOriginalOutput: Set whether or not to return probabilityMatrix matrix instead of clusterNumberVector and closestDistanceVector.

#### Returns:

* clusterNumberVector: A vector containing the cluster that the data belongs to.

* highestProbabilityVector: The probability (n x 1) matrix of the datapoint belongs to that particular cluster.

-OR-



## Inherited From

* [IterativeMethodBaseModel](IterativeMethodBaseModel.md)

## References

* [Online EM for Unsupervised Models](https://cs.stanford.edu/~pliang/papers/online-naacl2009.pdf)

* [On the Global Convergence of (Fast) Incremental Expectation Maximization Methods](https://arxiv.org/pdf/1910.12521)
