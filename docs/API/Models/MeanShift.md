# [API Reference](../../API.md) - [Models](../Models.md) - MeanShift

MeanShift is a an unsupervised machine learning model that finds cluster centers by moving points towards higher density regions.

## Stored Model Parameters

Contains a matrix.  

* ModelParameters[I][J]: Value of matrix at row I and column J. The rows represent the clusters. The columns represent the features.

## Constructors

### new()

Create new model object. If any of the arguments are nil, default argument values for that argument will be used.

```
MeanShift.new(maximumNumberOfIterations: integer, bandwidth: number, distanceFunction: string, kernelFunction: string, kernelParameters: table): ModelObject
```

#### Parameters:

* maximumNumberOfIterations: How many times should the model needed to be trained.

* bandwidth: The size of the area around each data point.

* distanceFunction: The function to calculate the distance between the data points and the centroids. Available options are:

  * Euclidean (Default)
 
  * Manhattan

  * Cosine

* kernelFunction: The function used to kernelize the distance between the data points and the centroids. Available options are:

 * Gaussian (Default)

 * Linear

#### Returns:

* ModelObject: The generated model object.

## Functions

### setParameters()

Set model's parameters. When any of the arguments are nil, previous argument values for that argument will be used.

```
MeanShift:setParameters(maximumNumberOfIterations: integer, bandwidth: number, distanceFunction: string, kernelFunction: string, kernelParameters: table)
```

#### Parameters:

* maximumNumberOfIterations: How many times should the model needed to be trained.

* bandwidth: The size of the area around each data point.

* bandwidthStep: The size of the update for each clusters.

* distanceFunction: The function to calculate the distance between the data points and the centroids. Available options are:

  * Euclidean (Default)
 
  * Manhattan

  * Cosine

* kernelFunction: The function used to kernelize the distance between the data points and the centroids. Available options are:

 * Gaussian (Default)

 * Linear

### train()

Train the model.

```
MeanShift:train(featureMatrix: Matrix)
```

#### Parameters:

* featureMatrix: Matrix containing all data.

#### Returns:

* costArray: An array containing cost values.

### predict()

Predict which clusters does it belong to for a given data.

```
MeanShift:predict(featureMatrix: Matrix): integer, number
```

#### Parameters:

* featureMatrix: Matrix containing data.

#### Returns:

* clusterNumber: The cluster which the data belongs to.

* shortestDistance: The distance between the datapoint and the center of the cluster (centroids).

## Inherited From

* [IterativeMethodBaseModel](IterativeMethodBaseModel.md)
