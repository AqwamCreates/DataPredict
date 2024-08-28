# [API Reference](../../API.md) - [Models](../Models.md) - MeanShift

MeanShift is a an unsupervised machine learning model that finds cluster centers by moving points towards higher density regions.

## Stored Model Parameters

Contains a matrix.  

* ModelParameters[I][J]: Value of matrix at row I and column J. The rows represent the clusters. The columns represent the features.

## Constructors

### new()

Create new model object. If any of the arguments are nil, default argument values for that argument will be used.

```
MeanShift.new(maximumNumberOfIterations: integer, bandwidth: number, bandwidthStep: integer, distanceFunction: string): ModelObject
```

#### Parameters:

* maximumNumberOfIterations: How many times should the model needed to be trained.

* bandwidth: The size of the area around each data point.

* bandwidthStep: The size of the update for each clusters.

* distanceFunction: The function that the model will use to train. Available options are:

  * Euclidean
 
  * Manhattan

  * Cosine

#### Returns:

* ModelObject: The generated model object.

## Functions

### setParameters()

Set model's parameters. When any of the arguments are nil, previous argument values for that argument will be used.

```
MeanShift:setParameters(maximumNumberOfIterations: integer, bandwidth: number, bandwidthStep: integer, distanceFunction: string)
```

#### Parameters:

* maximumNumberOfIterations: How many times should the model needed to be trained.

* bandwidth: The size of the area around each data point.

* bandwidthStep: The size of the update for each clusters.

* distanceFunction: The function that the model will use to train. Available options are:

  * Euclidean
 
  * Manhattan

  * Cosine

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

* [BaseModel](BaseModel.md)
