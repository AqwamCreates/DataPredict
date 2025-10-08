# [API Reference](../../API.md) - [Models](../Models.md) - MeanShift

MeanShift is a an unsupervised machine learning model that finds cluster centers by moving points towards higher density regions.

## Stored Model Parameters

Contains a matrix.  

* ModelParameters[I][J]: Value of matrix at row I and column J. The rows represent the clusters. The columns represent the features.

## Constructors

### new()

Create new model object. If any of the arguments are nil, default argument values for that argument will be used.

```
MeanShift.new(maximumNumberOfIterations: integer, bandwidth: number, mode: string, distanceFunction: string, kernelFunction: string, kernelParameters: table): ModelObject
```

#### Parameters:

* maximumNumberOfIterations: How many times should the model needed to be trained.

* bandwidth: The maximum distance in order to merge with other centroids (center of clusters).

* mode: Controls the mode of the model. Available options are:

  * Hybrid (Default)
 
  * Online
 
  * Offline

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
Meanshift:predict(featureMatrix: Matrix, returnOriginalOutput: boolean): Matrix, Matrix -OR- Matrix
```

#### Parameters:

* featureMatrix: Matrix containing data.

* returnOriginalOutput: Set whether or not to return distance matrix instead of clusterNumberVector and closestDistanceVector. 

#### Returns:

* clusterNumberVector: A vector containing which cluster that the data belongs to.

* closestDistanceVector: A vector containing the closest distance between the datapoint and the center of the cluster (centroids).

-OR-

* distanceMatrix: A matrix containing data-cluster pair distance.

## Inherited From

* [IterativeMethodBaseModel](IterativeMethodBaseModel.md)
