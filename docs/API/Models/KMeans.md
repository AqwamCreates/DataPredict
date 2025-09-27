# [API Reference](../../API.md) - [Models](../Models.md) - KMeans

KMeans is an unsupervised machine learning model that predicts which cluster that the input belongs to using distance.

## Stored Model Parameters

Contains a matrix.  

* ModelParameters[1][I][J]: Value of matrix at row I and column J. The rows represent the clusters. The columns represent the features.

* ModelParameters[2][I][1]: Value of matrix at row I. The rows represent the number of data points associated with each clusters.

## Constructors

### new()

Create new model object. If any of the arguments are nil, default argument values for that argument will be used.

```
KMeans.new(maximumNumberOfIterations: integer, numberOfClusters: integer, distanceFunction: string, mode: string, setInitialClustersOnDataPoints: boolean, setTheCentroidsDistanceFarthest: boolean): ModelObject
```

#### Parameters:

* maximumNumberOfIterations: How many times should the model needed to be trained.

* numberOfClusters: Number of clusters for model to train and predict on.

* distanceFunction: The function that the model will use to train. distanceFunction available are:
  
  *  Euclidean (Default)
    
  *  Manhattan
 
  *  Cosine

* mode: The mode that the model will use to train its model parameters:

  * Hybrid (Default)
 
  * Batch
 
  * Sequential

* setInitialClustersOnDataPoints: Set whether or not the model to create centroids on any data points.

* setTheCentroidsDistanceFarthest: Set whether or not the model to create centroids that are furthest from each other. This can only take effect if the "setInitialClustersOnDataPoints" is set to true.

#### Returns:

* ModelObject: The generated model object.

## Functions

### setParameters()

Set model's parameters. When any of the arguments are nil, previous argument values for that argument will be used.

```
KMeans:setParameters(maximumNumberOfIterations: integer, numberOfClusters: integer, distanceFunction: string, mode: string, setInitialClustersOnDataPoints: boolean, setTheCentroidsDistanceFarthest: boolean)
```

#### Parameters:

* maximumNumberOfIterations: How many times should the model needed to be trained.

* numberOfClusters: Number of clusters for model to train and predict on.

* distanceFunction: The function that the model will use to train. distanceFunction available are:
  
  *  Euclidean
    
  *  Manhattan
 
  *  Cosine

* mode: The mode that the model will use to train its model parameters:

  * Hybrid (Default)
 
  * Batch
 
  * Sequential

* setInitialClustersOnDataPoints: Set whether or not the model to create centroids on any data points.

* setTheCentroidsDistanceFarthest: Set whether or not the model to create centroids that are furthest from each other. This can only take effect if the "setInitialClustersOnDataPoints" is set to true.

### train()

Train the model.

```
KMeans:train(featureMatrix: Matrix)
```

#### Parameters:

* featureMatrix: Matrix containing all data.

#### Returns:

* costArray: An array containing cost values.

### predict()

Predict which clusters does it belong to for a given data.

```
KMeans:predict(featureMatrix: Matrix): integer, number
```

#### Parameters:

* featureMatrix: Matrix containing data.

#### Returns:

* clusterNumber: The cluster which the data belongs to.

* shortestDistance: The distance between the datapoint and the center of the cluster (centroids).

## Inherited From

* [IterativeMethodBaseModel](IterativeMethodBaseModel.md)
