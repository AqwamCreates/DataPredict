# [API Reference](../../API.md) - [Models](../Models.md) - OrderingPointsToIdentifyClusteringStructure (OPTICS)

OPTICS (Ordering Points To Identify the Clustering Structure) is an unsupervised machine learning model that analyzes data point density and spatial proximity to uncover cluster structures across varying density levels.

Unlike DBSCAN, OPTICS produces an ordering of points and their reachability distances, allowing more flexible extraction of clusters without fixing a single epsilon threshold.

## Stored Model Parameters

Contains a table.  

* ModelParameters[1]: Contains previously stored feature matrix.

* ModelParameters[2]: orderedPointArray.

* ModelParameters[3]: reachabilityDistanceArray.

* ModelParameters[2][k][p]: Contains a table of integers table, where k is the cluster number and p is the index of a single feature vector from previously stored feature matrix.

## Constructors

### new()

Create new model object. If any of the arguments are nil, default argument values for that argument will be used.

```
OrderingPointsToIdentifyClusteringStructure.new(epsilon: number, minimumNumberOfPoints: integer, distanceFunction: string): ModelObject
```

#### Parameters:

* epsilon: The maximum distance between two data points for them to be considered as part of the same cluster.

* epsilonPrime: Used for creating clusters (which is not a part of original OPTICS algorithm). Controls how fine-grained clusters are. The higher it is, the more you allow more points to belong together. Setting epsilon = epsilonPrime will cause it to act like DBSCAN algorithm.

* minimumNumberOfPoints: Minimum number of data points required to form a cluster.

* distanceFunction: The function that the model will use to train. distanceFunction available are:
  
  *  Euclidean
    
  *  Manhattan
 
  *  Cosine

#### Returns:

* ModelObject: The generated model object.

## Functions

### train()

Train the model.

```
OrderingPointsToIdentifyClusteringStructure:train(featureMatrix: Matrix)
```

#### Parameters:

* featureMatrix: Matrix containing all data.

#### Returns:

* costArray: An array containing cost values.

### predict()

Predict which cluster does it belong to for a given data.

```
OrderingPointsToIdentifyClusteringStructure:predict(featureMatrix: Matrix): Matrix, Matrix
```

#### Parameters:

* featureMatrix: Matrix containing data.

#### Returns:

* clusterNumberVector: The cluster which the data belongs to.

* shortestDistanceVector: The distance between the datapoint and the center of the cluster (centroids).

## Inherited From

* [IterativeMethodBaseModel](IterativeMethodBaseModel.md)

