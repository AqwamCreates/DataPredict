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
OrderingPointsToIdentifyClusteringStructure.new(epsilon: number, epsilonPrime: number, minimumNumberOfPoints: integer, distanceFunction: string): ModelObject
```

#### Parameters:

* epsilon: The maximum neighborhood radius used to determine which points are density-reachable during the OPTICS process.

* epsilonPrime: Used during cluster extraction, not part of the original OPTICS algorithm. [Default: epsilon * 0.5]

  * Controls how fine-grained clusters are.

  * A higher value merges nearby regions into larger clusters.

  * A lower value produces more, smaller, and denser clusters.

  * Setting epsilonPrime = epsilon makes the algorithm behave like DBSCAN.

* minimumNumberOfPoints: The minimum number of points required to form a dense region (core point condition).

* distanceFunction: The function that the model will use to train. Available options are:
  
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

