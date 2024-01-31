# [API Reference](../../API.md) - [Models](../Models.md) - DensityBasedSpatialClusteringOfApplicationsWithNoise (DBSCAN)

DBSCAN is an unsupervised machine learning model that clusters data points based on their spatial density and proximity to each other, using epsilon distance and minimum number of points required to form a cluster.

## Stored Model Parameters

Contains a table.  

* ModelParameters[1]: Contains previously stored feature matrix.

* ModelParameters[2][k][p]: Contains a table of integers table, where k is the cluster number and p is the index of a single feature vector from previously stored feature matrix.

## Constructors

### new()

Create new model object. If any of the arguments are nil, default argument values for that argument will be used.

```
DensityBasedSpatialClusteringOfApplicationsWithNoise.new(epsilon: number, minimumNumberOfPoints: integer, distanceFunction: string): ModelObject
```

#### Parameters:

* epsilon: The maximum distance between two data points for them to be considered as part of the same cluster.

* minimumNumberOfPoints: Minimum number of data points required to form a cluster.

* distanceFunction: The function that the model will use to train. distanceFunction available are “Euclidean” and “Manhattan“.

#### Returns:

* ModelObject: The generated model object.

## Functions

### setParameters()

Set model's parameters. When any of the arguments are nil, previous argument values for that argument will be used.

```
DensityBasedSpatialClusteringOfApplicationsWithNoise:setParameters(epsilon: number, minimumNumberOfPoints: integer, distanceFunction: string)
```

#### Parameters:

* epsilon: The maximum distance between two data points for them to be considered as part of the same cluster.

* minimumNumberOfPoints: Minimum number of data points required to form a cluster.

* distanceFunction: The function that the model will use to train. distanceFunction available are “Euclidean” and “Manhattan“.

### train()

Train the model.

```
DensityBasedSpatialClusteringOfApplicationsWithNoise:train(featureMatrix: Matrix)
```

#### Parameters:

* featureMatrix: Matrix containing all data.

#### Returns:

* costArray: An array containing cost values.

### predict()

Predict which cluster does it belong to for a given data.

```
DensityBasedSpatialClusteringOfApplicationsWithNoise:predict(featureMatrix: Matrix): Matrix, Matrix
```

#### Parameters:

* featureMatrix: Matrix containing data.

#### Returns:

* clusterNumberVector: The cluster which the data belongs to.

* shortestDistanceVector: The distance between the datapoint and the center of the cluster (centroids).

## Inherited From

* [BaseModel](BaseModel.md)
