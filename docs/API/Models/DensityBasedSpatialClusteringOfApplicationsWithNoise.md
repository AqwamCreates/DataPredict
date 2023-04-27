# [API Reference](../../API.md) - [Models](../Models.md) - DensityBasedSpatialClusteringOfApplicationsWithNoise

DBSCAN is an unsupervised machine learning model that clusters data points based on their spatial density and proximity to each other, using epsilon distance and minimum number of points required to form a cluster.

## Constructors

### new()

Create new model object. If any of the arguments are not given, default argument values for that argument will be used.

```
DensityBasedSpatialClusteringOfApplicationsWithNoise.new(epsilon: number, minimumNumberOfPoints: integer, distanceFunction: string, targetCost: number): ModelObject
```

#### Parameters:

* epsilon: The maximum distance between two data points for them to be considered as part of the same cluster.

* minimumNumberOfPoints: Minimum number of data points required to form a cluster.

* distanceFunction: The function that the model will use to train. distanceFunction available are “euclidean” and “manhattan“.

* targetCost: The cost at which the model stops training.

#### Returns:

* ModelObject: The generated model object.

## Functions

### setParameters()

Set model's parameters. When any of the arguments are not given, previous argument values for that argument will be used.

```
DensityBasedSpatialClusteringOfApplicationsWithNoise:setParameters(epsilon: number, minimumNumberOfPoints: integer, distanceFunction: string, targetCost: number): ModelObject
```

#### Parameters:

* epsilon: The maximum distance between two data points for them to be considered as part of the same cluster.

* minimumNumberOfPoints: Minimum number of data points required to form a cluster.

* distanceFunction: The function that the model will use to train. distanceFunction available are “euclidean” and “manhattan“.

* targetCost: The cost at which the model stops training.

### canAppendPreviousFeatureMatrix()

Set the option if the previous feature matrix should append with the new feature matrix during training.

```
DensityBasedSpatialClusteringOfApplicationsWithNoise:canAppendPreviousFeatureMatrix(option: boolean)
```

#### Parameters:

* option: the boolean value to set whether or not the feature matrices can be appended.

### clearPreviousFeatureMatrix()

Clears previous feature matrix that was stored in the model.

```
DensityBasedSpatialClusteringOfApplicationsWithNoise:clearPreviousFeatureMatrix()
```

### getPreviousFeatureMatrix()

Get the previous feature matrix that was stored in the model.

```
DensityBasedSpatialClusteringOfApplicationsWithNoise:getPreviousFeatureMatrix(): matrix
```

### returns:

* featureMatrix: a matrix containing data that was previously stored in the model

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
DensityBasedSpatialClusteringOfApplicationsWithNoise:predict(featureMatrix: Matrix): integer, number
```

#### Parameters:

* featureMatrix: Matrix containing data.

#### Returns:

* clusterNumber: The cluster which the data belongs to.

* shortestDistance: The distance between the datapoint and the center of the cluster (centroids).

## Inherited From

* [BaseModel](BaseModel.md)
