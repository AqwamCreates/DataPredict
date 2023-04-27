# [API Reference](../../API.md) - [Models](../Models.md) - KMedoids

KMedoids is an unsupervised machine learning model that assigns data points to clusters by selecting representative points, called medoids, as cluster centers. It then predicts the cluster membership of new data points based on their distances to the medoids.

## Constructors

### new()

Create new model object. If any of the arguments are not given, default argument values for that argument will be used.

```
KMedoids.new(maxNumberOfIterations: integer, numberOfClusters: integer, distanceFunction: string, targetCost: number, setTheCentroidsDistanceFarthest: boolean): ModelObject
```

#### Parameters:

* maxNumberOfIterations: How many times should the model needed to be trained.

* numberOfClusters: Number of clusters for model to train and predict on.

* distanceFunction: The function that the model will use to train. distanceFunction available are “euclidean” and “manhattan“.

* targetCost: The cost at which the model stops training.

* setTheCentroidsDistanceFarthest: Set whether or not the model to create centroids that are furthest from each other.

#### Returns:

* ModelObject: The generated model object.

## Functions

### setParameters()

Set model's parameters. When any of the arguments are not given, previous argument values for that argument will be used.

```
KMedoids:setParameters(maxNumberOfIterations: integer, numberOfClusters: integer, distanceFunction: string, targetCost: number, setTheCentroidsDistanceFarthest: boolean): ModelObject
```

#### Parameters:

* maxNumberOfIterations: How many times should the model needed to be trained.

* numberOfClusters: Number of clusters for model to train and predict on.

* distanceFunction: The function that the model will use to train. distanceFunction available are “euclidean” and “manhattan“

* targetCost: The cost at which the model stops training.

* setTheCentroidsDistanceFarthest: Set whether or not the model to create centroids that are furthest from each other.

### train()

Train the model.

```
KMedoids:train(featureMatrix: Matrix)
```

#### Parameters:

* featureMatrix: Matrix containing all data.

#### Returns:

* costArray: An array containing cost values.

### predict()

Predict which cluster does it belong to for a given data.

```
KMedoids:predict(featureMatrix: Matrix): integer, number
```

#### Parameters:

* featureMatrix: Matrix containing data.

#### Returns:

* clusterNumber: The cluster which the data belongs to.

* shortestDistance: The distance between the datapoint and the center of the cluster (centroids).

## Inherited From

* [BaseModel](BaseModel.md)
