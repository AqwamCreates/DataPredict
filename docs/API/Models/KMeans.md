# [API Reference](../../API.md) - [Models](../Models.md) - KMeans

KMeans is an unsupervised machine learning model that predicts which cluster that the input belongs to using distance.

## Stored Model Parameters

Contains a matrix.  

* ModelParameters[I][J]: Value of matrix at row I and column J. The rows represent the clusters. The columns represent the features.

## Constructors

### new()

Create new model object. If any of the arguments are nil, default argument values for that argument will be used.

```
KMeans.new(maxNumberOfIterations: integer, learningRate: number, numberOfClusters: integer, distanceFunction: string, targetCost: number, setInitialClustersOnDataPoints: boolean, setTheCentroidsDistanceFarthest: boolean, stopWhenModelParametersDoesNotChange:boolean): ModelObject
```

#### Parameters:

* maxNumberOfIterations: How many times should the model needed to be trained.

* learningRate: The speed at which the model learns. Recommended that the value is set between 0 to 1.

* numberOfClusters: Number of clusters for model to train and predict on.

* distanceFunction: The function that the model will use to train. distanceFunction available are “euclidean” and “manhattan“.

* targetCost: The cost at which the model stops training.

* setInitialClustersOnDataPoints: Set whether or not the model to create centroids on any data points.

* setTheCentroidsDistanceFarthest: Set whether or not the model to create centroids that are furthest from each other.

* stopWhenModelParametersDoesNotChange: Stop the training if the model parameters does not change from the previous iteration.

#### Returns:

* ModelObject: The generated model object.

## Functions

### setParameters()

Set model's parameters. When any of the arguments are nil, previous argument values for that argument will be used.

```
KMeans:setParameters(maxNumberOfIterations: integer, learningRate: number, numberOfClusters: integer, distanceFunction: string, targetCost: number, setInitialClustersOnDataPoints: boolean, setTheCentroidsDistanceFarthest: boolean, stopWhenModelParametersDoesNotChange:boolean)
```

#### Parameters:

* maxNumberOfIterations: How many times should the model needed to be trained.

* learningRate: The speed at which the model learns. Recommended that the value is set between (0 to 1).

* numberOfClusters: Number of clusters for model to train and predict on.

* distanceFunction: The function that the model will use to train. distanceFunction available are “euclidean” and “manhattan“

* targetCost: The cost at which the model stops training.

* setInitialClustersOnDataPoints: Set whether or not the model to create centroids on any data points.

* setTheCentroidsDistanceFarthest: Set whether or not the model to create centroids that are furthest from each other.

* stopWhenModelParametersDoesNotChange: Stop the training if the model parameters does not change from the previous iteration.

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

Predict which cluster does it belong to for a given data.

```
KMeans:predict(featureMatrix: Matrix): integer, number
```

#### Parameters:

* featureMatrix: Matrix containing data.

#### Returns:

* clusterNumber: The cluster which the data belongs to.

* shortestDistance: The distance between the datapoint and the center of the cluster (centroids).

## Inherited From

* [BaseModel](BaseModel.md)
