# [API Reference](../../API.md) - [Models](../Models.md) - MeanShift

MeanShift is a clustering algorithm that finds cluster centers by moving points towards higher density regions.

## Stored Model Parameters

Contains a matrix.  

* ModelParameters[I][J]: Value of matrix at row I and column J. The rows represent the clusters. The columns represent the features.

## Constructors

### new()

Create new model object. If any of the arguments are nil, default argument values for that argument will be used.

```
MeanShift.new(maxNumberOfIterations: integer, bandwidth: number, bandwidthStep: integer, distanceFunction: string, highestCost: number, lowestCost: number, stopWhenModelParametersDoesNotChange: boolean): ModelObject
```

#### Parameters:

* maxNumberOfIterations: How many times should the model needed to be trained.

* bandwidth: The speed at which the model learns. Recommended that the value is set between 0 to 1.

* bandwidthStep: Number of clusters for model to train and predict on.

* distanceFunction: The function that the model will use to train. distanceFunction available are “Euclidean” and “Manhattan“.

* highestCost: The highest cost at which the model stops training.

* lowestCost: The lowest cost at which the model stops training.

* stopWhenModelParametersDoesNotChange: Stop the training if the model parameters does not change from the previous iteration.

#### Returns:

* ModelObject: The generated model object.

## Functions

### setParameters()

Set model's parameters. When any of the arguments are nil, previous argument values for that argument will be used.

```
MeanShift:setParameters(maxNumberOfIterations: integer, bandwidth: number, bandwidthStep: integer, distanceFunction: string, highestCost: number, lowestCost: number, stopWhenModelParametersDoesNotChange: boolean)
```

#### Parameters:

* maxNumberOfIterations: How many times should the model needed to be trained.

* bandwidth: The speed at which the model learns. Recommended that the value is set between 0 to 1.

* bandwidthStep: Number of clusters for model to train and predict on.

* distanceFunction: The function that the model will use to train. distanceFunction available are “Euclidean” and “Manhattan“.

* highestCost: The highest cost at which the model stops training.

* lowestCost: The lowest cost at which the model stops training.

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

* [BaseModel](BaseModel.md)
