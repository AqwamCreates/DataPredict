# [API Reference](../../API.md) - [Models](../Models.md) - AgglomerativeHierarchical

AgglomerativeHierarchical is an unsupervised machine learning model that predicts which cluster that the input belongs to using distance.

## Constructors

### new()

Create new model object. If any of the arguments are not given, default argument values for that argument will be used.

```
AgglomerativeHierarchical.new(maxNumberOfIterations: integer, numberOfClusters: integer, distanceFunction: string, highestCost: number, lowestCost:number, stopWhenModelParametersDoesNotChange: boolean): ModelObject
```

#### Parameters:

* maxNumberOfIterations: How many times should the model needed to be trained.

* numberOfClusters: Number of clusters for model to train and predict on.

* distanceFunction: The function that the model will use to train. distanceFunction available are “euclidean” and “manhattan“.

* highestCost: The maximum cost where the training will stop when it is higher than this value.

* lowestCost: The minimum cost where the training will stop when it is lower than this value.

* targetCost: The cost at which the model stops training.

* stopWhenModelParametersDoesNotChange: Stop the training if the model parameters does not change from the previous iteration.

#### Returns:

* ModelObject: The generated model object.

## Functions

### setParameters()

Set model's parameters. When any of the arguments are not given, previous argument values for that argument will be used.

```
AgglomerativeHierarchical:setParameters(maxNumberOfIterations: integer, numberOfClusters: integer, distanceFunction: string, highestCost: number, lowestCost:number, stopWhenModelParametersDoesNotChange: boolean): ModelObject
```

#### Parameters:

* maxNumberOfIterations: How many times should the model needed to be trained.

* numberOfClusters: Number of clusters for model to train and predict on.

* distanceFunction: The function that the model will use to train. distanceFunction available are “euclidean” and “manhattan“.

* highestCost: The maximum cost where the training will stop when it is higher than this value.

* lowestCost: The minimum cost where the training will stop when it is lower than this value.

* targetCost: The cost at which the model stops training.

### train()

Train the model.

```
AgglomerativeHierarchical:train(featureMatrix: Matrix)
```

#### Parameters:

* featureMatrix: Matrix containing all data.

#### Returns:

* costArray: An array containing cost values.

### predict()

Predict which cluster does it belong to for a given data.

```
AgglomerativeHierarchical:predict(featureMatrix: Matrix): integer, number
```

#### Parameters:

* featureMatrix: Matrix containing data.

#### Returns:

* clusterNumber: The cluster which the data belongs to.

* shortestDistance: The distance between the datapoint and the center of the cluster (centroids).

## Inherited From

* [BaseModel](BaseModel.md)
