# [API Reference](../../API.md) - [Models](../Models.md) - AgglomerativeHierarchical

AgglomerativeHierarchical clustering groups similar data points into clusters based on distance, in a bottom-up approach.

## Stored Model Parameters

Contains a matrix.  

* ModelParameters[I][J]: Value of matrix at row I and column J. The rows represent the clusters. The columns represent the features.

## Constructors

### new()

Create new model object. If any of the arguments are nil default argument values for that argument will be used.

```
AgglomerativeHierarchical.new(numberOfClusters: integer, distanceFunction: string, linkageFunction: string): ModelObject
```

#### Parameters:

* numberOfClusters: Number of clusters for model to train and predict on.

* distanceFunction: The function that the model will use to train. Available options are:
  
  * Euclidean (Default)
 
  * Manhattan

* linkageFunction: The function to determine how clusters are merged together. Available options are:

  * Minimum (Default)

  * Maximum

  * GroupAverage
   
  * Ward

#### Returns:

* ModelObject: The generated model object.

## Functions

### setParameters()

Set model's parameters. When any of the arguments are nil previous argument values for that argument will be used.

```
AgglomerativeHierarchical:setParameters(numberOfClusters: integer, distanceFunction: string, linkageFunction: string)
```

#### Parameters:

* numberOfClusters: Number of clusters for model to train and predict on.

* distanceFunction: The function that the model will use to train. Available options are:

  * Euclidean
 
  * Manhattan

* linkageFunction: The function to determine how clusters are merged together. Available options are:

  * Minimum

  * Maximum

  * GroupAverage
   
  * Ward

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
