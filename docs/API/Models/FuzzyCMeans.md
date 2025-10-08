# [API Reference](../../API.md) - [Models](../Models.md) - FuzzyCMeans

FuzzyCMeans is an unsupervised machine learning model that predicts which cluster that the input belongs to using weighted distance.

## Stored Model Parameters

Contains a matrix.  

* ModelParameters[1][I][J]: Value of matrix at row I and column J. The rows represent the clusters. The columns represent the features.

* ModelParameters[2][I][1]: Value of matrix at row I. The rows represent the number of data points associated with each clusters.

## Constructors

### new()

Create new model object. If any of the arguments are nil, default argument values for that argument will be used.

```
FuzzyCMeans.new(maximumNumberOfIterations: integer, numberOfClusters: integer, fuzziness: number, distanceFunction: string, mode: string, setInitialClustersOnDataPoints: boolean, setTheCentroidsDistanceFarthest: boolean): ModelObject
```

#### Parameters:

* maximumNumberOfIterations: How many times should the model needed to be trained.

* numberOfClusters: Number of clusters for model to train and predict on.

* fuzziness: Controls how "fuzzy" the cluster assignments are.

* distanceFunction: The function that the model will use to train. Available options are:
  
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

### train()

Train the model.

```
FuzzyCMeans:train(featureMatrix: Matrix)
```

#### Parameters:

* featureMatrix: Matrix containing all data.

#### Returns:

* costArray: An array containing cost values.

### predict()

Predict which clusters does it belong to for a given data.

```
KMeans:predict(featureMatrix: Matrix, returnMode: string/boolean/nil): Matrix, Matrix -OR- Matrix
```

#### Parameters:

* featureMatrix: Matrix containing data.

* returnMode: Set whether or not to return distance matrix instead of clusterNumberVector and closestDistanceVector. Available options are:

  * nil (Default)

  * Distance (Can be set using "true" boolean)
 
  * Membership

#### Returns:

* clusterNumberVector: A vector containing which cluster that the data belongs to.

* closestDistanceVector: A vector containing the closest distance between the datapoint and the center of the cluster (centroids).

-OR-

* distanceMatrix: A matrix containing data-cluster pair distance.

## Inherited From

* [IterativeMethodBaseModel](IterativeMethodBaseModel.md)

## References

* [The fuzzy c-means clustering algorithm](https://www.sciencedirect.com/science/article/abs/pii/0098300484900207)
