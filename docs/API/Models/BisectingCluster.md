# [API Reference](../../API.md) - [Models](../Models.md) - BisectingCluster

BisectingCluster is a generalized wrapper for adding bisecting clustering ability to other clusters.

## Stored Model Parameters

Contains a matrix.  

* ModelParameters[I][J]: Value of matrix at row I and column J. The rows represent the clusters. The columns represent the features.

## Constructors

### new()

Create new model object. If any of the arguments are nil, default argument values for that argument will be used.

```
BisectingCluster.new(Model: Model, numberOfClusters: integer, distanceFunction: string, splitCriterion: string): ModelObject
```

#### Parameters:

* Model: The clustering model to use inside the BisectingCluster object.

* numberOfClusters: Number of clusters for model to train and predict on.

* distanceFunction: The function that the model will use to train. distanceFunction available are:
  
  *  Euclidean (Default)
    
  *  Manhattan
 
  *  Cosine

* splitCriterion: Controls which clusters should be split. Available options are:

  * LargestCluster (Default)
 
  * SumOfSquaredError

#### Returns:

* ModelObject: The generated model object.

## Functions

### train()

Train the model.

```
BisectingCluster:train(featureMatrix: matrix): {number}
```

#### Parameters:

* featureMatrix: Matrix containing all data.

#### Returns:

* costArray: An array containing cost values.

### predict()

Predict which clusters does it belong to for a given data.

```
BisectingCluster:predict(featureMatrix: matrix, returnOriginalOutput: boolean): matrix, matrix -OR- matrix
```

#### Parameters:

* featureMatrix: Matrix containing data.

* returnOriginalOutput: Set whether or not to return distance matrix instead of clusterNumberVector and closestDistanceVector. 

#### Returns:

* clusterNumberVector: A vector containing which cluster that the data belongs to.

* closestDistanceVector: A vector containing the closest distance between the datapoint and the center of the cluster (centroids).

-OR-

* distanceMatrix: A matrix containing data-cluster pair distance.

## Inherited From

* [IterativeMethodBaseModel](IterativeMethodBaseModel.md)

## References

* [Sequential k-Means Clustering](https://www.cs.princeton.edu/courses/archive/fall08/cos436/Duda/C/sk_means.htm)
