# [API Reference](../../API.md) - [Models](../Models.md) - AffinityPropagation

AffinityPropagation is an unsupervised machine learning model that predicts which cluster that the input belongs to using a similarity matrix that measures the similarity between each data point.

## Stored Model Parameters

Contains a table.  

* ModelParameters[1]: Contains previously stored feature matrix.

* ModelParameters[2]: Contains the cluster number array that has a length of m, where m is the number of data from previous feature matrix and its values contains the cluster numbers indicating which of the clusters that each feature vector belongs to.

* ModelParameters[3]: Contains the responsibility matrix of (m x m) size, where m is the number of data.

* ModelParameters[3]: Contains the availability matrix of (m x m) size, where m is the number of data.

## Constructors

### new()

Create new model object. If any of the arguments are nil, default argument values for that argument will be used.

```
AffinityPropagation.new(maxNumberOfIterations: integer, distanceFunction: string, preferenceType: string, damping: number, preferenceValueArray: {number}): ModelObject
```

#### Parameters:

* maxNumberOfIterations: How many times should the model needed to be trained.

* distanceFunction: The distance function to be used. Available options are:

    * Euclidean (Default)
      
    * Manhattan
      
    * Cosine

* preferenceType: Determines how preferences are calculated. Available options are:

   * Median (Default)
 
   * Average
   
   * Minimum
 
   * Maximum
 
   * Precomputed

* damping: A high value leads to fewer changes, while a low value leads to more exploration. The value can be set between 0 and 1.

* preferenceValueArray: An array containing preference values. The index determines the preference value for number of data for that index. Can only be used with "Precomputed" preferenceType.

#### Returns:

* ModelObject: The generated model object.

## Functions

### setParameters()

Set model's parameters. When any of the arguments are nil, previous argument values for that argument will be used.

```
AffinityPropagation:setParameters(maxNumberOfIterations: integer, distanceFunction: string, preferenceType: string, damping: number, preferenceValueArray: {number})
```

#### Parameters:

* maxNumberOfIterations: How many times should the model needed to be trained.

* distanceFunction: The distance function to be used. Available options are:

    * Euclidean
      
    * Manhattan
      
    * Cosine

* preferenceType: Determines how preferences are calculated. Available options are:

   * Median
 
   * Average
   
   * Minimum
 
   * Maximum
 
   * Precomputed

* damping: A high value leads to fewer changes, while a low value leads to more exploration. The value can be set between 0 and 1.

* preferenceValueArray: An array containing preference values. The index determines the preference value for number of data for that index. Can only be used with "Precomputed" preferenceType.

### train()

Train the model.

```
AffinityPropagation:train(featureMatrix: Matrix)
```

#### Parameters:

* featureMatrix: Matrix containing all data.

#### Returns:

* costArray: An array containing cost values.

### predict()

Predict which cluster does it belong to for a given data.

```
AffinityPropagation:predict(featureMatrix: Matrix): Matrix, Matrix
```

#### Parameters:

* featureMatrix: Matrix containing data.

#### Returns:

* clusterNumberVector: The cluster which the data belongs to.

* shortestDistanceVector: The distance between the datapoint and the center of the cluster (centroids).

## Inherited From

* [BaseModel](BaseModel.md)
