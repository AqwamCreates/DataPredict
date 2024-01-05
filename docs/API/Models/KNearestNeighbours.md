# [API Reference](../../API.md) - [Models](../Models.md) - KNearestNeighbours (KNN)

LinearRegression is a supervised machine learning model that predicts continuous values (e.g. 1.2, -32, 90, -1.2 and etc. ). It uses iterative calculations to find the best model parameters.

## Stored Model Parameters

Contains a table of matrices.  

* ModelParameters[1]: Feature Matrix

* ModelParameters[2]: Label Vector / Label Matrix

## Constructors

### new()

Create new model object. If any of the arguments are nil, default argument values for that argument will be used.

```
KNearestNeighbours.new(kValue: integer, distanceFunction: string): ModelObject
```

#### Parameters:

* kValue: The number of closest data points taken into consideration for majority voting to determine the class of a given data point.

* distanceFunction: The distance function to be used to measure the similarity between two data points. Available options are:

  * Euclidean

  * Manhattan

  * CosineDistance

#### Returns:

* ModelObject: The generated model object.

## Functions

### setParameters()

Set model's parameters. When any of the arguments are nil, previous argument values for that argument will be used.

```
KNearestNeighbours:setParameters(kValue: integer, distanceFunction: string)
```

#### Parameters:

* maxNumberOfIterations: How many times should the model needed to be trained.

* learningRate: The speed at which the model learns. Recommended that the value is set between (0 to 1).

* lossFunction: The function to calculate the cost of each training. Available options are "L1" and "L2".

* targetCost: The cost at which the model stops training.

### train()

Train the model.

```
KNearestNeighbours:train(featureMatrix: Matrix, labelVector: Matrix): number[]
```

#### Parameters:

* featureMatrix: Matrix containing all data.

* labelVector: A (n x 1) matrix containing values related to featureMatrix.

#### Returns:

* costArray: An array containing cost values.

### predict()

Predict the values for given data.

```
KNearestNeighbours:predict(featureMatrix: Matrix, returnOriginalOutput: boolean): number
```

#### Parameters:

* featureMatrix: Matrix containing data.

#### Returns:

* predictedValue: A value that is predicted by the model.

## Inherited From

* [BaseModel](BaseModel.md)
