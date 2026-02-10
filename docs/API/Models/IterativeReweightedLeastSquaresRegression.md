# [API Reference](../../API.md) - [Models](../Models.md) - IterativeReweightedLeastSquaresRegression

IterativeReweightedLeastSquaresRegression is a supervised machine learning model that predicts continuous values (e.g. 1.2, -32, 90, -1.2 and etc. ). It uses iterative calculations to find the best model parameters.

## Stored Model Parameters

Contains a matrix.  

* ModelParameters[I][J]: Value of matrix at row I and column J. The rows are the features.

## Constructors

### new()

Create new model object. If any of the arguments are nil, default argument values for that argument will be used.

```
IterativeReweightedLeastSquaresRegression.new(maximumNumberOfIterations: integer, linkFunction: string, pValue: number): ModelObject
```

#### Parameters:

* maximumNumberOfIterations: How many times should the model needed to be trained.

* linkFunction: The link function to use for generating output values from input values. Available options are:

  * Linear (Default)
 
  * Logit
 
  * Probit
 
  * LogLog
 
  * ComplementLogLog

* pValue: How complicated should the "cost landscape" should be. The higher the value, the more complicated the landscape would be. [Default: 2]

#### Returns:

* ModelObject: The generated model object.

## Functions

### train()

Train the model.

```
IterativeReweightedLeastSquaresRegression:train(featureMatrix: Matrix, labelVector: Matrix): number[]
```

#### Parameters:

* featureMatrix: Matrix containing all data.

* labelVector: A (n x 1) matrix containing values related to featureMatrix.

#### Returns:

* costArray: An array containing cost values.

### predict()

Predict the value for a given data.

```
IterativeReweightedLeastSquaresRegression:predict(featureMatrix: Matrix): Matrix
```

#### Parameters:

* featureMatrix: Matrix containing data.

#### Returns:

* predictedVector: A vector containing values that are predicted by the model.

## Inherited From

* [IterativeMethodBaseModel](IterativeMethodBaseModel.md)

## Refernces

* [Iterately Reweighted Least Squares](https://en.wikipedia.org/wiki/Iteratively_reweighted_least_squares)

* [Iterated Reweighted Least Squares and GLMs Explained](https://towardsdatascience.com/iterated-reweighted-least-squares-and-glms-explained-9c0cc0063526/)
