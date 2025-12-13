# [API Reference](../../API.md) - [Models](../Models.md) - QuantileRegression

QuantileRegression is a supervised machine learning model that predicts continuous values (e.g. 1.2, -32, 90, -1.2 and etc. ). It uses iterative calculations to find the best model parameters.

## Stored Model Parameters

Contains a matrix.  

* ModelParameters[I][J]: Value of matrix at row I and column J. The rows are the features.

## Constructors

### new()

Create new model object. If any of the arguments are nil, default argument values for that argument will be used.

```
QuantileRegression.new(maximumNumberOfIterations: integer, learningRate: number, quantilesList: {number}): ModelObject
```

#### Parameters:

* maximumNumberOfIterations: How many times should the model needed to be trained.

* learningRate: The speed at which the model learns. Recommended that the value is set between 0 to 1.

* quantilesList: The list of quantiles to be used for generating the model parameters. [Default: {0.5}]

#### Returns:

* ModelObject: The generated model object.

## Functions

### setOptimizer()

Set optimizer for the model by inputting the optimizer object.

```
QuantileRegression:setOptimizer(Optimizer: OptimizerObject)
```

#### Parameters:

* Optimizer: The optimizer object to be used.

### setRegularizer()

Set a regularization for the model by inputting the optimizer object.

```
QuantileRegression:setRegularizer(Regularizer: RegularizerObject)
```

#### Parameters:

* setRegularizer: The regularizer to be used.

### train()

Train the model.

```
QuantileRegression:train(featureMatrix: Matrix, labelVector: Matrix): number[]
```

#### Parameters:

* featureMatrix: Matrix containing all data.

* labelVector: A (n x 1) matrix containing values related to featureMatrix.

#### Returns:

* costArray: An array containing cost values.

### predict()

Predict the value for a given data.

```
QuantileRegression:predict(featureMatrix: Matrix): Matrix
```

#### Parameters:

* featureMatrix: Matrix containing data.

#### Returns:

* predictedMatrix: A matrix containing values that are predicted by the model. The columns are the quantiles.

## Inherited From

* [GradientMethodBaseModel](GradientMethodBaseModel.md)
