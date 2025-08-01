# [API Reference](../../API.md) - [Models](../Models.md) - LinearRegression

LinearRegression is a supervised machine learning model that predicts continuous values (e.g. 1.2, -32, 90, -1.2 and etc. ). It uses iterative calculations to find the best model parameters.

## Stored Model Parameters

Contains a matrix.  

* ModelParameters[I][J]: Value of matrix at row I and column J. The rows are the features.

## Constructors

### new()

Create new model object. If any of the arguments are nil, default argument values for that argument will be used.

```
LinearRegression.new(maximumNumberOfIterations: integer, learningRate: number, costFunction: string): ModelObject
```

#### Parameters:

* maximumNumberOfIterations: How many times should the model needed to be trained.

* learningRate: The speed at which the model learns. Recommended that the value is set between 0 to 1.

* costFunction: The function to calculate the cost of each training. Available options are "L1" and "L2".

#### Returns:

* ModelObject: The generated model object.

## Functions

### setParameters()

Set model's parameters. When any of the arguments are nil, previous argument values for that argument will be used.

```
LinearRegression:setParameters(maximumNumberOfIterations: integer, learningRate: number, costFunction: string)
```

#### Parameters:

* maximumNumberOfIterations: How many times should the model needed to be trained.

* learningRate: The speed at which the model learns. Recommended that the value is set between (0 to 1).

* costFunction: The function to calculate the cost of each training. Available options are "L1" and "L2".

### setOptimizer()

Set optimizer for the model by inputting the optimizer object.

```
LinearRegression:setOptimizer(Optimizer: OptimizerObject)
```

#### Parameters:

* Optimizer: The optimizer object to be used.

### setRegularizer()

Set a regularization for the model by inputting the optimizer object.

```
LinearRegression:setRegularizer(Regularizer: RegularizerObject)
```

#### Parameters:

* setRegularizer: The regularizer to be used.

### train()

Train the model.

```
LinearRegression:train(featureMatrix: Matrix, labelVector: Matrix): number[]
```

#### Parameters:

* featureMatrix: Matrix containing all data.

* labelVector: A (n x 1) matrix containing values related to featureMatrix.

#### Returns:

* costArray: An array containing cost values.

### predict()

Predict the values for given data.

```
LinearRegression:predict(featureMatrix: Matrix): number
```

#### Parameters:

* featureMatrix: Matrix containing data.

#### Returns:

* predictedValue: A value that is predicted by the model.

## Inherited From

* [GradientMethodBaseModel](GradientMethodBaseModel.md)
