# [API Reference](../../API.md) - [Models](../Models.md) - HuberRegression

HuberRegression is a supervised machine learning model that predicts continuous values (e.g. 1.2, -32, 90, -1.2 and etc. ). It uses iterative calculations to find the best model parameters.

## Stored Model Parameters

Contains a matrix.  

* ModelParameters[I][J]: Value of matrix at row I and column J. The rows are the features.

## Constructors

### new()

Create new model object. If any of the arguments are nil, default argument values for that argument will be used.

```
HuberRegression.new(maximumNumberOfIterations: integer, learningRate: number, costFunction: string): ModelObject
```

#### Parameters:

* maximumNumberOfIterations: How many times should the model needed to be trained.

* learningRate: The speed at which the model learns. Recommended that the value is set between 0 to 1.

* delta: The threshold parameter for the Huber loss function. Determines where the loss function transitions from quadratic to linear behavior. Typical values range from 0.1 to 10, with smaller values making the model more robust to outliers. [Default: 1]

#### Returns:

* ModelObject: The generated model object.

## Functions

### setOptimizer()

Set optimizer for the model by inputting the optimizer object.

```
HuberRegression:setOptimizer(Optimizer: OptimizerObject)
```

#### Parameters:

* Optimizer: The optimizer object to be used.

### setRegularizer()

Set a regularization for the model by inputting the optimizer object.

```
HuberRegression:setRegularizer(Regularizer: RegularizerObject)
```

#### Parameters:

* Regularizer: The regularizer to be used.

### setSolver()

Set a solver for the model by inputting the optimizer object.

```
HuberRegression:setSolver(Solver: SolverObject)
```

#### Parameters:

* Solver: The solver to be used.

### train()

Train the model.

```
HuberRegression:train(featureMatrix: Matrix, labelVector: Matrix): number[]
```

#### Parameters:

* featureMatrix: Matrix containing all data.

* labelVector: A (n x 1) matrix containing values related to featureMatrix.

#### Returns:

* costArray: An array containing cost values.

### predict()

Predict the value for a given data.

```
HuberRegression:predict(featureMatrix: Matrix): Matrix
```

#### Parameters:

* featureMatrix: Matrix containing data.

#### Returns:

* predictedVector: A vector containing values that are predicted by the model.

## Inherited From

* [GradientMethodBaseModel](GradientMethodBaseModel.md)
