# [API Reference](../../API.md) - [Models](../Models.md) - TweedieRegression

TweedieRegression is a supervised machine learning model that predicts continuous positive values. It uses iterative calculations to find the best model parameters.

## Stored Model Parameters

Contains a matrix.  

* ModelParameters[I][J]: Value of matrix at row I and column J. The rows are the features.

## Constructors

### new()

Create new model object. If any of the arguments are nil, default argument values for that argument will be used.

```
TweedieRegression.new(maximumNumberOfIterations: integer, learningRate: number, costFunction: string): ModelObject
```

#### Parameters:

* maximumNumberOfIterations: How many times should the model needed to be trained.

* learningRate: The speed at which the model learns. Recommended that the value is set between 0 to 1.

* power: Controls the mean-variance relationship and selects the underlying distribution from the Tweedie family. [Default: 1.5]

  * power = 0: Normal distribution (for continuous data with constant variance)

  * power = 1: Poisson distribution (for count data)

  * power = 2: Gamma distribution (for continuous positive data with constant coefficient of variation)

  * power = 3: Inverse Gaussian distribution (for continuous positive data with heavy tails)

  * 1 < power < 2: Compound Poisson-Gamma distribution (for data with exact zeros and positive continuous values)

#### Returns:

* ModelObject: The generated model object.

## Functions

### setOptimizer()

Set optimizer for the model by inputting the optimizer object.

```
TweedieRegression:setOptimizer(Optimizer: OptimizerObject)
```

#### Parameters:

* Optimizer: The optimizer object to be used.

### setRegularizer()

Set a regularization for the model by inputting the optimizer object.

```
TweedieRegression:setRegularizer(Regularizer: RegularizerObject)
```

#### Parameters:

* Regularizer: The regularizer to be used.

### setSolver()

Set a solver for the model by inputting the optimizer object.

```
TweedieRegression:setSolver(Solver: SolverObject)
```

#### Parameters:

* Solver: The solver to be used.

### train()

Train the model.

```
TweedieRegression:train(featureMatrix: Matrix, labelVector: Matrix): number[]
```

#### Parameters:

* featureMatrix: Matrix containing all data.

* labelVector: A (n x 1) matrix containing values related to featureMatrix.

#### Returns:

* costArray: An array containing cost values.

### predict()

Predict the value for a given data.

```
TweedieRegression:predict(featureMatrix: Matrix): Matrix
```

#### Parameters:

* featureMatrix: Matrix containing data.

#### Returns:

* predictedVector: A vector containing values that are predicted by the model.

## Inherited From

* [GradientMethodBaseModel](GradientMethodBaseModel.md)
