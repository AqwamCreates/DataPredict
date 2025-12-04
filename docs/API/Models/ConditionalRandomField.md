# [API Reference](../../API.md) - [Models](../Models.md) - ConditionalRandomField

LogisticRegression is a supervised machine learning model that predicts values of 0 and 1 only.

## Stored Model Parameters

Contains a matrix.  

* ModelParameters[I][J]: Value of matrix at row I and column J. The rows are the features.

## Constructors

### new()

Create new model object. If any of the arguments are nil, default argument values for that argument will be used.

```
ConditionalRandomField.new(maximumNumberOfIterations: integer, learningRate: number, addBias: boolean): ModelObject
```

#### Parameters:

* maximumNumberOfIterations: How many times should the model needed to be trained.

* learningRate: The speed at which the model learns. Recommended that the value is set between 0 to 1.

* addBias: Set whether or not to add bias [Default: true]

#### Returns:

* ModelObject: The generated model object.

## Functions

### setOptimizer()

Set optimizer for the model by inputting the optimizer object.

```
ConditionalRandomField:setOptimizer(Optimizer: OptimizerObject)
```

#### Parameters:

* Optimizer: The optimizer object to be used.

### setRegularizer()

Set a regularization for the model by inputting the optimizer object.

```
LogisticRegression:setRegularizer(Regularizer: RegularizerObject)
```

#### Parameters:

* Regularizer: The regularizer to be used.

### train()

Train the model.

```
ConditionalRandomField:train(previousStateMatrix: Matrix, currentStateMatrix: Matrix): number[]
```
#### Parameters:

* previousStateMatrix: A matrix containing all previous state data.

* currentStateMatrix: A matrix containing all current state data.

#### Returns:

* costArray: An array containing cost values.

### predict()

Predict the values for given data.

```
ConditionalRandomField:predict(previousStateMatrix: Matrix, returnOriginalOutput: boolean): Matrix, Matrix -OR- Matrix
```

#### Parameters:

* previousStateMatrix: A matrix containing all previous state data.

* returnOriginalOutput: Set whether or not to return predicted current state matrix instead of value with highest probability. 

#### Returns:

* predictedVector: A vector that is predicted by the model.

* probabilityVector: A vector that contains the probability of predicted values in predictedVector.

-OR-

* predictedCurrentStateMatrix: A matrix containing all predicted values from all classes.

## Inherited From

* [GradientMethodBaseModel](GradientMethodBaseModel.md)
