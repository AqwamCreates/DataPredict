# [API Reference](../../API.md) - [Models](../Models.md) - LogisticRegression

LogisticRegression is a supervised machine learning model that predicts values of 0 and 1 only.

## Stored Model Parameters

Contains a matrix.  

* ModelParameters[I][J]: Value of matrix at row I and column J. The rows are the features.

## Constructors

### new()

Create new model object. If any of the arguments are nil, default argument values for that argument will be used.

```
LogisticRegression.new(maximumNumberOfIterations: integer, learningRate: number, sigmoidFunction: string): ModelObject
```

#### Parameters:

* maximumNumberOfIterations: How many times should the model needed to be trained.

* learningRate: The speed at which the model learns. Recommended that the value is set between 0 to 1.

* sigmoidFunction: The function to calculate the cost and cost derivaties of each training. Available options are "Sigmoid".

#### Returns:

* ModelObject: The generated model object.

## Functions

### setParameters()

Set model's parameters. When any of the arguments are nil, previous argument values for that argument will be used.

```
LogisticRegression:setParameters(maximumNumberOfIterations: integer, learningRate: number, sigmoidFunction: string)
```

#### Parameters:

* maximumNumberOfIterations: How many times should the model needed to be trained.

* learningRate: The speed at which the model learns. Recommended that the value is set between (0 to 1).

* sigmoidFunction: The function to calculate the cost and cost derivaties of each training. Available options are "Sigmoid".

### setOptimizer()

Set optimizer for the model by inputting the optimizer object.

```
LogisticRegression:setOptimizer(Optimizer: OptimizerObject)
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
LogisticRegression:train(featureMatrix: Matrix, labelVector: Matrix): number[]
```
#### Parameters:

* featureMatrix: Matrix containing all data.

* labelVector: A (n x 1) matrix containing values related to featureMatrix.

#### Returns:

* costArray: An array containing cost values.

### predict()

Predict the values for given data.

```
LogisticRegression:predict(featureMatrix: Matrix, returnOriginalOutput: boolean): Matrix, Matrix -OR- Matrix
```

#### Parameters:

* featureMatrix: Matrix containing all data.

* returnOriginalOutput: Set whether or not to return predicted matrix instead of value with highest probability. 

#### Returns:

* predictedVector: A vector that is predicted by the model.

* probabilityVector: A vector that contains the probability of predicted values in predictedVector.

-OR-

* predictedMatrix: A matrix containing all predicted values from all classes.

## Inherited From

* [GradientMethodBaseModel](GradientMethodBaseModel.md)
