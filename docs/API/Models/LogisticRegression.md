# [API Reference](../../API.md) - [Models](../Models.md) - LogisticRegression

LogisticRegression is a supervised machine learning model that predicts values of 0 and 1 only.

## Stored Model Parameters

Contains a matrix.  

* ModelParameters[I][J]: Value of matrix at row I and column J. The rows are the features.

## Constructors

### new()

Create new model object. If any of the arguments are nil, default argument values for that argument will be used.

```
LogisticRegression.new(maxNumberOfIterations: integer, learningRate: number, sigmoidFunction: string, targetCost: number): ModelObject
```

#### Parameters:

* maxNumberOfIterations: How many times should the model needed to be trained.

* learningRate: The speed at which the model learns. Recommended that the value is set between (0 to 1).

* sigmoidFunction: The function to calculate the cost and cost derivaties of each training. Available options are "sigmoid".

* targetCost: The cost at which the model stops training.

#### Returns:

* ModelObject: The generated model object.

## Functions

### setParameters()

Set model's parameters. When any of the arguments are nil, previous argument values for that argument will be used.

```
LogisticRegression:setParameters(maxNumberOfIterations: integer, learningRate: number, sigmoidFunction: string, targetCost: number)
```

#### Parameters:

* maxNumberOfIterations: How many times should the model needed to be trained.

* learningRate: The speed at which the model learns. Recommended that the value is set between (0 to 1).

* sigmoidFunction: The function to calculate the cost and cost derivaties of each training. Available options are "sigmoid".

* targetCost: The cost at which the model stops training.

### setOptimizer()

Set optimizer for the model by inputting the optimizer object.

```
LogisticRegression:setOptimizer(Optimizer: OptimizerObject)
```

#### Parameters:

* Optimizer: The optimizer object to be used.

### setRegularization()

Set a regularization for the model by inputting the optimizer object.

```
LogisticRegression:setRegularization(Regularization: RegularizationObject)
```

#### Parameters:

* Regularization: The regularization to be used.

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

* Predict the value for a given data.

```
LogisticRegression:predict(featureMatrix: Matrix): integer, number
```

#### Parameters:

* featureMatrix: Matrix containing all data.

#### Returns:

* predictedValue: A value that is predicted by the model.

* probability: The probability of predicted value.

## Inherited From

* [BaseModel](BaseModel.md)
