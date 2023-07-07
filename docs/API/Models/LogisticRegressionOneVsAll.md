# [API Reference](../../API.md) - [Models](../Models.md) - LogisticRegressionOneVsAll

LogisticRegressionOneVsAll is a supervised machine learning model that predicts values of positive integers. It uses multiple logistic regression models to produce multi-class prediction capabilities for this model.

## Constructors

### new()

Create new model object. If any of the arguments are nil, default argument values for that argument will be used.

```
LogisticRegressionOneVsAll.new(maxNumberOfIterations: integer, learningRate: number, sigmoidFunction: string, targetCost: number): ModelObject
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
LogisticRegressionOneVsAll:setParameters(maxNumberOfIterations: integer, learningRate: number, sigmoidFunction: string, targetCost: number)
```

#### Parameters:

* maxNumberOfIterations: How many times should the model needed to be trained.

* learningRate: The speed at which the model learns. Recommended that the value is set between (0 to 1).

* sigmoidFunction: The function to calculate the cost and cost derivaties of each training. Available options are "sigmoid".

* targetCost: The cost at which the model stops training.

### train()

Train the model.

```
LogisticRegressionOneVsAll:train(featureMatrix: Matrix, labelVector: Matrix): number[]
```
#### Parameters:

* featureMatrix: Matrix containing all data.

* labelVector: A (n x 1) matrix containing values related to featureMatrix.

#### Returns:

* costArray: An array containing cost values.

### predict()

* Predict the value for a given data.

```
LogisticRegressionOneVsAll:predict(featureMatrix: Matrix): number
```

#### Parameters:

* featureMatrix: Matrix containing all data.

#### Returns:

* predictedValue: A value that is predicted by the model.

## Inherited From

* [BaseModel](BaseModel.md)
