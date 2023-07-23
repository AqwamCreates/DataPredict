# [API Reference](../../API.md) - [Models](../Models.md) - SupportVectorMachineOneVsAll

SupportVectorMachineOneVsAll is a supervised machine learning model that predicts values of positive integers. It uses multiple support vector machine models to achieve multi-class classification capabilities.

## Constructors

### new()

Create new model object. If any of the arguments are nil, default argument values for that argument will be used.

```
SupportVectorMachineOneVsAll.new(maxNumberOfIterations: integer, learningRate: number, cValue: number, distanceFunction: string, targetCost: number): ModelObject
```

#### Parameters:

* maxNumberOfIterations: How many times should the model needed to be trained.

* learningRate: The speed at which the model learns. Recommended that the value is set between (0 to 1).

* cValue: How strict should the model can classify the data correctly. Higher the cValue, less stricter the model can classify the data correctly.

* distanceFunction: The function to calculate the cost and cost derivatives of each training. Available options are "manhattan" and "euclidean".

* targetCost: The cost at which the model stops training.

#### Returns:

* ModelObject: The generated model object.

## Functions

### setParameters()

Set model's parameters. When any of the arguments are nil, previous argument values for that argument will be used.

```
SupportVectorMachineOneVsAll:setParameters(maxNumberOfIterations: integer, learningRate: number, cValue: number, distanceFunction: string, targetCost: number)
```

#### Parameters:

* maxNumberOfIterations: How many times should the model needed to be trained.

* learningRate: The speed at which the model learns. Recommended that the value is set between (0 to 1).

* cValue: How strict should the model can classify the data correctly. Higher the cValue, less stricter the model can classify the data correctly.

* distanceFunction: The function to calculate the cost and cost derivaties of each training. Available options are "manhattan" and "euclidean".

* targetCost: The cost at which the model stops training.

### setCValue()

Set how hard the margin should be.

```
SupportVectorMachine:setCValue(cValue: number)
```

#### Parameters:

* cValue: The value of c to be used.

### train()

Train the model.

```
SupportVectorMachineOneVsAll:train(featureMatrix: Matrix, labelVector: Matrix): number[]
```
#### Parameters:

* featureMatrix: Matrix containing all data.

* labelVector: A (n x 1) matrix containing values related to featureMatrix.

#### Returns:

* costArray: An array containing cost values.

### predict()

* Predict the value for a given data.

```
SupportVectorMachineOneVsAll:predict(featureMatrix: Matrix): number
```

#### Parameters:

* featureMatrix: Matrix containing all data.

#### Returns:

* predictedValue: A value that is predicted by the model.

## Inherited From

* [BaseModel](BaseModel.md)
