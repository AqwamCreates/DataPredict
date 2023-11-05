# [API Reference](../../API.md) - [Models](../Models.md) - SupportVectorMachine

SupportVectorMachine is a supervised machine learning model that predicts values of -1 and 1 only.

## Stored Model Parameters

Contains a matrix.  

* ModelParameters[I][J]: Value of matrix at row I and column J. The rows represent the features.

## Constructors

### new()

Create new model object. If any of the arguments are nil, default argument values for that argument will be used.

```
SupportVectorMachine.new(maxNumberOfIterations: integer, cValue: number, targetCost: number, kernelFunction: string, kernelParameters: table): ModelObject
```

#### Parameters:

* maxNumberOfIterations: How many times should the model needed to be trained.

* learningRate: The speed at which the model learns. Recommended that the value is set between (0 to 1).

* cValue: How strict should the model can classify the data correctly. Higher the cValue, the closer the data points to the decision boundary.

* targetCost: The cost at which the model stops training.

* kernelFunction: The kernel function to be used to train the model. Available options are:
  
  *  linear

  *  polynomial

  *  radialBasisFunction

  *  cosineSimilarity

  *  sigmoid

* kernelParameters: A table containg the required parameters 

#### Returns:

* ModelObject: The generated model object.

## Functions

### setParameters()

Set model's parameters. When any of the arguments are nil, previous argument values for that argument will be used.

```
SupportVectorMachine:setParameters(maxNumberOfIterations: integer, cValue: number, targetCost: number,  kernelFunction: string, kernelParameters: table)
```

#### Parameters:

* maxNumberOfIterations: How many times should the model needed to be trained.

* learningRate: The speed at which the model learns. Recommended that the value is set between (0 to 1).

* cValue: How strict should the model can classify the data correctly. Higher the cValue, the closer the data points to the decision boundary.

* targetCost: The cost at which the model stops training.

* kernelFunction: The kernel function to be used to train the model. Available options are:
  
  *  linear

  *  polynomial

  *  radialBasisFunction

  *  cosineSimilarity

  *  sigmoid

* kernelParameters: A table containg the required parameters 

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
SupportVectorMachine:train(featureMatrix: Matrix, labelVector: Matrix): number[]
```
#### Parameters:

* featureMatrix: Matrix containing all data.

* labelVector: A (n x 1) matrix containing values related to featureMatrix.

#### Returns:

* costArray: An array containing cost values.

### predict()

Predict the values for given data.

```
SupportVectorMachine:predict(featureMatrix: Matrix, returnOriginalOutput): Matrix -OR- Matrix
```

#### Parameters:

* featureMatrix: Matrix containing all data.

#### Returns:

* predictedVector: A vector that is predicted by the model.

-OR-

* originalPredictedVector: A vector that contains the original predicted values.

## Inherited From

* [BaseModel](BaseModel.md)

## References

* Lecture 5: Kernel Methods and Support Vector Machines by Zhiyuan Chen from University Of Nottingham Malaysia. It is the university that Aqwam got his Bachelor's degree in.
