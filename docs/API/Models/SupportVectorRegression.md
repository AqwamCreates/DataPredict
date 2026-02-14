# [API Reference](../../API.md) - [Models](../Models.md) - SupportVectorRegression

SupportVectorRegression is a supervised machine learning model that predicts values from given inputs.

## Stored Model Parameters

Contains a matrix.  

* ModelParameters[I][J]: Value of matrix at row I and column J. The rows represent the features.

## Constructors

### new()

Create new model object. If any of the arguments are nil, default argument values for that argument will be used.

```
SupportVectorRegression.new(maximumNumberOfIterations: integer, cValue: number, epsilon: number, kernelFunction: string, kernelParameters: table): ModelObject
```

#### Parameters:

* maximumNumberOfIterations: How many times should the model needed to be trained.

* cValue: How strict should the model can classify the data correctly. Higher the cValue, the closer the data points to the decision boundary.

* epsilon: How far the datapoint should be so that it does not contribute to the error calculation. Higher the value, the further the datapoint can be so that it does not contribute to the error calculations.

* kernelFunction: The kernel function to be used to train the model. Available options are:
  
  *  Linear

  *  Polynomial

  *  RadialBasisFunction

  *  Sigmoid

  *  Cosine

* kernelParameters: A table containg the required parameters 

#### Returns:

* ModelObject: The generated model object.

## Functions

### setParameters()

Set model's parameters. When any of the arguments are nil, previous argument values for that argument will be used.

```
SupportVectorRegression:setParameters(maximumNumberOfIterations: integer, cValue: number, epsilon: number, kernelFunction: string, kernelParameters: table)
```

#### Parameters:

* maximumNumberOfIterations: How many times should the model needed to be trained.

* cValue: How strict should the model can classify the data correctly. Higher the cValue, the closer the data points to the decision boundary.

* epsilon: How far the datapoint should be so that it does not contribute to the error calculation. Higher the value, the further the datapoint can be so that it does not contribute to the error calculations.

* kernelFunction: The kernel function to be used to train the model. Available options are:
  
  *  Linear

  *  Polynomial

  *  RadialBasisFunction

  *  Sigmoid

  *  Cosine

* kernelParameters: A table containg the required parameters 

### setCValue()

Set how hard the margin should be.

```
SupportVectorRegression:setCValue(cValue: number)
```

#### Parameters:

* cValue: The value of c to be used.

### train()

Train the model.

```
SupportVectorRegression:train(featureMatrix: Matrix, labelVector: Matrix): number[]
```
#### Parameters:

* featureMatrix: Matrix containing all data.

* labelVector: A (n x 1) matrix containing values related to featureMatrix.

#### Returns:

* costArray: An array containing cost values.

### predict()

Predict the values for given data.

```
SupportVectorRegression:predict(featureMatrix: Matrix): Matrix
```

#### Parameters:

* featureMatrix: Matrix containing all data.

#### Returns:

* predictedVector: A vector that is predicted by the model.

## Inherited From

* [IterativeMethodBaseModel](IterativeMethodBaseModel.md)

## References

* Kernel Methods and Support Vector Machines by Zhiyuan Chen from University Of Nottingham Malaysia (2022/2023). It is the university that Aqwam got his Bachelor's degree in.

* [Lecture 16: Kernels and Feature Extraction](https://www.cs.cornell.edu/courses/cs4787/2021sp/notebooks/Slides16.html)
