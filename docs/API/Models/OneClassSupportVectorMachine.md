# [API Reference](../../API.md) - [Models](../Models.md) - OneClassSupportVectorMachine

OneClassSupportVectorMachine is a supervised machine learning model that predicts values of -1 / 0 and 1 only.

## Stored Model Parameters

Contains a matrix.  

* ModelParameters[I][J]: Value of matrix at row I and column J. The rows represent the features.

## Constructors

### new()

Create new model object. If any of the arguments are nil, default argument values for that argument will be used.

```
OneClassSupportVectorMachine.new(maximumNumberOfIterations: integer, cValue: number, kernelFunction: string, kernelParameters: table): ModelObject
```

#### Parameters:

* maximumNumberOfIterations: How many times should the model needed to be trained.

* cValue: How strict should the model can classify the data correctly. Higher the cValue, the closer the data points to the decision boundary.

* kernelFunction: The kernel function to be used to train the model. Available options are:
  
  *  Linear

  *  Polynomial

  *  RadialBasisFunction

  *  Sigmoid

  *  Cosine

* beta: Fraction of samples treated as normal during each training iteration (between 0 and 1). The higher the value, the more it will try to include all the datapoints. [Default: 0.9]

#### Returns:

* ModelObject: The generated model object.

## Functions

### setParameters()

Set model's parameters. When any of the arguments are nil, previous argument values for that argument will be used.

```
OneClassSupportVectorMachine:setParameters(maximumNumberOfIterations: integer, cValue: number, kernelFunction: string, kernelParameters: table)
```

#### Parameters:

* maximumNumberOfIterations: How many times should the model needed to be trained.

* cValue: How strict should the model can classify the data correctly. Higher the cValue, the closer the data points to the decision boundary.

* kernelFunction: The kernel function to be used to train the model. Available options are:
  
  *  Linear

  *  Polynomial

  *  RadialBasisFunction

  *  Sigmoid

  *  Cosine

* beta: Fraction of samples treated as normal during each training iteration (between 0 and 1). The higher the value, the more it will try to include all the datapoints. [Default: 0.9]

### setCValue()

Set how hard the margin should be.

```
OneClassSupportVectorMachine:setCValue(cValue: number)
```

#### Parameters:

* cValue: The value of c to be used.

### train()

Train the model.

```
OneClassSupportVectorMachine:train(featureMatrix: Matrix, labelVector: Matrix): number[]
```
#### Parameters:

* featureMatrix: Matrix containing all data.

* labelVector: A (n x 1) matrix containing values related to featureMatrix. Optional.

#### Returns:

* costArray: An array containing cost values.

### predict()

Predict the values for given data.

```
OneClassSupportVectorMachine:predict(featureMatrix: Matrix, returnOriginalOutput: boolean): Matrix -OR- Matrix
```

#### Parameters:

* featureMatrix: Matrix containing all data.

#### Returns:

* predictedVector: A vector that is predicted by the model.

-OR-

* originalPredictedVector: A vector that contains the original predicted values.

## Inherited From

* [IterativeMethodBaseModel](IterativeMethodBaseModel.md)

## References

* Kernel Methods and Support Vector Machines by Zhiyuan Chen from University Of Nottingham Malaysia (2022/2023). It is the university that Aqwam got his Bachelor's degree in.

* [Lecture 16: Kernels and Feature Extraction](https://www.cs.cornell.edu/courses/cs4787/2021sp/notebooks/Slides16.html)

* [Enhancing one-class Support Vector Machines for unsupervised anomaly detection](https://www.researchgate.net/publication/262288578_Enhancing_one-class_Support_Vector_Machines_for_unsupervised_anomaly_detection)
