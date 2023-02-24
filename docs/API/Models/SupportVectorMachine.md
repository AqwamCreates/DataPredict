# API Reference - Model - SupportVectorMachine

SupportVectorMachine is a supervised machine learning model that predicts values of -1 and 1 only. It assumes that the data is linearly seperable.

## Constructors

### new()

Create new model object. If any of the arguments are not given, default argument values for that argument will be used.

```
SupportVectorMachine.new(maxNumberOfIterations: integer, learningRate: number, cValue: number, distanceFunction: string, targetCost: number): ModelObject
```

#### Parameters:

* maxNumberOfIterations: How many times should the model needed to be trained.

* learningRate: The speed at which the model learns. Recommended that the value is set between (0 to 1).

* cValue: How strict should the model can classify the data correctly. Higher the cValue, less stricter the model can classify the data correctly.

* sigmoidFunction: The function to calculate the cost and cost derivaties of each training. Available options are "L1" and "L2".

* targetCost: The cost at which the model stops training.

#### Returns:

* ModelObject: The generated model object.

## Functions

### setParameters(maxNumberOfIterations: integer, learningRate: number, cValue: number, distanceFunction: string, targetCost: number)

Set model's parameters. When any of the arguments are not given, previous argument values for that argument will be used.

#### Parameters:

* maxNumberOfIterations: How many times should the model needed to be trained.

* learningRate: The speed at which the model learns. Recommended that the value is set between (0 to 1).

* cValue: How strict should the model can classify the data correctly. Higher the cValue, less stricter the model can classify the data correctly.

* sigmoidFunction: The function to calculate the cost and cost derivaties of each training. Available options are "L1" and "L2".

* targetCost: The cost at which the model stops training.

### setOptimizer()

Set optimizer for the model by inputting the optimizer object.

```
SupportVectorMachine:setOptimizer(Optimizer: OptimizerObject)
```

#### Parameters:

* Optimizer: The optimizer to be used.

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

* Predict the value for a given data.

```
SupportVectorMachine:predict(featureMatrix: Matrix): number
```

#### Parameters:

* featureMatrix: Matrix containing all data.

#### Returns:

* predictedValue: A value that is predicted by the model

## Inherited From

* [MachineLearningBaseModel](MachineLearningBaseModel.md)
