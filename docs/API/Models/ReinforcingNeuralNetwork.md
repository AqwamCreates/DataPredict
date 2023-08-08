# [API Reference](../../API.md) - [Models](../Models.md) - ReinforcingNeuralNetwork

NeuralNetwork is a supervised machine learning model that predicts any positive numbers of discrete values.

## Stored Model Parameters

Contains a table of matrices.  

* ModelParameters[L][I][J]: Matrix at layer L. Value of matrix at row I and column J.

## Constructors

### new()

Create new model object. If any of the arguments are nil, default argument values for that argument will be used.

```
ReinforcingNeuralNetwork.new(maxNumberOfIterations: integer, learningRate: number, targetCost: number): ModelObject
```

#### Parameters:

* maxNumberOfIterations: How many times should the model needed to be trained.

* learningRate: The speed at which the model learns. Recommended that the value is set between (0 to 1).

* targetCost: The cost at which the model stops training.

#### Returns:

* ModelObject: The generated model object.

## Functions

### setParameters()

Set model's parameters. When any of the arguments are nil, previous argument values for that argument will be used.

```
ReinforcingNeuralNetwork:setParameters(maxNumberOfIterations: integer, learningRate: number, targetCost: number)
```

#### Parameters:

* maxNumberOfIterations: How many times should the model needed to be trained.

* learningRate: The speed at which the model learns. Recommended that the value is set between (0 to 1).

* targetCost: The cost at which the model stops training.

### reinforce()

Set model's parameters. When any of the arguments are nil, previous argument values for that argument will be used.

```
ReinforcingNeuralNetwork:reinforce(featureVector: FeatureVector, label: integer, rewardValue: number, punishValue: number)
```

#### Parameters:

* featureVector: How many times should the model needed to be trained.

* label: The speed at which the model learns. Recommended that the value is set between (0 to 1).

* rewardValue: The cost at which the model stops training.

* punishValue: The value 

## Inherited From

* [NeuralNetwork](NeuralNetwork.md)
