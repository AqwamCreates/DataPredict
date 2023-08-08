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

Reward or punish model based on the predicted output.

```
NeuralNetwork:reinforce(featureVector: Matrix, label: integer, rewardValue: number, punishValue: number): integer
```

#### Parameters:

* featureVector: Matrix containing data.

* label: Actual label.

* rewardValue: How much do we reward the model if it gets the prediction correct (value between 0 and 1).

* punishValue: How much do we punish the model if it gets the prediction incorrect (value between 0 and 1).

#### Returns:

* predictedValue: A value that is predicted by the model.

* probability: The 

## Inherited From

* [NeuralNetwork](NeuralNetwork.md)
