# [API Reference](../../API.md) - [Others](../Others.md) - RandomNetworkDistillation

RandomNetworkDistillation is a network for producing internal rewards to encourage exploration in an environment lacking external rewards.

## Constructors

### new()

Create new model object. If any of the arguments are nil, default argument values for that argument will be used.

```
RandomNetworkDistillation.new(maxNumberOfIterations: integer, useNegativeOneBinaryLabel: boolean, targetTotalCost: number): RandomNetworkDistillationObject
```

#### Parameters:

* maxNumberOfIterations: How many times should the model needed to be trained.

* learningRate: The speed at which the model learns. Recommended that the value is set between (0 to 1).

#### Returns:

* RandomNetworkDistillation: The generated RandomNetworkDistillation object.

## Functions

### setParameters()

Set model's parameters. When any of the arguments are nil, previous argument values for that argument will be used.

```
RandomNetworkDistillation:setParameters(maxNumberOfIterations: integer, learningRate: number, targetTotalCost: number)
```

#### Parameters:

* maxNumberOfIterations: How many times should the model needed to be trained.

* learningRate: The speed at which the model learns. Recommended that the value is set between (0 to 1).

### train()

Train the model.

```
NeuralNetwork:generateReward(featureMatrix: Matrix): number
```
#### Parameters:

* featureMatrix: Matrix containing all data.

#### Returns:

* reward: A value produced when comparing the target and predictor network outputs.

### getModelParametersArray()

Gets the model parameters from the base model.

```
OneVsAll:getModelParametersArray(): ModelParameters []
```

#### Returns

* ModelParameters: An array containing model parameters (matrix/table) fetched from each model. The index of the array determines which model it belongs to.

### setModelParametersArray()

Set the model parameters to the base model.

```
OneVsAll:setModelParametersArray(TargetModelParameters: ModelParameters, PredictorModelParameters: ModelParameters)
```

#### Parameters

* ModelParametersArray: A table containing model parameters (matrix/table) to be given to be given to each model stored in OneVsAll object.  The position of the parameters determines which model it belongs to.

## Inherited From

* [NeuralNetwork](NeuralNetwork.md)
