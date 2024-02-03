# [API Reference](../../API.md) - [Others](../Others.md) - RandomNetworkDistillation

RandomNetworkDistillation is a neural network for producing internal rewards to encourage exploration.

## Constructors

### new()

Create new model object. If any of the arguments are nil, default argument values for that argument will be used.

```
RandomNetworkDistillation.new(maxNumberOfIterations: integer, learningRate: number): RandomNetworkDistillationObject
```

#### Parameters:

* maxNumberOfIterations: How many times should the model needed to be trained.

* learningRate: The speed at which the model learns. Recommended that the value is set between (0 to 1).

#### Returns:

* RandomNetworkDistillationObject: The generated RandomNetworkDistillation object.

## Functions

### setParameters()

Set model's parameters. When any of the arguments are nil, previous argument values for that argument will be used.

```
RandomNetworkDistillation:setParameters(maxNumberOfIterations: integer, learningRate: number)
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

## References

* [Random Network Distillation: A New Take on Curiosity-Driven Learning By Thomas Simonini](https://blog.dataiku.com/random-network-distillation-a-new-take-on-curiosity-driven-learning)

* [Exploration by Random Network Distillation (Research Paper)](https://arxiv.org/abs/1810.12894v1)
