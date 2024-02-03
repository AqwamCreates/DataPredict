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

Gets the model parameters from the network.

```
OneVsAll:getModelParametersArray(): ModelParametersArray []
```

#### Returns

* TargetModelParameters: Target network model parameters to be used for predictor network training.

* PredictorModelParameters: Predictor network model parameters to be trained so that it tries to match up with target network model parameters.

### setModelParametersArray()

Set the model parameters to the network

```
OneVsAll:setModelParametersArray(TargetModelParameters: ModelParameters, PredictorModelParameters: ModelParameters)
```

#### Parameters

* TargetModelParameters: Target network model parameters to be used for predictor network training.

* PredictorModelParameters: Predictor network model parameters to be trained so that it tries to match up with target network model parameters.

## Inherited From

* [NeuralNetwork](NeuralNetwork.md)

## References

* [Random Network Distillation: A New Take on Curiosity-Driven Learning By Thomas Simonini](https://blog.dataiku.com/random-network-distillation-a-new-take-on-curiosity-driven-learning)

* [Exploration by Random Network Distillation (Research Paper)](https://arxiv.org/abs/1810.12894v1)
