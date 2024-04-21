# [API Reference](../../API.md) - [Others](../Others.md) - RandomNetworkDistillation

RandomNetworkDistillation is a neural network for producing internal rewards to encourage exploration.

## Constructors

### new()

Create new model object. If any of the arguments are nil, default argument values for that argument will be used.

```
RandomNetworkDistillation.new(maxNumberOfIterations: integer: RandomNetworkDistillationObject
```

#### Parameters:

* maxNumberOfIterations: How many times should the model needed to be trained.

#### Returns:

* RandomNetworkDistillationObject: The generated RandomNetworkDistillation object.

## Functions

### setParameters()

Set model's parameters. When any of the arguments are nil, previous argument values for that argument will be used.

```
RandomNetworkDistillation:setParameters(maxNumberOfIterations: integer)
```

#### Parameters:

* maxNumberOfIterations: How many times should the model needed to be trained.

### train()

Train the model.

```
RandomNetworkDistillation:generateReward(featureMatrix: Matrix): number
```
#### Parameters:

* featureMatrix: Matrix containing all data.

#### Returns:

* reward: A value produced when comparing the target and predictor network outputs.

### getTargetModelParameters()

Gets the target model parameters from the network.

```
RandomNetworkDistillation:getTargetModelParameters(): ModelParameters
```

#### Returns

* TargetModelParameters: Target network model parameters to be used for predictor network training.

### getPredictorModelParameters()

Gets the target model parameters from the network.

```
RandomNetworkDistillation:getPredictorModelParameters(): ModelParameters
```

#### Returns

* PredictorModelParameters: Target network model parameters to be used for predictor network training.

### setTargetModelParameters()

Set the model parameters to the network

```
RandomNetworkDistillation:setTargetModelParameters(TargetModelParameters: ModelParameters)
```

#### Parameters

* TargetModelParameters: Target network model parameters to be used for predictor network training.

### setPredictorModelParameters()

Set the model parameters to the network

```
RandomNetworkDistillation:setPredictorModelParameters(PredictorModelParameters: ModelParameters)
```

#### Parameters

* PredictorModelParameters: Predictor network model parameters to be trained so that it tries to match up with target network model parameters.

## Inherited From

* [NeuralNetwork](NeuralNetwork.md)

## References

* [Random Network Distillation: A New Take on Curiosity-Driven Learning By Thomas Simonini](https://blog.dataiku.com/random-network-distillation-a-new-take-on-curiosity-driven-learning)

* [Exploration by Random Network Distillation (Research Paper)](https://arxiv.org/abs/1810.12894v1)
