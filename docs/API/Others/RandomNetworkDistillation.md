# [API Reference](../../API.md) - [Others](../Others.md) - RandomNetworkDistillation

RandomNetworkDistillation is a neural network for producing internal rewards to encourage exploration. Requires neural network as your model.

## Constructors

### new()

Create new model object. If any of the arguments are nil, default argument values for that argument will be used.

```
RandomNetworkDistillation.new(): RandomNetworkDistillationObject
```

* RandomNetworkDistillationObject: The generated RandomNetworkDistillation object.

## Functions

### setModel()

```
RandomNetworkDistillation:setModel(Model: ModelObject)
```

#### Parameters

* Model: The model to be used by the RandomNetworkDistillation object.

### getModel()

```
RandomNetworkDistillation:setModel(): ModelObject
```

#### Returns

* Model: The model that is used by the RandomNetworkDistillation object.

### getTargetModelParameters(doNotDeepCopy: boolean)

Gets the target model parameters from the network.

```
RandomNetworkDistillation:getTargetModelParameters(doNotDeepCopy: boolean): ModelParameters
```

#### Parameters

* doNotDeepCopy: Set whether or not to deep copy the model parameters.

#### Returns

* TargetModelParameters: Target network model parameters to be used for predictor network training.

### getPredictorModelParameters()

Gets the target model parameters from the network.

```
RandomNetworkDistillation:getPredictorModelParameters(doNotDeepCopy: boolean): ModelParameters
```

#### Parameters

* doNotDeepCopy: Set whether or not to deep copy the model parameters.

#### Returns

* PredictorModelParameters: Target network model parameters to be used for predictor network training.

### setTargetModelParameters()

Set the model parameters to the network

```
RandomNetworkDistillation:setTargetModelParameters(TargetModelParameters: ModelParameters, doNotDeepCopy: boolean)
```

#### Parameters

* TargetModelParameters: Target network model parameters to be used for predictor network training.

* doNotDeepCopy: Set whether or not to deep copy the model parameters.

### setPredictorModelParameters()

Set the model parameters to the network

```
RandomNetworkDistillation:setPredictorModelParameters(PredictorModelParameters: ModelParameters, doNotDeepCopy: boolean)
```

#### Parameters

* PredictorModelParameters: Predictor network model parameters to be trained so that it tries to match up with target network model parameters.

* doNotDeepCopy: Set whether or not to deep copy the model parameters.

## References

* [Random Network Distillation: A New Take on Curiosity-Driven Learning By Thomas Simonini](https://blog.dataiku.com/random-network-distillation-a-new-take-on-curiosity-driven-learning)

* [Exploration by Random Network Distillation (Research Paper)](https://arxiv.org/abs/1810.12894v1)
