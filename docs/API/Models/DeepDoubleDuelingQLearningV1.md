# [API Reference](../../API.md) - [Models](../Models.md) - DeepDoubleDuelingQLearningV1 (D3QN)

DeepDoubleDuelingQLearningV1 is a neural network with reinforcement learning capabilities. It can predict any positive numbers of discrete values.

It uses Hasselt et al. (2010) version, where a single neural network is selected from two neural networks with equal probability for training.

## Constructors

### new()

Create new model object. If any of the arguments are nil, default argument values for that argument will be used.

```
DeepDoubleDuelingQLearning.new(discountFactor: number): ModelObject
```

#### Parameters:

* discountFactor: The higher the value, the more likely it focuses on long-term outcomes. The value must be set between 0 and 1.

#### Returns:

* ModelObject: The generated model object.

## Functions

### setParameters()

Set model's parameters. When any of the arguments are nil, previous argument values for that argument will be used.

```
DeepDoubleDuelingQLearning:setParameters(discountFactor: number)
```

#### Parameters:

* discountFactor: The higher the value, the more likely it focuses on long-term outcomes. The value must be set between 0 and 1.

### setModelParameters1()

Sets model parameters to be used by the model.

```
DeepDoubleQLearning:setModelParameters1(ModelParameters1: ModelParameters)
```

#### Parameters:

* ModelParameters1: First model parameters to be used by the model.

### setModelParameters2()

Sets model parameters to be used by the model.

```
DeepDoubleQLearning:setModelParameters2(ModelParameters2: ModelParameters)
```

#### Parameters:

* ModelParameters2: Second model parameters to be used by the model.

### getModelParameters1()

Sets model parameters to be used by the model.

```
DeepDoubleQLearning:getModelParameters1(): ModelParameters
```

#### Returns:

* ModelParameters1: First model parameters that was used by the model.

### getModelParameters2()

Sets model parameters to be used by the model.

```
DeepDoubleQLearning:getModelParameters2(): ModelParameters
```

#### Returns:

* ModelParameters2: Second model parameters that was used by the model.

## Inherited From

* [ReinforcementLearningDeepDuelingQLearningBaseModel](ReinforcementLearningDeepDuelingQLearningBaseModel.md)
