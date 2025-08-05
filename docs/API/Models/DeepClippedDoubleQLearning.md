# [API Reference](../../API.md) - [Models](../Models.md) - DeepClippedDoubleQLearning (Clipped Deep Double Q-Learning)

DeepClippedDoubleQLearning is a neural network with reinforcement learning capabilities. It can predict any positive numbers of discrete values.

It uses two neural networks where lowest maximum Q-values are selected for training.

## Constructors

### new()

Create new model object. If any of the arguments are nil, default argument values for that argument will be used.

```
DeepClippedDoubleQLearning.new(discountFactor: number): ModelObject
```

#### Parameters:

* lambda: At 0, the model acts like the Temporal Difference algorithm. At 1, the model acts as Monte Carlo algorithm. Between 0 and 1, the model acts as both. [Default: 0]

* discountFactor: The higher the value, the more likely it focuses on long-term outcomes. The value must be set between 0 and 1. [Default: 0.95]

#### Returns:

* ModelObject: The generated model object.

## Functions

### setModelParameters1()

Sets model parameters to be used by the model.

```
DeepClippedDoubleQLearning:setModelParameters1(ModelParameters1: ModelParameters)
```

#### Parameters:

* ModelParameters1: First model parameters to be used by the model.

### setModelParameters2()

Sets model parameters to be used by the model.

```
DeepClippedDoubleQLearning:setModelParameters1(ModelParameters2: ModelParameters)
```

#### Parameters:

* ModelParameters2: Second model parameters to be used by the model.

### getModelParameters1()

Sets model parameters to be used by the model.

```
DeepClippedDoubleQLearning:getModelParameters1(): ModelParameters
```

#### Returns:

* ModelParameters1: First model parameters that was used by the model.

### getModelParameters2()

Sets model parameters to be used by the model.

```
DeepClippedDoubleQLearning:getModelParameters2(): ModelParameters
```

#### Returns:

* ModelParameters2: Second model parameters that was used by the model.

## Inherited From

* [DeepReinforcementLearningBaseModel](DeepReinforcementLearningBaseModel.md)

## References

* [Double Deep Q Networks](https://towardsdatascience.com/double-deep-q-networks-905dd8325412)

* [Reinforcement Learning with Neural Network](https://www.baeldung.com/cs/reinforcement-learning-neural-network)

* [Reinforcement Q-Learning from Scratch in Python with OpenAI Gym](https://www.learndatasci.com/tutorials/reinforcement-q-learning-scratch-python-openai-gym/)
