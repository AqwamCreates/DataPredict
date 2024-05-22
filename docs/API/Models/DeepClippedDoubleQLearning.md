# [API Reference](../../API.md) - [Models](../Models.md) - DeepClippedDoubleQLearning (Clipped Double Deep Q-Learning)

DeepClippedDoubleQLearning is a neural network with reinforcement learning capabilities. It can predict any positive numbers of discrete values.

It uses two neural networks where lowest maximum Q-values are selected for training.

## Constructors

### new()

Create new model object. If any of the arguments are nil, default argument values for that argument will be used.

```
DeepClippedDoubleQLearning.new(discountFactor: number): ModelObject
```

#### Parameters:

* discountFactor: The higher the value, the more likely it focuses on long-term outcomes. The value must be set between 0 and 1.

#### Returns:

* ModelObject: The generated model object.

## Functions

### setParameters()

Set model's parameters. When any of the arguments are nil, previous argument values for that argument will be used.

```
DeepClippedDoubleQLearning:setParameters(discountFactor: number)
```

#### Parameters:

* discountFactor: The higher the value, the more likely it focuses on long-term outcomes. The value must be set between 0 and 1.

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

* [ReinforcementLearningBaseModel](ReinforcementLearningBaseModel.md)

## References

* [Double Deep Q Networks](https://towardsdatascience.com/double-deep-q-networks-905dd8325412)

* [Reinforcement Learning with Neural Network](https://www.baeldung.com/cs/reinforcement-learning-neural-network)

* [Deep Q Networks (DQN) in Python From Scratch by Using OpenAI Gym and TensorFlow- Reinforcement Learning Tutorial](https://aleksandarhaber.com/deep-q-networks-dqn-in-python-from-scratch-by-using-openai-gym-and-tensorflow-reinforcement-learning-tutorial/)

* [Reinforcement Q-Learning from Scratch in Python with OpenAI Gym](https://www.learndatasci.com/tutorials/reinforcement-q-learning-scratch-python-openai-gym/)
