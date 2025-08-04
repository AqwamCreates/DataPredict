# [API Reference](../../API.md) - [Models](../Models.md) - DeepDoubleQLearningV1 (DDQN)

DeepDoubleQLearningV1 is a neural network with reinforcement learning capabilities. It can predict any positive numbers of discrete values.

It uses Hasselt et al. (2010) version, where a single neural network is selected from two neural networks with equal probability for training.

## Constructors

### new()

Create new model object. If any of the arguments are nil, default argument values for that argument will be used.

```
DeepDoubleQLearning.new(lambda: number, discountFactor: number): ModelObject
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
DeepDoubleQLearning:setModelParameters1(ModelParameters1: ModelParameters, doNotDeepCopy: boolean)
```

#### Parameters:

* ModelParameters1: First model parameters to be used by the model.

* doNotDeepCopy: Set whether or not to deep copy the model parameters.

### setModelParameters2()

Sets model parameters to be used by the model.

```
DeepDoubleQLearning:setModelParameters2(ModelParameters2: ModelParameters, doNotDeepCopy: boolean)
```

#### Parameters:

* ModelParameters2: Second model parameters to be used by the model.

* doNotDeepCopy: Set whether or not to deep copy the model parameters.

### getModelParameters1()

Sets model parameters to be used by the model.

```
DeepDoubleQLearning:getModelParameters1(doNotDeepCopy: boolean): ModelParameters
```

#### Parameters:

* doNotDeepCopy: Set whether or not to deep copy the model parameters.

#### Returns:

* ModelParameters1: First model parameters that was used by the model.

### getModelParameters2()

Sets model parameters to be used by the model.

```
DeepDoubleQLearning:getModelParameters2(doNotDeepCopy: boolean): ModelParameters
```

#### Parameters:

* doNotDeepCopy: Set whether or not to deep copy the model parameters.

#### Returns:

* ModelParameters2: Second model parameters that was used by the model.

## Inherited From

* [DeepReinforcementLearningBaseModel](DeepReinforcementLearningBaseModel.md)

## References

* [Double Deep Q Networks](https://towardsdatascience.com/double-deep-q-networks-905dd8325412)

* [Reinforcement Learning with Neural Network](https://www.baeldung.com/cs/reinforcement-learning-neural-network)

* [Reinforcement Q-Learning from Scratch in Python with OpenAI Gym](https://www.learndatasci.com/tutorials/reinforcement-q-learning-scratch-python-openai-gym/)
