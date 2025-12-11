# [API Reference](../../API.md) - [Models](../Models.md) - DeepNStepQLearning (Deep N-Step Q Network)

DeepQLearning is a neural network with reinforcement learning capabilities. It can predict any positive numbers of discrete values.

## Constructors

### new()

Create new model object. If any of the arguments are nil, default argument values for that argument will be used.

```
DeepNStepQLearning.new(discountFactor: number, nStep: number): ModelObject
```

#### Parameters:

* discountFactor: The higher the value, the more likely it focuses on long-term outcomes. The value must be set between 0 and 1. [Default: 0.95]

* nStep: The number of future steps considered for credit assignment. Higher values extend the planning horizon but increase variance. Set to 1 for standard Q-learning. [Default: 3]

#### Returns:

* ModelObject: The generated model object.

## Inherited From

* [DeepReinforcementLearningBaseModel](DeepReinforcementLearningBaseModel.md)

## References

* [Reinforcement Learning with Neural Network](https://www.baeldung.com/cs/reinforcement-learning-neural-network)

* [Reinforcement Q-Learning from Scratch in Python with OpenAI Gym](https://www.learndatasci.com/tutorials/reinforcement-q-learning-scratch-python-openai-gym/)
