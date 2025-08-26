# [API Reference](../../API.md) - [Models](../Models.md) - DeepQLearning (Deep Q Network)

DeepQLearning is a neural network with reinforcement learning capabilities. It can predict any positive numbers of discrete values.

## Constructors

### new()

Create new model object. If any of the arguments are nil, default argument values for that argument will be used.

```
DeepQLearning.new(discountFactor: number, EligibilityTrace: EligibilityTraceObject): ModelObject
```

#### Parameters:

* discountFactor: The higher the value, the more likely it focuses on long-term outcomes. The value must be set between 0 and 1. [Default: 0.95]

* EligibilityTrace: The eligibility trace object to keep track of credit assignments of state-action pairs.

#### Returns:

* ModelObject: The generated model object.

## Inherited From

* [DeepReinforcementLearningBaseModel](DeepReinforcementLearningBaseModel.md)

## References

* [Reinforcement Learning with Neural Network](https://www.baeldung.com/cs/reinforcement-learning-neural-network)

* [Reinforcement Q-Learning from Scratch in Python with OpenAI Gym](https://www.learndatasci.com/tutorials/reinforcement-q-learning-scratch-python-openai-gym/)
