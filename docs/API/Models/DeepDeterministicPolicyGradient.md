# [API Reference](../../API.md) - [Models](../Models.md) - DeepDeterministicPolicyGradient

DeepDeterministicPolicyGradient is a base class for reinforcement learning.

## Notes

* The Actor and Critic models must be created separately. Then use setActorModel() and setCriticModel() to put it inside the ActorCritic model.

* Actor and Critic models must be a part of NeuralNetwork model. If you decide to use linear regression or logistic regression, then it must be constructed using NeuralNetwork model. 

* Ensure the final layer of the Critic model has only one neuron. It is the default setting for all Critic models in research papers.

* Ensure the first layer of the Critic model has the same number of neurons as the total number of actions and the number of environment features.

## Constructors

### new()

Create new model object. If any of the arguments are nil, default argument values for that argument will be used.

```
DeepDeterministicPolicyGradient.new(averagingRate: number, discountFactor: number): ModelObject
```

#### Parameters:

* averagingRate: The higher the value, the faster the weights changes. The value must be set between 0 and 1. [Default: 0.995]

* discountFactor: The higher the value, the more likely it focuses on long-term outcomes. The value must be set between 0 and 1. [Default: 0.95]

#### Returns:

* ModelObject: The generated model object.

## Inherited From

* [DeepReinforcementLearningActorCriticBaseModel](DeepReinforcementLearningActorCriticBaseModel.md)

## References

* [Deep Deterministic Policy Gradient By OpenAI](https://spinningup.openai.com/en/latest/algorithms/ddpg.html)
