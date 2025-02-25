# [API Reference](../../API.md) - [Models](../Models.md) - VanillaPolicyGradient (VPG)

VanillaPolicyGradient is a base class for reinforcement learning.

## Notes

* The Actor and Critic models must be created separately. Then use setActorModel() and setCriticModel() to put it inside the AdvantageActorCritic model.

* Actor and Critic models must be a part of NeuralNetwork model. If you decide to use linear regression or logistic regression, then it must be constructed using NeuralNetwork model. 

* Ensure the final layer of the Critic model has only one neuron. It is the default setting for all Critic models in research papers.

## Constructors

### new()

Create new model object. If any of the arguments are nil, default argument values for that argument will be used.

```
VanillaPolicyGradient.new(discountFactor: number): ModelObject
```

#### Parameters:

* discountFactor: The higher the value, the more likely it focuses on long-term outcomes. The value must be set between 0 and 1. [Default: 0.95]

#### Returns:

* ModelObject: The generated model object.

## Inherited From

* [ReinforcementLearningActorCriticBaseModel](ReinforcementLearningActorCriticBaseModel.md)

## References

* [Vanilla Policy Gradient By OpenAI](https://spinningup.openai.com/en/latest/algorithms/vpg.html)
