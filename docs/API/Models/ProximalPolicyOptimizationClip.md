# [API Reference](../../API.md) - [Models](../Models.md) - ProximalPolicyOptimizationClip (PPO-Clip)

ProximalPolicyOptimizationClip is a base class for reinforcement learning.

## Notes

* The Actor and Critic models must be created separately. Then use setActorModel() and setCriticModel() to put it inside the AdvantageActorCritic model.

* Actor and Critic models must be a part of NeuralNetwork model. If you decide to use linear regression or logistic regression, then it must be constructed using NeuralNetwork model. 

* Ensure the final layer of the Critic model has only one neuron. It is the default setting for all Critic models in research papers.

## Constructors

### new()

Create new model object. If any of the arguments are nil, default argument values for that argument will be used.

```
ProximalPolicyOptimizationClip.new(clipRatio: number, lambda: number, discountFactor: number): ModelObject
```

#### Parameters:

* clipRatio: A value that controls how far the new policy can get far from old policy. The value must be set between 0 and 1. [Default: 0.3]

* lambda: At 0, the model acts like the Temporal Difference algorithm. At 1, the model acts as Monte Carlo algorithm. Between 0 and 1, the model acts as both. [Default: 0]

* discountFactor: The higher the value, the more likely it focuses on long-term outcomes. The value must be set between 0 and 1. [Default: 0.95]

#### Returns:

* ModelObject: The generated model object.

## Inherited From

* [ReinforcementLearningActorCriticBaseModel](ReinforcementLearningActorCriticBaseModel.md)

## References

* [Proximal Policy Optimization By OpenAI](https://spinningup.openai.com/en/latest/algorithms/ppo.html)
