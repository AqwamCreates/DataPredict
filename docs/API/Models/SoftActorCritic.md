# [API Reference](../../API.md) - [Models](../Models.md) - SoftActorCritic

SoftActorCritic is a base class for reinforcement learning.

## Notes

* The Actor and Critic models must be created separately. Then use setActorModel() and setCriticModel() to put it inside the ActorCritic model.

* Actor and Critic models must be a part of NeuralNetwork model. If you decide to use linear regression or logistic regression, then it must be constructed using NeuralNetwork model. 

* Ensure the final layer of the Critic model has only one neuron. It is the default setting for all Critic models in research papers.

## Constructors

### new()

Create new model object. If any of the arguments are nil, default argument values for that argument will be used.

```
SoftActorCritic.new(alpha: number, averagingRate: number, discountFactor: number): ModelObject
```

#### Parameters:

* alpha: Entropy regularization coefficient. The higher the value, the more the model explores. Generally the value is set between 0 and 1. [Default: 0.1]

* averagingRate: The higher the value, the faster the weights changes. The value must be set between 0 and 1. [Default: 0.995]

* discountFactor: The higher the value, the more likely it focuses on long-term outcomes. The value must be set between 0 and 1. [Default: 0.95]

#### Returns:

* ModelObject: The generated model object.

## Inherited From

* [ReinforcementLearningActorCriticBaseModel](ReinforcementLearningActorCriticBaseModel.md)

## References

* [Soft Actor Critic By OpenAI](https://spinningup.openai.com/en/latest/algorithms/sac.html)
