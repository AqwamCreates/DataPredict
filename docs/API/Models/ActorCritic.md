# [API Reference](../../API.md) - [Models](../Models.md) - ActorCritic

ActorCritic is a base class for reinforcement learning.

## Notes:

* The Actor and Critic models must be created separately. Then use setActorModel() and setCriticModel() to put it inside the ActorCritic model.

* Actor and Critic models must be a part of NeuralNetwork model. If you decide to use linear regression or logistic regression, then it must be constructed using NeuralNetwork model. 

* Ensure the final layer of the Critic model has only one neuron. It is the default setting for all Critic models in research papers.

## Constructors

### new()

Create new model object. If any of the arguments are nil, default argument values for that argument will be used.

```
ActorCritic.new(discountFactor: number): ModelObject
```

#### Parameters:

* discountFactor: The higher the value, the more likely it focuses on long-term outcomes. The value must be set between 0 and 1.

#### Returns:

* ModelObject: The generated model object.

## Functions

### setParameters()

Set model's parameters. When any of the arguments are nil, previous argument values for that argument will be used.

```
ActorCritic:setParameters(discountFactor: number)
```

#### Parameters:

* discountFactor: The higher the value, the more likely it focuses on long-term outcomes. The value must be set between 0 and 1.

## Inherited From

* [ReinforcementLearningActorCriticNeuralNetworkBaseModel](ReinforcementLearningActorCriticNeuralNetworkBaseModel.md)

## References

* [Actor Critic Method by Apoorv Nandan - Keras](https://keras.io/examples/rl/actor_critic_cartpole/)
