# [API Reference](../../API.md) - [AqwamCustomModels](../AqwamCustomModels.md) - AqwamAdvantageActorCritic (Aqwam A2C)

AqwamAdvantageActorCritic is a base class for reinforcement learning. 

It is more sample efficient than the AdvantageActorCritic model, but its mathematical proof might be incorrect or non-existent. 

Aqwam's version of AdvantageActorCritic have two types of actions: sampled action and selected action. An AI will act on "selected action" that has highest value but the AI will reward on "sampled action". 

The theory is that the when a "sampled action" is associated with a positive reward, it pushes the probability to choose the "sampled action" higher. This leads to "sampled action" more likely to become "selected action". This also allows the algorithm not requiring additional exploration techniques. 

Meanwhile, the original A2C model randomly samples actions for "selected action" instead of choosing action with highest values.

The algorithm was found by accident when I first incorrectly implemented the AdvantageActorCritic model.

## Notes

* The Actor and Critic models must be created separately. Then use setActorModel() and setCriticModel() to put it inside the AqwamAdvantageActorCritic model.

* Actor and Critic models must be a part of NeuralNetwork model. If you decide to use linear regression or logistic regression, then it must be constructed using NeuralNetwork model. 

* Ensure the final layer of the Critic model has only one neuron. It is the default setting for all Critic models in research papers.

* Ensure you choose actions that have highest values for best results.

## Constructors

### new()

Create new model object. If any of the arguments are nil, default argument values for that argument will be used.

```
AqwamAdvantageActorCritic.new(discountFactor: number): ModelObject
```

#### Parameters:

* discountFactor: The higher the value, the more likely it focuses on long-term outcomes. The value must be set between 0 and 1.

#### Returns:

* ModelObject: The generated model object.

## Functions

### setParameters()

Set model's parameters. When any of the arguments are nil, previous argument values for that argument will be used.

```
AqwamAdvantageActorCritic:setParameters(discountFactor: number)
```

#### Parameters:

* discountFactor: The higher the value, the more likely it focuses on long-term outcomes. The value must be set between 0 and 1.

## Inherited From

* [ReinforcementLearningActorCriticBaseModel](ReinforcementLearningActorCriticBaseModel.md)

## References

* [Asynchronous Methods for Deep Reinforcement Learning By Volodymyr Mnih, Adrià Puigdomènech Badia, Mehdi Mirza, Alex Graves, Timothy P. Lillicrap, Tim Harley, David Silver, Koray Kavukcuoglu](https://arxiv.org/pdf/1602.01783v2.pdf)

* [Advantage Actor Critic (A2C) implementation by Alvaro Durán Tovar](https://medium.com/deeplearningmadeeasy/advantage-actor-critic-a2c-implementation-944e98616b)

* [Actor Critic Method by Apoorv Nandan - Keras](https://keras.io/examples/rl/actor_critic_cartpole/)
