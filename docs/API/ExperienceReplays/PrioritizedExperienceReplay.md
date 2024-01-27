# [API Reference](../../API.md) - [ExperienceReplays](../ExperienceReplays.md) - PrioritizedExperienceReplay

It is used to update the models from experiences stored in the experience replay object. It boosts learning in reinforcement by focusing on important experiences, improving efficiency compared to regular replay.

## Constructors

### new()

Creates a new experience replay object.

```
PrioritizedExperienceReplay.new(batchSize: number, numberOfExperienceToUpdate: number, maxBufferSize: number, alpha: number, beta: number, aggregateFunction: string, epsilon: number)
```

#### Parameters:

* batchSize: The number of experience to sample from for training.

* numberOfExperienceToUpdate: The number of experience needed for a single event of experience replay.

* maxBufferSize: The maximum number of experiences that can be kept inside the object.

* alpha: Controls the degree of prioritization in sampling from the replay buffer. Must set the value between 0 and 1. 0 for uniform sampling, 1 for full prioritization.

* beta: Corrects the bias introduced by prioritization. Adjusts the importance sampling weights. Must set the value between 0 and 1. 1 for fully compensation.

* aggregateFunction: The function to apply to temporal difference error if it is a vector. The options are:

  * Maximum

  * Sum  

* epsilon: A number that prevents 0 priority. Recommended to set to very small values.

## Functions

### setParameters()

Change the parameters of an experience replay object.

```
PrioritizedExperienceReplay:setParametersbatchSize: number, numberOfExperienceToUpdate: number, maxBufferSize: number, alpha: number, beta: number, aggregateFunction: string, epsilon: number)
```

#### Parameters:

* batchSize: The number of experience to sample from for training.

* numberOfExperienceToUpdate: The number of experience needed for a single event of experience replay.

* maxBufferSize: The maximum number of experiences that can be kept inside the object.

* alpha: Controls the degree of prioritization in sampling from the replay buffer. Must set the value between 0 and 1. 0 for uniform sampling, 1 for full prioritization.

* beta: Corrects the bias introduced by prioritization. Adjusts the importance sampling weights. Must set the value between 0 and 1. 1 for fully compensation.

* aggregateFunction: The function to apply to temporal difference error if it is a vector. The options are:

  * Maximum

  * Sum  

* epsilon: A number that prevents 0 priority. Recommended to set to very small values.

## addModel()

* Adds a model to the experience replay object. Used for calculating priorities.

```
PrioritizedExperienceReplay:addModel()
```

## Parameters:

* Model: The model to be set.

## Inherited From

[BaseExperienceReplay](BaseExperienceReplay.md)

## Reference

[Prioritized ExperienceReplay By Tom Schaul, John Quan, Ioannis Antonoglou and David Silver (Google DeepMind)](https://arxiv.org/pdf/1511.05952.pdf)
