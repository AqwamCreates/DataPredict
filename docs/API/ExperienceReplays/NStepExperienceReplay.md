# [API Reference](../../API.md) - [ExperienceReplays](../ExperienceReplays.md) - NStepExperienceReplay

It is used to update the models from experiences stored in the experience replay object. It uses longer experience sequences to enhance reinforcement learning.

## Constructors

### new()

Creates a new PrioritizedExperienceReplay object.

```
PrioritizedExperienceReplay.new(batchSize: number, numberOfRunsToUpdate: number, maxBufferSize: number, nStep: number)
```

#### Parameters:

* batchSize: The number of experience to sample from for training.

* numberOfRunsToUpdate: The number of run() function needed to be called to run a single event of experience replay.

* maxBufferSize: The maximum number of experiences that can be kept inside the object.

* nStep: The maximum length of experience sequences to be sampled.

## Functions

### setParameters()

Change the parameters of an experience replay object.

```
PrioritizedExperienceReplay:setParametersbatchSize: number, numberOfRunsToUpdate: number, maxBufferSize: number, nStep: number)
```

#### Parameters:

* batchSize: The number of experience to sample from for training.

* numberOfRunsToUpdate: The number of run() function needed to be called to run a single event of experience replay.

* maxBufferSize: The maximum number of experiences that can be kept inside the object.

* nStep: The maximum length of experience sequences to be sampled.

## Inherited From

[BaseExperienceReplay](BaseExperienceReplay.md)
