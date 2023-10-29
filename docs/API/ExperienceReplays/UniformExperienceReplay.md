# [API Reference](../../API.md) - [ExperienceReplays](../ExperienceReplays.md) - UniformExperienceReplay

It is used to update the models from experiences stored in the experience replay object. All experience have equal chances of being chosen.

## Constructors

### new()

Creates a new experience replay object.

```
UniformExperienceReplay.new(batchSize: number, numberOfExperienceToUpdate: number, maxBufferSize: number)
```

#### Parameters:

* batchSize: The number of experience to sample from for training.

* numberOfExperienceToUpdate: The number of experience needed for a single event of experience replay.

* maxBufferSize: The maximum number of experiences that can be kept inside the object.

## Functions

### setParameters()

Change the parameters of an experience replay object.

```
UniformExperienceReplay:setParametersbatchSize: number, numberOfExperienceToUpdate: number, maxBufferSize: number)
```

#### Parameters:

* batchSize: The number of experience to sample from for training.

* numberOfExperienceToUpdate: The number of experience needed for a single event of experience replay.

* maxBufferSize: The maximum number of experiences that can be kept inside the object.

### Inherited From

[BaseExperienceReplay](../BaseExperienceReplay.md)
