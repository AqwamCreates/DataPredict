# [API Reference](../../API.md) - [ExperienceReplays](../ExperienceReplays.md) - BaseExperienceReplay

The base class for most experience replay classes. It serves as a template for using it with the reinforcement learning models.

## Constructors

### new()

Creates a new base experience replay object.

```
BaseExperienceReplay.new(batchSize: number, numberOfExperienceToUpdate: number, maxBufferSize: number)
```

#### Parameters:

* batchSize: The number of experience to sample from for training.

* numberOfExperienceToUpdate: The number of experience needed for a single event of experience replay.

* maxBufferSize: The maximum number of experiences that can be kept inside the object.

### Functions

### setParameters()

Creates a new base experience replay object.

```
BaseExperienceReplay:setParametersbatchSize: number, numberOfExperienceToUpdate: number, maxBufferSize: number)
```

#### Parameters:

* batchSize: The number of experience to sample from for training.

* numberOfExperienceToUpdate: The number of experience needed for a single event of experience replay.

* maxBufferSize: The maximum number of experiences that can be kept inside the object.

