# [API Reference](../../API.md) - [Models](../Models.md) - ReinforcementLearningDeepDuelingQLearningBaseModel

ReinforcementLearningDeepDuelingQLearningBaseModel is a base class for reinforcement learning neural network models.

## Constructors

### new()

Creates a new base model object. If any of the arguments are nil, default argument values for that argument will be used.

```
ReinforcementLearningDeepDuelingQLearningBaseModel.new(discountFactor: number): ModelObject
```

#### Parameters:

* discountFactor: The higher the value, the more likely it focuses on long-term outcomes. The value must be set between 0 and 1.

#### Returns:

* ModelObject: The generated model object.

## Functions

### setParameters()

Set model's parameters. When any of the arguments are nil, previous argument values for that argument will be used.

```
ReinforcementLearningDeepDuelingQLearningBaseModel:setParameters(discountFactor: number)
```

#### Parameters:

* discountFactor: The higher the value, the more likely it focuses on long-term outcomes. The value must be set between 0 and 1.

### setAdvantageModel()

Sets the advantage model. The outputs of the actor model is required to be in normal distribution format.

```
ReinforcementLearningDeepDuelingQLearningBaseModel:setAdvantageModel(Model: ModelObject)
```

#### Parameters:

* Model: The model to be used as an advantage model.

### setValueModel()

Sets the critic model.

```
ReinforcementLearningDeepDuelingQLearningBaseModel:setValueModel(Model: ModelObject)
```

#### Parameters:

* Model: The model to be used as a value model.

### getAdvantageModel()

Gets the advantage model.

```
ReinforcementLearningDeepDuelingQLearningBaseModel:getAdvantageModel(): ModelObject
```

#### Returns:

* Model: The model that was used as an advantage model.

### getValueModel()

Gets the value model.

```
ReinforcementLearningDeepDuelingQLearningBaseModel:getValueModel(): ModelObject
```

#### Returns:

* Model: The model that was used as a value model.

### setUpdateFunction()

Sets the model's update function.

```
ReinforcementLearningDeepDuelingQLearningBaseModel:setUpdateFunction(updateFunction)
```

#### Parameters:

* updateFunction: The function to run when update() is called.

### setEpisodeUpdateFunction()

Sets the model's episode update function.

```
ReinforcementLearningDeepDuelingQLearningBaseModel:setEpisodeUpdateFunction(episodeUpdateFunction)
```

#### Parameters:

* episodeUpdateFunction: The function to run when episodeUpdate() is called.

### update()

Updates the model parameters using updateFunction().

```
ReinforcementLearningDeepDuelingQLearningBaseModel:update(previousFeatiureVector: featureVector, action: number/string, rewardValue: number, currentFeatureVector: featureVector)
```

#### Parameters:

* previousFeatiureVector: The previous state of the environment.

* action: The action selected.

* rewardValue: The reward gained at current state.

* currentFeatureVector: The currrent state of the environment.

### episodeUpdate()

Updates the model parameters using episodeUpdateFunction().

```
ReinforcementLearningDeepDuelingQLearningBaseModel:episodeUpdate()
```

### extendResetFunction()

Sets a new function on reset alongside with the current reset() function. 

```
ReinforcementLearningDeepDuelingQLearningBaseModel:extendResetFunction(resetFunction)
```

#### Parameters:

* resetFunction: The function to run when reset() is called.

### reset()

Reset model's stored values (excluding the parameters).

```
ReinforcementLearningDeepDuelingQLearningBaseModel:reset()
```
