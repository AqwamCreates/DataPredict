# [API Reference](../../API.md) - [Models](../Models.md) - TabularReinforcementLearningBaseModel

TabularReinforcementLearningBaseModel is a base class for tabular reinforcement learning models.

## Constructors

### new()

Creates a new base model object. If any of the arguments are nil, default argument values for that argument will be used.

```
TabularReinforcementLearningBaseModel.new(discountFactor: number): ModelObject
```

#### Parameters:

* StatesList: A list containing all the states.

* ActionList: A list containing all the actions. 

* discountFactor: The higher the value, the more likely it focuses on long-term outcomes. The value must be set between 0 and 1.

#### Returns:

* ModelObject: The generated model object.

## Functions

### predict()

```
TabularReinforcementLearningBaseModel:predict(stateVector, returnOriginalOutput)
```

### setCategoricalUpdateFunction()

Sets the model's categorical policy update function.

```
TabularReinforcementLearningBaseModel:setCategoricalUpdateFunction(categoricalUpdateFunction)
```

#### Parameters:

* categoricalUpdateFunction: The function to run when categoricalUpdate() is called.

### setEpisodeUpdateFunction()

Sets the model's episode update function.

```
TabularReinforcementLearningBaseModel:setEpisodeUpdateFunction(episodeUpdateFunction)
```

#### Parameters:

* episodeUpdateFunction: The function to run when episodeUpdate() is called.

### categoricalUpdate()

Updates the model parameters using categoricalUpdateFunction().

```
DeepReinforcementLearningBaseModel:categoricalUpdate(previousFeatureVector: featureVector, action: number/string, rewardValue: number, currentFeatureVector: featureVector, terminalStateValue: number)
```

#### Parameters:

* previousFeatureVector: The previous state of the environment.

* action: The action selected.

* rewardValue: The reward gained at current state.

* currentFeatureVector: The current state of the environment.

* terminalStateValue: A value of 1 indicates that the current state is a terminal state. A value of 0 indicates that the current state is not terminal.

### episodeUpdate()

Updates the model parameters using episodeUpdateFunction().

```
TabularReinforcementLearningBaseModel:episodeUpdate()
```

### setResetFunction()

Sets a new function on reset alongside with the current reset() function. 

```
TabularReinforcementLearningBaseModel:setResetFunction(resetFunction)
```

#### Parameters:

* resetFunction: The function to run when reset() is called.

### reset()

Reset model's stored values (excluding the parameters).

```
TabularReinforcementLearningBaseModel:reset()
```

## Inherited From

* [BaseModel](BaseModel.md)
