# [API Reference](../../API.md) - [Models](../Models.md) - DeepReinforcementLearningBaseModel

DeepReinforcementLearningBaseModel is a base class for reinforcement learning neural network models.

## Constructors

### new()

Creates a new base model object. If any of the arguments are nil, default argument values for that argument will be used.

```
DeepReinforcementLearningBaseModel.new(discountFactor: number): ModelObject
```

#### Parameters:

* discountFactor: The higher the value, the more likely it focuses on long-term outcomes. The value must be set between 0 and 1.

#### Returns:

* ModelObject: The generated model object.

## Functions

### setParameters()

Set model's parameters. When any of the arguments are nil, previous argument values for that argument will be used.

```
DeepReinforcementLearningBaseModel:setParameters(discountFactor: number)
```

### setModel()

Sets the model.

```
DeepReinforcementLearningBaseModel:setModel(Model: ModelObject)
```

#### Parameters:

* Model: The model to be used.

### getModel()

Gets the model.

```
DeepReinforcementLearningBaseModel:getModel(): ModelObject
```

#### Returns:

* Model: The model that was set.

### setCategoricalUpdateFunction()

Sets the model's categorical policy update function.

```
DeepReinforcementLearningBaseModel:setCategoricalUpdateFunction(categoricalUpdateFunction)
```

#### Parameters:

* categoricalUpdateFunction: The function to run when categoricalUpdate() is called.

### setDiagonalGaussianUpdateFunction()

Sets the model's diagonal Gausian policy update function.

```
DeepReinforcementLearningBaseModel:setDiagonalGaussianUpdateFunction(diagonalGaussianUpdateFunction)
```

#### Parameters:

* diagonalGaussianUpdateFunction: The function to run when diagonalGaussianUpdate() is called.

### setEpisodeUpdateFunction()

Sets the model's episode update function.

```
DeepReinforcementLearningBaseModel:setEpisodeUpdateFunction(episodeUpdateFunction)
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

### diagonalGaussianUpdate()

Updates the model parameters using diagonalGaussianUpdateFunction().

```
DeepReinforcementLearningBaseModel:diagonalGaussianUpdate(previousFeatureVector: featureVector, actionMeanVector: vector, actionStandardDeviationVector, rewardValue: number, currentFeatureVector: featureVector, terminalStateValue: number)
```

#### Parameters:

* previousFeatureVector: The previous state of the environment.

* actionMeanVector: The vector containing mean values for all actions.

* actionStandardDeviationVector: The vector containing standard deviation values for all actions.

* rewardValue: The reward gained at current state.

* currentFeatureVector: The current state of the environment.

* terminalStateValue: A value of 1 indicates that the current state is a terminal state. A value of 0 indicates that the current state is not terminal.

### episodeUpdate()

Updates the model parameters using episodeUpdateFunction().

```
DeepReinforcementLearningBaseModel:episodeUpdate()
```

### setResetFunction()

Sets a new function on reset alongside with the current reset() function. 

```
DeepReinforcementLearningBaseModel:setResetFunction(resetFunction)
```

#### Parameters:

* resetFunction: The function to run when reset() is called.

### reset()

Reset model's stored values (excluding the parameters).

```
DeepReinforcementLearningBaseModel:reset()
```
