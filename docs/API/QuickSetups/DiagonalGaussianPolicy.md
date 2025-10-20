# [API Reference](../../API.md) - [QuickSetups](../QuickSetups.md) - SimpleDiagonalGaussianPolicy

SimpleDiagonalGaussianPolicy is a base class for setuping up reinforcement learning functions.

## Constructors

### new()

Create new model object. If any of the arguments are nil, default argument values for that argument will be used.

```
SimpleDiagonalGaussianPolicy.new(numberOfReinforcementsPerEpisode: integer): DiagonalGaussianPolicyObject
```

#### Parameters:

* numberOfReinforcementsPerEpisode: The number of reinforcements to be considered as a single episode.

#### Returns:

* ModelObject: The generated model object.

## Functions

### setParameters()

Set model's parameters. When any of the arguments are nil, previous argument values for that argument will be used.

```
SimpleDiagonalGaussianPolicy:setParameters(numberOfReinforcementsPerEpisode: integer)
```

#### Parameters:

* numberOfReinforcementsPerEpisode: The number of reinforcements to decay the epsilon value.

### setModel()

```
SimpleDiagonalGaussianPolicy:setModel(Model: ModelObject)
```

#### Parameters:

* Model: The model object.

### getModel()

```
SimpleDiagonalGaussianPolicy:getModel(): ModelObject
```

#### Returns:

* Model: The model object.

### extendUpdateFunction()

Sets a new function on update alongside with the current model's update() function. 

```
SimpleDiagonalGaussianPolicy:extendUpdateFunction(updateFunction)
```

#### Parameters:

* updateFunction: The function to run after calling the model's update() function

### extendEpisodeUpdateFunction()

Sets a new function on episode update alongside with the current model's episodeUpdate() function. 

```
DiagonalGaussianPolicy:extendEpisodeUpdateFunction(episodeUpdateFunction)
```

#### Parameters:

* episodeUpdateFunction: The function to run after calling the model's episodeUpdate() function

### reinforce()

Reward or punish model based on the current state of the environment.

```
SimpleDiagonalGaussianPolicy:reinforce(currentFeatureVector: matrix, actionStandardDeviationVector: matrix, rewardValue: number): matrix
```

#### Parameters:

* currentFeatureVector: Matrix containing data from the current state.

* actionStandardDeviationVector: The vector containing values of action's standard deviations. The number of columns must match the number of actions.

* rewardValue: The reward value added/subtracted from the current state (recommended value between -1 and 1, but can be larger than these values). 

#### Returns:

* predictedVector: A matrix containing all predicted values from all classes.

### reset()

Resets the current parameters values.

```
SimpleDiagonalGaussianPolicy:reset()
```

### setPrintOutput()

Set whether or not to show the current number of episodes and current epsilon.

```
SimpleDiagonalGaussianPolicy:setPrintOutput(option: boolean)
```

#### Parameters:

* option: A boolean value that determines the reinforcement output to be printed or not.

### getCurrentNumberOfEpisodes()

```
SimpleDiagonalGaussianPolicy:getCurrentNumberOfEpisodes(): integer
```

#### Returns

* currentNumberOfEpisodes: The current number of episode stored inside the reinforcement learning quick setup object.

### getCurrentNumberOfReinforcements()

```
SimpleDiagonalGaussianPolicy:getCurrentNumberOfReinforcements(): integer
```

#### Returns

* currentNumberOfReinforcements: The current number of times reinforce() has been called stored inside the reinforcement learning quick setup object.

## Inherited From

* [DiagonalGaussianPolicyBaseQuickSetup](DiagonalGaussianPolicyBaseQuickSetup.md)
