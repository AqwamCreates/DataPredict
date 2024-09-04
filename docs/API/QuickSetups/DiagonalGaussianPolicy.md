# [API Reference](../../API.md) - [QuickSetups](../QuickSetups.md) - DiagonalGaussianPolicy

DiagonalGaussianPolicy is a base class for setuping up reinforcement learning functions.

## Constructors

### new()

Create new model object. If any of the arguments are nil, default argument values for that argument will be used.

```
DiagonalGaussianPolicy.new(numberOfReinforcementsPerEpisode: integer): DiagonalGaussianPolicyObject
```

#### Parameters:

* numberOfReinforcementsPerEpisode: The number of reinforcements to decay the epsilon value.

#### Returns:

* ModelObject: The generated model object.

## Functions

### setParameters()

Set model's parameters. When any of the arguments are nil, previous argument values for that argument will be used.

```
DiagonalGaussianPolicy:setParameters(numberOfReinforcementsPerEpisode: integer)
```

#### Parameters:

* numberOfReinforcementsPerEpisode: The number of reinforcements to decay the epsilon value.

### setModel()

```
DiagonalGaussianPolicy:setModel(Model: ModelObject)
```

#### Parameters:

* Model: The model object.

### getModel()

```
DiagonalGaussianPolicy:getModel(): ModelObject
```

#### Returns:

* Model: The model object.

### setClassesList()

```
DiagonalGaussianPolicy:setClassesList(classesList: [])
```

#### Parameters:

* classesList: A list of classes. The index of the class relates to which the neuron at output layer belong to. For example, {3, 1} means that the output for 3 is at first neuron, and the output for 1 is at second neuron.

### getClassesList()

Gets all the classes stored in the NeuralNetwork model.

```
DiagonalGaussianPolicy:getClassesList(): []
```

#### Returns:

* classesList: A list of classes. The index of the class relates to which the neuron at output layer belong to. For example, {3, 1} means that the output for 3 is at first neuron, and the output for 1 is at second neuron.

### extendUpdateFunction()

Sets a new function on update alongside with the current model's update() function. 

```
DiagonalGaussianPolicy:extendUpdateFunction(updateFunction)
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
DiagonalGaussianPolicy:reinforce(currentFeatureVector: Matrix, rewardValue: number, returnOriginalOutput: boolean, childModelNumber: integer): integer, number -OR- Matrix
```

#### Parameters:

* currentFeatureVector: Matrix containing data from the current state.

* rewardValue: The reward value added/subtracted from the current state (recommended value between -1 and 1, but can be larger than these values). 

* childModelNumber: The child model number to be selected from the parent model.

#### Returns:

* predictedVector: A matrix containing all predicted values from all classes.

### reset()

Resets the current parameters values.

```
DiagonalGaussianPolicy:reset()
```

### setPrintOutput()

Set whether or not to show the current number of episodes and current epsilon.

```
DiagonalGaussianPolicy:setPrintOutput(option: boolean)
```

#### Parameters:

* option: A boolean value that determines the reinforcement output to be printed or not.

### getCurrentNumberOfEpisodes()

```
DiagonalGaussianPolicy:getCurrentNumberOfEpisodes(): integer
```

#### Returns

* currentNumberOfEpisodes: The current number of episode stored inside the reinforcement learning quick setup object.

### getCurrentNumberOfReinforcements()

```
DiagonalGaussianPolicy:getCurrentNumberOfReinforcements(): integer
```

#### Returns

* currentNumberOfReinforcements: The current number of times reinforce() has been called stored inside the reinforcement learning quick setup object.
