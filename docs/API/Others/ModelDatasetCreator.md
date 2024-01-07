# [API Reference](../../API.md) - [Others](../Others.md) - ModelDatasetCreator

Creates a dataset for models to train on

## Constructors

### new()

Creates a new gradient descent modifier object. If any of the arguments are not given, default argument values for that argument will be used.

```
ModelDatasetCreator.new(): ModelDatasetCreatorObject
```

## Functions

### setDatasetSplitPercentages()

Set the split percentages for training, validation and testing. The current default values are 0.7 for training and 0.3 for testing.

```
ModelDatasetCreator:setDatasetSplitPercentages(trainDataPercentage: number, validationDataPercentage: number, testDataPercentage: number)
```

#### Parameters:

* trainDataPercentage: The percentage of dataset to be turned to training data. Must convert the percentage to its decimal form first.

* validationDataPercentage: The percentage of dataset to be turned to validation data. Must convert the percentage to its decimal form first.

* testDataPercentage: The percentage of dataset to be turned to testing data. Must convert the percentage to its decimal form first.

### train()

Trains the machine/deep learning model under specific gradient descent mode.

```
GradientDescentModifier:train(...): number[]
```

#### Parameters:

* ...: The parameters are the same to the original model's train() function.

#### Returns:

* costArray: An array containing cost values.

### predict()

Predict the values for given data.

```
GradientDescentModifier:predict(...): ...
```

#### Parameters:

...: The parameters are the same to the original model's predict() function.

#### Returns:

...: The outputs are the same to the original model's predict() function.

### reinforce()

Reward or punish model based on the current state of the environment.

```
ActorCritic:reinforce(currentFeatureVector: Matrix, rewardValue: number, returnOriginalOutput: boolean): integer, number -OR- Matrix
```

#### Parameters:

* currentFeatureVector: Matrix containing data from the current state.

* rewardValue: The reward value added/subtracted from the current state (recommended value between -1 and 1, but can be larger than these values). 

* returnOriginalOutput: Set whether or not to return predicted vector instead of value with highest probability.
