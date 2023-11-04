# [API Reference](../../API.md) - [Models](../Models.md) - DistributedLearning

DistributedLearning is a base class for distributed learning. The individual child model will calculate their own model parameters and these will create a new main model parameters using average.

## Notes:

* The child models must be created separately. Then use addModel() to put it inside the DistributedLearning object.

## Constructors

### new()

Create new model object. If any of the arguments are nil, default argument values for that argument will be used.

```
DistributedLearning.new(totalNumberOfChildModelUpdatesToUpdateMainModel: number): DistributedLearningObject
```

#### Parameters:

* totalNumberOfChildModelUpdatesToUpdateMainModel: The required total number of reinforce() and train() function calls from all child models to update the main model.

#### Returns:

* DistributedLearningObject: The generated distributed learning object.

## Functions

### setParameters()

Set model's parameters. When any of the arguments are nil, previous argument values for that argument will be used.

```
DistributedLearning:setParameters(totalNumberOfChildModelUpdatesToUpdateMainModel: number)
```

#### Parameters:

* totalNumberOfChildModelUpdatesToUpdateMainModel: The required total number of reinforce() and train() function calls from all child models to update the main model.

### addModel()

```
DistributedLearning:addModel(Model: ModelObject)
```

#### Parameters:

* Model: The child model to be added to main model.

### setMainModelParameters()

```
DistributedLearning:setMainModelParameters(MainModelParameters: any)
```

#### Parameters:

* MainModelParameters: The model parameters for the main model.

### getMainModelParameters()

```
DistributedLearning:getMainModelParameters(): any
```

#### Returns:

* MainModelParameters: The model parameters for the main model.

### reinforce()

Reward or punish model based on the current state of the environment.

```
DistributedLearning:reinforce(currentFeatureVector: Matrix, rewardValue: number, returnOriginalOutput: boolean, modelNumber: number): integer, number -OR- Matrix
```

#### Parameters:

* currentFeatureVector: Matrix containing data from the current state.

* rewardValue: The reward value added/subtracted from the current state (recommended value between -1 and 1, but can be larger than these values). 

* returnOriginalOutput: Set whether or not to return predicted vector instead of value with highest probability.

* modelNumber: The model number to be reinforced.

#### Returns:

* predictedLabel: A label that is predicted by the model.

* value: The value of predicted label.

-OR-

* predictedVector: A matrix containing all predicted values from all classes.

### getCurrentTotalNumberOfChildModelUpdatesToUpdateMainModel()

```
DistributedLearning:getCurrentTotalNumberOfChildModelUpdatesToUpdateMainModel(): number
```

#### Returns:

* getCurrentTotalNumberOfChildModelUpdatesToUpdateMainModel: The current total number of reinforcements from all child models.

### reset()

Reset the main model's stored values (excluding the parameters).

```
DistributedLearning:reset()
```

### destroy()

Destroys the model object.

```
DistributedLearning:destroy()
```
