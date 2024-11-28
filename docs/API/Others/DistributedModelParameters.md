# [API Reference](../../API.md) - [Others](../Others.md) - DistributedModelParameters

DistributedTraining is a base class for distributed learning. The individual child models will calculate their own model parameters and these will create a new main model parameters using average.

## Notes:

* The child models must be created separately. Then use addModel() to put it inside the DistributedLearning object.

## Constructors

### new()

Create new model object. If any of the arguments are nil, default argument values for that argument will be used.

```
DistributedModelParameters.new(totalNumberOfChildModelUpdatesToUpdateMainModel: number): DistributedLearningObject
```

#### Parameters:

* totalNumberOfChildModelUpdatesToUpdateMainModel: The required total number of reinforce() and train() function calls from all child models to update the main model.

#### Returns:

* DistributedLearningObject: The generated distributed learning object.

## Functions

### setParameters()

Set model's parameters. When any of the arguments are nil, previous argument values for that argument will be used.

```
DistributedModelParameters:setParameters(totalNumberOfChildModelUpdatesToUpdateMainModel: number)
```

#### Parameters:

* totalNumberOfChildModelUpdatesToUpdateMainModel: The required total number of reinforce() and train() function calls from all child models to update the main model.

### addModel()

```
DistributedModelParameters:addModelParameters(ModelParameters: Matrix/TableOfMatrices)
```

#### Parameters:

* ModelParameters: The model parameters to be received by the DistributedModelParameters.

### setModelParametersMerger()

Sets the ModelParametersMerger into the DistributedModelParameters.

```
DistributedModelParameters:setModelParametersMerger(ModelParametersMerger: ModelParametersMergerObject)
```

ModelParametersMerger: A ModelParametersMerger object to be used by the DistributedModelParameters object

### setMainModelParameters()

```
DistributedModelParameters:setMainModelParameters(MainModelParameters: any)
```

#### Parameters:

* MainModelParameters: The model parameters for the main model.

### getMainModelParameters()

```
DistributedModelParameters:getMainModelParameters(): any
```

#### Returns:

* MainModelParameters: The model parameters for the main model.

### getCurrentTotalNumberOfChildModelUpdatesToUpdateMainModel()

```
DistributedModelParameters:getCurrentTotalNumberOfChildModelUpdatesToUpdateMainModel(): number
```

#### Returns:

* getCurrentTotalNumberOfChildModelUpdatesToUpdateMainModel: The current total number of reinforcements from all child models.

### reset()

Reset the main model's stored values (excluding the parameters).

```
DistributedModelParameters:reset()
```

### destroy()

Destroys the model object.

```
DistributedModelParameters:destroy()
```
