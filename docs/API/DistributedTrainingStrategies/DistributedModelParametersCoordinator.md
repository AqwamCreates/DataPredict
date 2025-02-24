# [API Reference](../../API.md) - [DistributedTrainingStrategies](../DistributedTrainingStrategies.md) - DistributedModelParameters

DistributedModelParametersCoordinator is a base class for distributed learning. The individual child models will calculate their own model parameters and these will create a new main model parameters using average.

## Notes:

* The child models must be created separately. Then use addModel() to put it inside the DistributedLearning object.

## Constructors

### new()

Create new model object. If any of the arguments are nil, default argument values for that argument will be used.

```
DistributedModelParametersCoordinator.new(totalNumberOfChildModelUpdatesToUpdateMainModel: number): DistributedLearningObject
```

#### Parameters:

* totalNumberOfChildModelUpdatesToUpdateMainModel: The required total number of reinforce() and train() function calls from all child models to update the main model.

#### Returns:

* DistributedModelParametersCoordinatorObject: The generated distributed model parameters coordinator object.

## Functions

### setParameters()

Set model's parameters. When any of the arguments are nil, previous argument values for that argument will be used.

```
DistributedModelParametersCoordinator:setParameters(totalNumberOfChildModelUpdatesToUpdateMainModel: number)
```

#### Parameters:

* totalNumberOfChildModelUpdatesToUpdateMainModel: The required total number of reinforce() and train() function calls from all child models to update the main model.

### addModelParameters()

```
DistributedModelParametersCoordinator:addModelParameters(ModelParameters: Matrix/TableOfMatrices)
```

#### Parameters:

* ModelParameters: The model parameters to be received by the DistributedModelParameters.

### setModelParametersMerger()

Sets the ModelParametersMerger into the DistributedModelParameters.

```
DistributedModelParametersCoordinator:setModelParametersMerger(ModelParametersMerger: ModelParametersMergerObject)
```

ModelParametersMerger: A ModelParametersMerger object to be used by the DistributedModelParameters object

### setMainModelParameters()

```
DistributedModelParametersCoordinator:setMainModelParameters(MainModelParameters: any)
```

#### Parameters:

* MainModelParameters: The model parameters for the main model.

### getMainModelParameters()

```
DistributedModelParametersCoordinator:getMainModelParameters(): any
```

#### Returns:

* MainModelParameters: The model parameters for the main model.

### getCurrentTotalNumberOfChildModelUpdatesToUpdateMainModel()

```
DistributedModelParametersCoordinator:getCurrentTotalNumberOfChildModelUpdatesToUpdateMainModel(): number
```

#### Returns:

* getCurrentTotalNumberOfChildModelUpdatesToUpdateMainModel: The current total number of reinforcements from all child models.

### start()

Creates a new thread for real-time gradient descent / ascent.

```
DistributedModelParametersCoordinator:start(): coroutine
```

#### Returns:

* modelParameterChangeCoroutine: A coroutine that handles the modification of the model parameters.

### reset()

Reset the main model's stored values (excluding the parameters).

```
DistributedModelParametersCoordinator:reset()
```

#### Inherited From:

* [BaseInstance]()
