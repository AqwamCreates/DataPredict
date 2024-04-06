# [API Reference](../../API.md) - [Others](../Others.md) - DistributedGradients

DistributedGradients is a base class for distributed gradient ascent / descent.

## Constructors

### new()

Create new model object. If any of the arguments are nil, default argument values for that argument will be used.

```
DistributedGradients.new(gradientChangeMode: string): DistributedGradientObject
```

#### Parameters:

* gradientChangeMode: Set what to do with the model parameters for a given gradient. Available options are:

  * Descent (Default)

  * Ascent  

#### Returns:

* DistributedLearningObject: The generated distributed learning object.

## Functions

### setParameters()

Set model's parameters. When any of the arguments are nil, previous argument values for that argument will be used.

```
DistributedGradients:setParameters(gradientChangeMode: string)
```

#### Parameters:

* gradientChangeMode: Set what to do with the model parameters for a given gradient. Available options are:

  * Descent

  * Ascent  

### addGradients()

```
DistributedGradients:addGradients(Gradients: any)
```

#### Parameters:

* Gradients: The table of matrices / matrix containing the gradient values.

### setModelParameters()

```
DistributedGradients:setMainModelParameters(ModelParameters: any, doNotDeepCopy: boolean)
```

#### Parameters:

* ModelParameters: The model parameters for the main model.

* doNotDeepCopy: Set whether or not to deep copy the model parameters.

### getModelParameters()

```
DistributedGradients:getModelParameters(doNotDeepCopy: boolean): any
```

#### Returns:

* ModelParameters: The model parameters for the main model.

* doNotDeepCopy: Set whether or not to deep copy the model parameters.

### start()

Creates a new thread for real-time gradient descent / ascent.

```
ModelParameters:start(): coroutine
```

#### Returns:

* gradientChangeCoroutine: A coroutine that handles the modification of the model parameters.

### stop()

Stops the threads for real-time training.

```
ModelParameters:stop()
```

### clearGradients()

Clears the stored gradients inside the DistributedGradients object.

```
DistributedGradients:clearGradients()
```

### destroy()

Destroys the model object.

```
DistributedGradients:destroy()
```
