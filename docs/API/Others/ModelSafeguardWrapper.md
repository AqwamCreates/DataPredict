# [API Reference](../../API.md) - [Others](../Others.md) - ModelSafeguardWrapper

## Constructors

### new()

Creates a new safeguard wrapper object. If any of the arguments are not given, default argument values for that argument will be used.

```
ModelSafeguardWrapper.new(Model: ModelObject): ModelSafeguardWrapperObject
```

#### Parameters:

* Model: The model object to modify its training capabilities.

#### Returns:

* TrainingModifierObject: A training modifier object that uses the model's train(), predict() and reinforce() functions so that it behaves like a regular model.

## Functions

### train()

Trains the machine/deep learning model under specific gradient descent mode.

```
ModelSafeguardWrapper:train(...): number[]
```

#### Parameters:

* ...: The parameters are the same to the original model's train() function.

#### Returns:

* costArray: An array containing cost values.

### predict()

Predict the values for given data.

```
ModelSafeguardWrapper:predict(...): ...
```

#### Parameters:

...: The parameters are the same to the original model's predict() function.

#### Returns:

...: The outputs are the same to the original model's predict() function.

### getModelParameters()

Gets the model parameters from the base model.

```
ModelSafeguardWrapper:getModelParameters(doNotDeepCopy: boolean): ModelParameters
```

#### Parameters

* doNotDeepCopy: Set whether or not to deep copy the model parameters.

#### Returns

* ModelParameters: A matrix/table containing model parameters fetched from the base model.

### setModelParameters()

Set the model parameters to the base model.

```
ModelSafeguardWrapper:setModelParameters(ModelParameters: ModelParameters, doNotDeepCopy: boolean)
```

#### Parameters

* ModelParameters: A matrix/table containing model parameters to be given to the base model.

* doNotDeepCopy: Set whether or not to deep copy the model parameters.

## Inherited From

* [BaseInstance](../Cores/BaseInstance.md)
