# [API Reference](../../API.md) - [Models](../Models.md) - BaseModel

The base model for all machine and deep learning models.

## Constructors

### new()

Creates a new machine learning base model.

```
BaseModel.new(): BaseModelObject
```

## Functions

### getModelParameters()

Gets the model parameters from the base model.

```
BaseModel:getModelParameters(doNotDeepCopy: boolean): ModelParameters
```

#### Parameters

* doNotDeepCopy: Set whether or not to deep copy the model parameters.

#### Returns

* ModelParameters: A matrix/table containing model parameters fetched from the base model.

### setModelParameters()

Set the model parameters to the base model.

```
BaseModel:setModelParameters(ModelParameters: ModelParameters, doNotDeepCopy: boolean)
```

#### Parameters

* ModelParameters: A matrix/table containing model parameters to be given to the base model.

* doNotDeepCopy: Set whether or not to deep copy the model parameters.

### clearModelParameters()

Clears the model parameters contained inside base model.

```
BaseModel:clearModelParameters()
```

### setPrintOutput()

Set if the model prints output.

```
BaseModel:setPrintOutput(option: boolean)
```

#### Parameters:

* option: A boolean value that specifies if the output is printed.
