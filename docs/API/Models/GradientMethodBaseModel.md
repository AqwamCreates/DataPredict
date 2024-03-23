# [API Reference](../../API.md) - [Models](../Models.md) - BaseModel

The base model for all machine and deep learning models that uses gradient descent

## Constructors

### new()

Creates a new gradient method base model.

```
GradientMethodBaseModel.new(): GradientMethodBaseModelObject
```

## Functions

### setAreGradientsSaved()

```
GradientMethodBaseModel:setAreGradientSaved(option: boolean)
```

#### Parameters:

* option: Set whether or not to store gradients.

### getGradients()

```
GradientMethodBaseModel:getGradients(doNotDeepCopy: boolean): any
```

#### Parameters:

* doNotDeepCopy: Set whether or not to get a deep copy of the gradients.

#### Returns:

* Gradients: The gradient stored inside the model.

### setGradients()

```
GradientMethodBaseModel:setGradients(Gradients: any, doNotDeepCopy: boolean)
```

#### Parameters:

* Gradients: The gradient to be stored by the model.

* doNotDeepCopy: Set whether or not to get a deep copy of the gradients.

### clearGradients()

Clears stored gradients inside the gradient method base model.

```
GradientMethodBaseModel:clearGradients()
```
