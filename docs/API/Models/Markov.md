# [API Reference](../../API.md) - [Models](../Models.md) - Markov

## Constructors

### new()

Creates a new base model object. If any of the arguments are nil, default argument values for that argument will be used.

```
Markov.new(): ModelObject
```

#### Parameters:

* StatesList: A list containing all the states.

* ObservationsList: A list containing all the observations. 

#### Returns:

* ModelObject: The generated model object.

## Functions

### predict()

```
Markov:predict(stateVector, returnOriginalOutput)
```

## Inherited From

* [BaseModel](BaseModel.md)
