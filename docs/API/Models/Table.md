# [API Reference](../../API.md) - [Models](../Models.md) - Table

## Constructors

### new()

Creates a new base model object. If any of the arguments are nil, default argument values for that argument will be used.

```
Table.new(learningRate: number, Optimizer: Optimizer: Object, FeaturesList: {any}, ClassesList: {any}): ModelObject
```

#### Parameters:

* learningRate: The speed at which the algorithm learns. Recommended to set between 0 and 1.

* Optimizer: The optimizer object to be used.

* FeaturesList: A list containing all the features.

* ClassesList: A list containing all the classes. 

#### Returns:

* ModelObject: The generated model object.

## Functions

### train()

```
Table:predict(featureVector, labelVector)
```

### predict()

```
Table:predict(featureVector, returnOriginalOutput)
```

## Inherited From

* [BaseModel](BaseModel.md)
