# [API Reference](../../API.md) - [Models](../Models.md) - Markov

## Constructors

### new()

Creates a new base model object. If any of the arguments are nil, default argument values for that argument will be used.

```
Markov.new(learningRate: number, isHidden: boolean, StatesList: {any}, ObservationsList: {any}): ModelObject
```

#### Parameters:

* learningRate: The speed at which the algorithm learns. Recommended to set between 0 and 1.

* isHidden: Set whether or not this Markov Model is a Hidden Markov Model.

* StatesList: A list containing all the states.

* ObservationsList: A list containing all the observations. 

#### Returns:

* ModelObject: The generated model object.

## Functions

### train()

```
Markov:train(previousStateVector, currentStateVector, observationStateVector)
```

### predict()

```
Markov:predict(stateVector, returnOriginalOutput)
```

## Inherited From

* [BaseModel](BaseModel.md)

## References

* [Hidden Markov Model](https://web.stanford.edu/~jurafsky/slp3/A.pdf)
