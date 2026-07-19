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

#### Parameters:

* previousStateMatrix: A matrix containing all previous state data.

* currentStateMatrix: A matrix containing all current state data.

### predict()

Predict the values for given data.

```
Markov:predict(previousStateMatrix: matrix, returnOriginalOutput: boolean): matrix, matrix -OR- matrix
```

#### Parameters:

* previousStateMatrix: A matrix containing all previous state data.

* returnOriginalOutput: Set whether or not to return predicted current state matrix instead of value with highest probability. 

#### Returns:

* predictedVector: A vector that is predicted by the model.

* probabilityVector: A vector that contains the probability of predicted values in predictedVector.

-OR-

* predictedCurrentStateMatrix: A matrix containing all predicted values from all classes.

## Inherited From

* [BaseModel](BaseModel.md)

## References

* [Hidden Markov Model](https://web.stanford.edu/~jurafsky/slp3/A.pdf)
