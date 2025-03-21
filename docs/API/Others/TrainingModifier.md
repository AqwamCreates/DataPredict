# [API Reference](../../API.md) - [Others](../Others.md) - TrainingModifier

Modifies the model's batch training to other modes.

## Notes

* When using "Stochastic" mode, make sure you set the model's max number of iterations to 1.

* batchSize is only applicable for "Minibatch" mode.

## Constructors

### new()

Creates a new training modifier object. If any of the arguments are not given, default argument values for that argument will be used.

```
TrainingModifier.new(Model: ModelObject, gradientDescentType: string, batchSize: integer, showOutput: boolean): GradientDescentModifierObject
```

#### Parameters:

* Model: The model object to modify its training capabilities.

* gradientDescentType: The type of gradient descent to be used when train() function is called. Available modes are "Batch", "MiniBatch" and "Stochastic".

* batchSize: The batch size to split the featureMatrix and labelVector into multiple parts.

* showOutput: Set whether or not to show the final cost for each epoch (MiniBatch) or data (Stochastic).

#### Returns:

* TrainingModifierObject: A training modifier object that uses the model's train(), predict() and reinforce() functions so that it behaves like a regular model.

## Functions

### train()

Trains the machine/deep learning model under specific gradient descent mode.

```
TrainingModifier:train(...): number[]
```

#### Parameters:

* ...: The parameters are the same to the original model's train() function.

#### Returns:

* costArray: An array containing cost values.

### predict()

Predict the values for given data.

```
TrainingModifier:predict(...): ...
```

#### Parameters:

...: The parameters are the same to the original model's predict() function.

#### Returns:

...: The outputs are the same to the original model's predict() function.

### reinforce()

Reward or punish model based on the current state of the environment.

```
TrainingModifier:reinforce(currentFeatureVector: Matrix, rewardValue: number, returnOriginalOutput: boolean): integer, number -OR- Matrix
```

#### Parameters:

* currentFeatureVector: Matrix containing data from the current state.

* rewardValue: The reward value added/subtracted from the current state (recommended value between -1 and 1, but can be larger than these values). 

* returnOriginalOutput: Set whether or not to return predicted vector instead of value with highest probability.

## Inherited From

* [BaseInstance](../Cores/BaseInstance.md)