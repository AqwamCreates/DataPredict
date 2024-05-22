# [API Reference](../../API.md) - [Models](../Models.md) - DeepDuelingQLearning

DeepDuelingQLearning is a base class for reinforcement learning.

## Notes:

* The Advantage and Value models must be created separately. Then use setAdvantageModel() and setValueModel() to put it inside the DuelingQLearning model.

* Advantage and Value models must be a part of NeuralNetwork model. If you decide to use linear regression or logistic regression, then it must be constructed using NeuralNetwork model. 

* Ensure the final layer of the Value model has only one neuron. It is the default setting for all Value models in research papers.

## Constructors

### new()

Create new model object. If any of the arguments are nil, default argument values for that argument will be used.

```
DeepDuelingQLearning.new(discountFactor: number): ModelObject
```

#### Parameters:

* discountFactor: The higher the value, the more likely it focuses on long-term outcomes. The value must be set between 0 and 1.

#### Returns:

* ModelObject: The generated model object.

## Functions

### setParameters()

Set model's parameters. When any of the arguments are nil, previous argument values for that argument will be used.

```
DeepDuelingQLearning:setParameters(discountFactor: number)
```

#### Parameters:

* discountFactor: The higher the value, the more likely it focuses on long-term outcomes. The value must be set between 0 and 1.

### setAdvantageModel()

```
DeepDuelingQLearning:setAdvantageModel(Model: ModelObject)
```

#### Parameters:

* Model: The model to be used as an Advantage model.

### setValueModel()

```
DeepDuelingQLearning:setValueModel(Model: ModelObject)
```

#### Parameters:

* Model: The model to be used as a Value model.

#### Parameters:

* ExperienceReplay: The experience replay object.

### reinforce()

Reward or punish model based on the current state of the environment.

```
DeepDuelingQLearning:reinforce(currentFeatureVector: Matrix, rewardValue: number, returnOriginalOutput: boolean): integer, number -OR- Matrix
```

#### Parameters:

* currentFeatureVector: Matrix containing data from the current state.

* rewardValue: The reward value added/subtracted from the current state (recommended value between -1 and 1, but can be larger than these values). 

* returnOriginalOutput: Set whether or not to return predicted vector instead of value with highest probability.

#### Returns:

* predictedLabel: A label that is predicted by the model.

* value: The value of predicted label.

-OR-

* predictedVector: A matrix containing all predicted values from all classes.

### update()

Updates the model parameters.

```
DuelingQLearning:update(previousFeatiureVector: featureVector, action: number/string, rewardValue: number, currentFeatureVector: featureVector)
```

#### Parameters:

* previousFeatiureVector: The previous state of the environment.

* action: The action selected.

* rewardValue: The reward gained at current state.

* currentFeatureVector: The currrent state of the environment.

### reset()

Reset model's stored values (excluding the parameters).

```
DuelingQLearning:reset()
```

### destroy()

Destroys the model object.

```
ActorCritic:destroy()
```

## References

* [Dueling Deep Q Networks by Chris Yoon](https://towardsdatascience.com/dueling-deep-q-networks-81ffab672751)
