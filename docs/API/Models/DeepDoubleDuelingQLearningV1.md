# [API Reference](../../API.md) - [Models](../Models.md) - DeepDoubleDuelingQLearningV1 (D3QN)

DeepDoubleDuelingQLearningV1 is a neural network with reinforcement learning capabilities. It can predict any positive numbers of discrete values.

It uses Hasselt et al. (2010) version, where a single neural network is selected from two neural networks with equal probability for training.

## Constructors

### new()

Create new model object. If any of the arguments are nil, default argument values for that argument will be used.

```
DeepDoubleDuelingQLearning.new(discountFactor: number): ModelObject
```

#### Parameters:

* discountFactor: The higher the value, the more likely it focuses on long-term outcomes. The value must be set between 0 and 1.

#### Returns:

* ModelObject: The generated model object.

## Functions

### setParameters()

Set model's parameters. When any of the arguments are nil, previous argument values for that argument will be used.

```
DeepDoubleDuelingQLearning:setParameters(discountFactor: number)
```

#### Parameters:

* discountFactor: The higher the value, the more likely it focuses on long-term outcomes. The value must be set between 0 and 1.

### setAdvantageModelParameters1()

Sets model parameters to be used by the model.

```
DeepDoubleDuelingQLearning:setAdvantageModelParameters1(AdvantageModelParameters1: ModelParameters, doNotDeepCopy: boolean)
```

#### Parameters:

* AdvantageModelParameters1: First model parameters to be used by the model.

* doNotDeepCopy: Set whether or not to deep copy the model parameters.

### setAdvantageModelParameters2()

Sets model parameters to be used by the model.

```
DeepDoubleDuelingQLearning:setAdvantageModelParameters2(AdvantageModelParameters2: ModelParameters, doNotDeepCopy: boolean)
```

#### Parameters:

* AdvantageModelParameters2: Second model parameters to be used by the model.

* doNotDeepCopy: Set whether or not to deep copy the model parameters.

### getAdvantageModelParameters1()

Sets model parameters to be used by the model.

```
DeepDoubleQLearning:getAdvantageModelParameters1(doNotDeepCopy: boolean): ModelParameters
```

#### Parameters:

* doNotDeepCopy: Set whether or not to deep copy the model parameters.

#### Returns:

* AdvantageModelParameters1: First advantage model parameters that was used by the model.

### getModelParameters2()

Sets model parameters to be used by the model.

```
DeepDoubleQLearning:getAdvantageModelParameters2(doNotDeepCopy: boolean): ModelParameters
```

#### Parameters:

* doNotDeepCopy: Set whether or not to deep copy the model parameters.

#### Returns:

* AdvantageModelParameters2: Second advantage model parameters that was used by the model.

### setValueModelParameters1()

Sets model parameters to be used by the model.

```
DeepDoubleDuelingQLearning:setValueModelParameters1(ValueModelParameters1: ModelParameters, doNotDeepCopy: boolean)
```

#### Parameters:

* ValueModelParameters1: First model parameters to be used by the model.

* doNotDeepCopy: Set whether or not to deep copy the model parameters.

### setValueModelParameters2()

Sets model parameters to be used by the model.

```
DeepDoubleDuelingQLearning:setValueModelParameters2(ValueModelParameters2: ModelParameters, doNotDeepCopy: boolean)
```

#### Parameters:

* ValueModelParameters2: Second model parameters to be used by the model.

* doNotDeepCopy: Set whether or not to deep copy the model parameters.

### getValueModelParameters1()

Sets model parameters to be used by the model.

```
DeepDoubleQLearning:getValueModelParameters1(doNotDeepCopy: boolean): ModelParameters
```

#### Parameters:

* doNotDeepCopy: Set whether or not to deep copy the model parameters.

#### Returns:

* ValueModelParameters1: First value model parameters that was used by the model.

### getModelParameters2()

Sets model parameters to be used by the model.

```
DeepDoubleQLearning:getValueModelParameters2(doNotDeepCopy: boolean): ModelParameters
```

#### Parameters:

* doNotDeepCopy: Set whether or not to deep copy the model parameters.

#### Returns:

* ValueModelParameters2: Second value model parameters that was used by the model.

## Inherited From

* [ReinforcementLearningDeepDuelingQLearningBaseModel](ReinforcementLearningDeepDuelingQLearningBaseModel.md)
