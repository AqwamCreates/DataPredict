# [API Reference](../../API.md) - [Models](../Models.md) - TabularOffPolicyMonteCarloControl

TabularOffPolicyMonteCarloControl is a state-action grid with reinforcement learning capabilities. It can predict any positive numbers of discrete values.

## Constructors

### new()

Create new model object. If any of the arguments are nil, default argument values for that argument will be used.

```
TabularOffPolicyMonteCarloControl.new(learningRate: number, discountFactor: number, targetPolicyFunction: string): ModelObject
```

#### Parameters:

* learningRate: The speed at which the model learns. Recommended that the value is set between 0 to 1. [Default: 0.1]

* discountFactor: The higher the value, the more likely it focuses on long-term outcomes. The value must be set between 0 and 1. [Default: 0.95]

* targetPolicyFunction: A function that defines the target policy used to select actions. The policy should be based on the current Q-values (or state-action values). This function determines how the agent chooses actions based on its current knowledge. Available options include:

	* Greedy: Selects the action with the highest Q-value for a given state. This is typically the optimal policy, assuming the Q-values are accurate.

	* Softmax: Selects actions probabilistically, where actions with higher Q-values are more likely to be chosen. The probability of selecting an action is determined by a temperature parameter that controls the exploration-exploitation trade-off.

	* StableSoftmax: The more stable option of Softmax (Default)

#### Returns:

* ModelObject: The generated model object.

## Inherited From

* [TabularReinforcementLearningBaseModel](TabularReinforcementLearningBaseModel.md)

## References

* [Off-Policy Monte Carlo Control, Page 90](http://incompleteideas.net/book/bookdraft2017nov5.pdf)

* [Forgetting Early Estimates in Monte Carlo Control Methods](https://ev.fe.uni-lj.si/3-2015/Vodopivec.pdf)
