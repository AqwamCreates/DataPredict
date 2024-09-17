# [API Reference](../../API.md) - [ValueSchedulers](../ValueSchedulers.md) - StepDecay

## Constructors

### new()

Creates a new optimizer object. If there are no parameters given for that particular argument, then that argument will use default value.

```
StepDecay.new(decayRate: number, timeStepToDecay: integer): ValueSchedulerObject
```

#### Parameters:

* decayRate: The decay rate for value.

* timeStepToDecay: The number of time steps to decay the value.

#### Returns:

* ValueSchedulerObject: The generated value scheduler object.

## Functions

### setDecayRate()

```
StepDecay:setDecayRate(decayRate: number)
```

#### Parameters:

* decayRate: The decay rate for the value.

### setTimeStepToDecay()

```
StepDecay:setTimeStepToDecay(timeStepToDecay: integer)
```

#### Parameters:

* timeStepToDecay: The number of time steps to decay the value.

### reset()

Reset optimizer's stored values (excluding the parameters).

```
StepDecay:reset()
```

## Inherited From

* [BaseValueScheduler](BaseValueScheduler.md)
