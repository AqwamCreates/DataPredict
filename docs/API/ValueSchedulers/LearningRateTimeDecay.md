# [API Reference](../../API.md) - [ValueSchedulers](../ValueSchedulers.md) - TimeDecay

## Constructors

### new()

Creates a new value scheduler object. If there are no parameters given for that particular argument, then that argument will use default value.

```
TimeDecay.new(decayRate: number): ValueSchedulerObject
```

#### Parameters:

* decayRate: The decay rate for learning rate.

#### Returns:

* ValueSchedulerObject: The generated value scheduler object.

## Functions

### setDecayRate()

```
TimeDecay:setDecayRate(decayRate: number)
```

#### Parameters:

* decayRate: The decay rate for learning rate.

## Inherited From

* [BaseValueScheduler](BaseValueScheduler.md)
