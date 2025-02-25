# [API Reference](../../API.md) - [ValueSchedulers](../ValueSchedulers.md) - BaseValueScheduler

BaseValueScheduler is a base for all value schedulers.

## Constructors

### new()

Creates a new base value scheduler object.

```
BaseValueScheduler.new(valueSchedulerName: string): BaseValueSchedulerObject
```

#### Parameters

* valueSchedulerName: The value scheduler name that is stored in base value scheduler.

#### Returns:

* valueSchedulerObject: The generated value scheduler object.

## Functions

### calculate()

Returns a modified cost function derivatives.

```
BaseValueScheduler:calculate(value: number): number
```

#### Parameters:

* value: The value to be scheduled.

#### Returns:

* value: The scheduled value.

### getvalueSchedulerName()

Gets the value scheduler's name from the base value scheduler.

```
BaseValueScheduler:getvalueSchedulerName()
```

#### Returns:

* valueSchedulerName: The value scheduler name that is stored in base value scheduler.

### setCalculateFunction()

Sets a calculate function for the base value scheduler.

```
BaseValueScheduler:setCalculateFunction(calculateFunction: Function)
```

#### Parameters:

* The calculate function to be used by the base valueScheduler when calculate() function is called.

### getvalueSchedulerInternalParameters()

Gets the value scheduler's internal parameters from the base value scheduler.

```
BaseValueScheduler:getvalueSchedulerInternalParameters(doNotDeepCopy: boolean): {}
```

#### Parameters:

* doNotDeepCopy: Set whether or not to deep copy the value scheduler internal parameters.

#### Returns:

* valueSchedulerInternalParameters: A matrix/table containing value scheduler internal parameters fetched from the base value scheduler.

### setvalueSchedulerInternalParameters()

Sets the value scheduler's internal parameters from the base value scheduler.

```
BaseValueScheduler:setvalueSchedulerInternalParameters(valueSchedulerInternalParameters: {}, doNotDeepCopy: boolean)
```

#### Parameters:

* valueSchedulerInternalParameters: A matrix/table containing valueScheduler internal parameters that will be used by the base valueScheduler.

* doNotDeepCopy: Set whether or not to deep copy the value scheduler internal parameters.

### reset()

Reset valueScheduler's stored values (excluding the parameters).

```
BaseValueScheduler:reset()
```

## Inherited From

* [BaseInstance](../Cores/BaseInstance.md)