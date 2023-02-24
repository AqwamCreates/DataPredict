# API Reference - Model - LinearRegression

## Constructors

### new()

Create new model object. If any of the arguments are not given, default argument values for that argument will be used.

## Functions

### setParameters()

Set model's parameters. When any of the arguments are not given, previous argument values for that argument will be used.

### setOptimizer()

Set optimizer for the model by inputting the optimizer object.

```
LinearRegression:setOptimizer(Optimizer: OptimizerObject)
```

### train()

Train the model.

```
LinearRegression:train(featureMatrix: Matrix, labelVector: Matrix)
```

### predict()

Predict the value for a given data.

```
LinearRegression:predict(featureMatrix: Matrix): number
```


