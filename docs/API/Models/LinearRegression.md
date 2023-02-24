# API Reference - Models - LinearRegression

## Constructors

### new()

Create new model object. If any of the arguments are not given, default argument values for that argument will be used.

```
LinearRegression.new(maxNumberOfIterations: integer, learningRate: number, lambda: number, lossFunction: string, targetCost: number)
```

#### Parameters:

maxNumberOfIterations = How many times should the model needed to be trained.

learningRate = The speed at which the model learns. Recommended that the value is set between (0 to 1).

lambda = Reqularisation factor

lossFunction = The function to calculate the cost of each training. Available options are "L1" and "L2".

targetCost = The cost at which the model stops training.

## Functions

### setParameters()

```
LinearRegression:setParameters(maxNumberOfIterations: integer, learningRate: number, lambda: number, lossFunction: string, targetCost: number)
```

#### Parameters:

maxNumberOfIterations = How many times should the model needed to be trained.

learningRate = The speed at which the model learns. Recommended that the value is set between (0 to 1).

lambda = Reqularisation factor

lossFunction = The function to calculate the cost of each training. Available options are "L1" and "L2".

targetCost = The cost at which the model stops training.

Set model's parameters. When any of the arguments are not given, previous argument values for that argument will be used.

### setOptimizer()

Set optimizer for the model by inputting the optimizer object.

```
LinearRegression:setOptimizer(Optimizer: OptimizerObject)
```

#### Parameters:

Optimer: The optimizer to be used.

### train()

Train the model.

```
costArray = LinearRegression:train(featureMatrix: Matrix, labelVector: Matrix)
```

#### Parameters:

featureMatrix: Matrix containing all data.

labelVector: A (n x 1) matrix containing values related to featureMatrix.

#### Returns:

costArray = An array containing cost values

### predict()

Predict the value for a given data.

```
LinearRegression:predict(featureMatrix: Matrix): number
```

#### Parameters:

featureMatrix: Matrix containing all data.
