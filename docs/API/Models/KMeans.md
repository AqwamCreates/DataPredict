# API Reference - Models - KMeans

KMeans is an unsupervised machine learning model that predicts which cluster that the input belongs to using distance.

## Constructors

### new()

Create new model object. If any of the arguments are not given, default argument values for that argument will be used.

## Functions

### setParameters()

Set model's parameters. When any of the arguments are not given, previous argument values for that argument will be used.

### setOptimizer()

Set optimizer for the model by inputting the optimizer object.

```
KMeans:setOptimizer(Optimizer: OptimizerObject)
```

### train()

Train the model.

```
KMeans:train(featureMatrix: Matrix, labelVector: Matrix)
```

### predict()

Predict which cluster does it belong to for a given data.

```
KMeans:predict(featureMatrix: Matrix): number
```
