# API Reference - Model - NaiveBayes

NaiveBayes is an supervised machine learning model that predicts which classes that the input belongs to using probability.

## Constructors

### new()

Create new model object. If any of the arguments are not given, default argument values for that argument will be used.

## Functions

### setParameters()

Set model's parameters. When any of the arguments are not given, previous argument values for that argument will be used.

### setOptimizer()

Set optimizer for the model by inputting the optimizer object.

```
NaiveBayes:setOptimizer(Optimizer: OptimizerObject)
```

### train()

Train the model.

```
NaiveBayes:train(featureMatrix: Matrix, labelVector: Matrix)
```

### predict()

Predict which cluster does it belong to for a given data.

```
NaiveBayes:predict(featureMatrix: Matrix): number
```
