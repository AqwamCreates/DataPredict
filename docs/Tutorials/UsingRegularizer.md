# What Is Regularization?

Regularization is to ensure that the model do not overfit. In other words, we want to ensure that the model to generalize instead of memorizing solutions.

# Getting Started

In order for us to use the regularization, we need to create an regularizer object.

```lua
local RegularizerObject = DataPredict.Others.Regularizer.new()
```

Regularizer object takes in two parameters. For the sake of this tutorial, we will leave them empty.

# Combining Our Regularizer With Our Model

To combine, you must put the regularization object into the model's setOptimizer() function.

```lua
LogisticRegressionModel:setRegularizer(RegularizerObject)
```

Then, we can now train with our regularizer included. Do note that not all models uses regularizer, so please check the API reference if this option is available or not.

That's all for now!
