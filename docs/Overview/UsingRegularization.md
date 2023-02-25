# What is regularization?

Regularization is to ensure that the model do not overfit. In other words, we want to ensure that the model to generalize more than memorizing solutions.

# Getting Started

In order for us to use the regularization, we need to create an regularization object.

```
local Regularization = MDLL.Others.Regularization

local RegularizationObject = Regularization.new()
```

Regularization object takes in two parameters. For the sake of this tutorial, we will leave them empty.

# Combining Our Regularization With Our Model

To combine, you must put the regularization object into the model's setOptimizer() function.

```
LogisticRegressionModel:setRegularization(RegularizationObject)
```

Then, we can now train with our regularization  included. Do note that not all models uses regularization, so please check the API reference if this option is available or not.

That's all for now!
