# What are optimizers?

Optimizers are deep learning techinque that adjusts our machine/deep learning model learning rates. They make our models train faster and hence require less number of iterations.

# Getting Started

In order for us to use the optimizers, we need to create an optimizer object. In this tutorial, we will use "Adaptive Gradient" (a.k.a. Adagrad).

So first, lets initialize a new Adagrad optimizer object.

```
local Adagrad = MDLL.Optimizers.AdaptiveGradient

local AdagradOptimizer = Adagrad.new()
```

For this optimizer, there are no parameters for us to set. So, we will leave it empty. However, for others, they may use default parameter values.

# Combining our optimizer with our model

To combine, you must put the optimizer object into the model's setOptimizer() function.

```
LogisticRegressionModel:setOptimizer(AdagradOptimizer)
```

Then, we can now train with our optimizer included. Do note that not all models uses optimizers, so please check the API reference if this option is available or not.

That's all for now!
