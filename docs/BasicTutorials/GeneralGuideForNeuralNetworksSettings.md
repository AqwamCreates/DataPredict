# General Guide For Neural Networks Settings

This guide assumes that you have basic understanding of neural networks. If not, you can find the resources online and have a look at how neural networks works.

Make note that this guide is an oversimplification and may not exactly cover the whole details of how neural network works.

Without further ado, let's begin.

## Layers

The choice of your layers are quite important. It is one of the factor determine the accuracy and the training speed of your neural network model.

Usually, it is recommended that you have few layers if you can determine what pattern leads to certain predictions. For example:

* Four classes for four different combination of inputs, where each combination belongs to one class. In other words, two inputs and four outputs.

*  Two classes for two different combinations of inputs. If the input is greater than 0, then it belongs to class 1, otherwise class -1.

If you can determine the pattern, then I recommend you that you only build two layer neural networks as more complex models may produce less accurate outputs.

In other words, the more complex the pattern means more layers are needed to produce more accurate outputs (generally).

Here's a formula for you to remember (for qualitative analysis, not quantitative):

* C = Model Complexity (a higher value indicates a more complex model), where 0 is simple and 1 is complex.

* P = Pattern Detectability (a higher value indicates an easier-to-detect pattern), where 0 is easily detected and 1 is impossible to detect.

* A = Accuracy, where value is between 0 and 1.

* A = (1 - C) * P

## Activation Functions

Different activation functions have different properties. It is very important to choose the correct ones to achieve high accuracy. Here are the functions with their properties listed below:

* ReLU: Great for making sure only few neurons get activated. The problem is that it gives the output of 0 for all negative input values, which could lead to lower number of neurons activating for each iteration during training. Eventually, it will lead to no neurons activating and produce innacurate predictions.

* LeakyReLU: Same as ReLU, but less terrible at handling negative input values.

* ELU: Same as ReLU, but capable of handling negative input values. The only problem is the computational cost as it uses exponent function.

* Sigmoid: As input values goes further from 0.5, the output slowly reaches 1 or 0; excellent for making sure no large outputs being passed on to next neuron. But being not centered around 0 may cause some issue with some optimizers and weight initialization strategies.

* Tanh: Same as sigmoid, but it is centered around 0. So less issues with optimizers and some initialization strategies. Also, great for negative input values.

## Weight Initialization

How we initialize our weights can affect how fast the model can learn. For example:

* Having all weight values falls either on positive side or negative side only: If the global optimum is on the opposite side, the model have to move all its weights from one side to another as opposed to changing some weights.

* Having all weight values that falls between 1 and 0 (or -1). This allows the weight values to start at the center of dimensional space and hence shorter distance to travel to global optimum (most of the time).

## Regularization

Regularization avoids our model from "memorizing" the connection between the inputs and outputs, which could lead to lower accuracy. It ensure that the model generalizes the connection between the inputs and outputs.

## Bias Neurons

The presence of bias neuron must not be underestimated. It allows the calculated values to move away from 0 instead of being centered to it. In most cases, these are usually added to each layer except the final layer.
