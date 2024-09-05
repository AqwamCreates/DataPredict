# The Action Space, Simplified

An action space just means the set of actions that the AI could take for any given state. There are two types of action spaces: 

  * Discrete

  * Continuous 

## Discrete Action Space

Discrete action space are where the AI could choose only one action from a set of actions that exists for a specific environment. For example:

  * Movements: Up, down, forward, backward

  * Policeman actions: Move towards the criminal, run away, patrol, check, arrest

Notice that you can only choose one action from a set of actions. More than one action cannot be done at the same time.

## Continuous Action Space

Continuous action space, on the other hand, are where the AI could choose different values for each of the actions that exists for a specific environment. For example:

* Driving: Throttle speed, steering rotation, brake amount

* Robotic hand movements: Finger 1 rotation, finger 2 rotation, finger 3 rotation

As you can see, you can set the values for each of the actions. More than one action can be done at the same time.

# Choosing The Correct Algorithm For A Given Action Space

From the above, you can see that different types of action spaces have different types of properties. That also means that the way that our AI will have different way of learning things due for different properties. However, because of how much mathematics are involved, we will not cover them any further.

What you will need to know instead that you will need to match the correct functions and algorithms for a given action space type.

| Action Space | QuickSetup Object To Use | Function To Perform The Step Updates |
|--------------|--------------------------|--------------------------------------|
| Discrete     | CategoricalPolicy        | categoricalUpdate()                  |
| Continuous   | DiagonalGaussianPolicy   | diagonalGaussianUpdate()             |

That's all you need to know for today!

Thank you very much for reading this tutorial. Your support means very much to me.
