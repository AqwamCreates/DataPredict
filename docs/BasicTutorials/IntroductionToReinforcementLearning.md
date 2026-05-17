# Introduction To Reinforcement Learning

## What Is Reinforcement Learning?

Reinforcement learning is a way for our models to learn on its own without the labels.

We can expect our models to perform poorly at the start of the training but they will gradually improve over time.

## Different Types Of Reinforcement Learning

Currently, the DataPredict™ library provides two different methods of reinforcement learning. The table below will show you the comparison between them

| Property          | Tabular Reinforcement Learning | Deep Reinforcement Learning   |
|-------------------|--------------------------------|-------------------------------|
| Input             | An environment feature value   | An environment feature vector |
| Discrete Output   | Single action value            | Single action value           |
| Continuous Output | Not applicable                 | A mean action vector          |

## Environment Feature Inputs

Currently, because different methods requires different way on how we input things, we will break them into both cases

### Environment Value For Tabular Reinforcement Learning

An environment feature value is one of 



### Environment Vector For Deep Reinforcement Learning

An environment feature vector is a vector containing all the information related to model's environment. It can contain as many information such as:

* Distance

* Health

* Speed

An example of environment feature vector will look like this:

```lua
local environmentFeatureVector = {

  {1, -32, 234, 12, -97} -- 1 is added at first column for bias, but it is optional.

}
```

## Action spaces

An action space just means the set of actions that the AI could take for any given state. There are two types of action spaces: 

  * Discrete

  * Continuous 

### Discrete Action Space

Discrete action space are where the AI could choose only one action from a set of actions that exists for a specific environment. For example:

  * Movements: Up, down, forward, backward

  * Policeman actions: Move towards the criminal, run away, patrol, check, arrest

Notice that you can only choose one action from a set of actions. More than one action cannot be performed at the same time.

### Continuous Action Space

Continuous action space, on the other hand, are where the AI could choose different values for each of the actions that exists for a specific environment. For example:

* Driving: Throttle speed, steering rotation, brake amount

* Robotic hand movements: Finger 1 rotation, finger 2 rotation, finger 3 rotation

As you can see, you can get the values for each of the actions. More than one action can be performed at the same time.

## Choosing The Correct Algorithm For A Given Action Space

From the above, you can see that different types of action spaces have different types of properties. That also means that the way that our AI will have different way of learning things due for different properties. Because of how much mathematics are involved, we will not cover them any further.

What you will need to know instead that you will need to match the correct QuickSetup object and algorithm functions to use for a given action space type.

| Action Space | QuickSetup Object To Use | Function To Use To Perform The Step Updates | What Value Type Is Used To Update The Algorithm        |
|--------------|--------------------------|---------------------------------------------|--------------------------------------------------------|
| Discrete     | CategoricalPolicy        | categoricalUpdate()                         | A single action                                        |
| Continuous   | DiagonalGaussianPolicy   | diagonalGaussianUpdate()                    | An action vector containing all values for all actions |

# Reward Values

This is the value where we reward or punish the models. The properties of reward value is shown below:

* Positive value: Reward

* Negative Value: Punishment

* Large value: Large reward / punishment

* Small value: Small reward / punishment

It is recommended to set the reward that is within the range of:

```lua
-1 <= (total reward * learning rate) <= 1
```

That's all what you need to know for today!

Thank you very much for reading this tutorial. Have a nice day!
