# Introduction To Reinforcement Learning

## What Is Reinforcement Learning?

Reinforcement learning is a way for our models to learn on its own without the labels.

We can expect our models to perform poorly at the start of the training but they will gradually improve over time.

## The Basics

### Environment Feature Vector

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
