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
