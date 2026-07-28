# Choosing The Goal For The Model

In the general artificial intelligence space, they generally perform these tasks:

| Task                     | Explanation                                                                                                        | 
|--------------------------|--------------------------------------------------------------------------------------------------------------------|
| Supervised Learning      | Learns patterns between the input features and the output features.                                                |
| Unsupervised Learning    | Learns patterns within the input features.                                                                         |
| Semi-Supervised Learning | Use Supervised Learning predictions on inputs features that have unknown output features as its own training data. |
| Reinforcement Learning   | Learns what actions to take for a set of input features that maximizes the likelihood of reaching a goal.          |

* Note: Generative AIs falls under supervised learning because they require output features like text, images and audios. The generation part is the result of predicting from the input features.

## Supervised Learning VS Reinforcement Learning

On the surface, these two may look like the same. However, the difference lies on how they are trained.

* In supervised learning, we tell the algorithm that this input is what leads to that output.

* In reinforcement learning, we don't provide the output, but instead a score of how good the output is to reach a stated goal when the algorithm is given a particular input.

In other words, the supervised learning algorithms "translates" the input to a solution, while the reinforcement learning algorithms "searches" a solution for a given input.

## What's Your Goal?

* Goal Maximization -> Use "measurement of engagement" metrics as rewards and combine it with reinforcement learning models.

* Prediction -> Use regression and classification models.

* Best Middle Values -> Use clustering models.
