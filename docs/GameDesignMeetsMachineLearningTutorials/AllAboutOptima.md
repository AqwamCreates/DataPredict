# All About Optima

## Interpreting Local And Global Optimum In Game Design

* Local Optimum -> The best score for a subset of input values.
 
* Global Optimum -> The best score for a maximum number of input values. This maximum number of input values can lead to the best score for every single input values.

## Interpreting Minima And Maxima

To make it short, Optimum Minimum (Minima) and Optimum Maximum (Maxima) is basically a measurement of score related to the algorithms' weights. 

The score can be based on:

* The difference between the translated output features and the actual output features.

* The difference between the expected reward and the actual reward received.

Minima is basically the lowest score that an algorithm could achieve if the algorithm's weights leads to that lowest score.

Maxima is basically the highest score that an algorithm could achieve if the algorithm's weights leads to that highest score.

In supervised learning, the algorithm's main goal is to find a local or global minima because its aim is to minimize the the translation difference when translating from input to outputs.

In reinforcement learning, the algorithm's main goal is to find a local or global maxima because its aim is to maximize the score that leads to a stated goal.
