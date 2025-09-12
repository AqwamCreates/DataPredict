# High-Value Project Tutorials

## Retention Systems

* [Creating Time-To-Leave Prediction Model](HighValueProjectTutorials/CreatingTimeToLeavePredictionModel.md)

* [Creating Probability-To-Leave Prediction Model](HighValueProjectTutorials/CreatingProbabilityToLeavePredictionModel.md)

## Recommendation Systems

* Creating Similarity-Based Recommendation System

* Creating Reward-Maximization-Based Recommendation System

* Creating Classification-Based Recommendation System

## AI Players

* Creating Data-Based AI Players

  * Uses real players' data so that the AI players mimic real players.
 
  * Matches with real players' general performance.

* Creating Reward-Based AI Players

  * Allows the creation of AI players that maximizes positive rewards.
 
  * May outcompete real players.

## Adaptive Difficulty Systems

* Creating Success-Based Adaptive Difficulty System

  * Encodes positive label for every successful kills and associate it with current enemy's data.

* Creating Reward-Maximization-Based Adaptive Difficulty System

  * Everytime an enemy is killed, the positive reward tells the model to "make more enemies similar to this". 

  * If the player ignores or doesn't kill the enemy, the negative reward tells the model that "this enemy is not interesting to the player" or "this enemy is too hard for the player to kill".

* Creating Cluster-Based Difficulty System

  * Uses players' data to generate the best enemies' data.

## Quality Assurance

* Creating Reward-Maximization-Based AI Bug Hunter

  * For a given "normal" data, the AI is rewarded based on how far the difference of the collected data compared to current data.

* Creating Curiosity-Based AI Bug Hunter

  * The AI will maximize actions that allows it to explore the game. 

## Targeting Systems

* [Creating Cluster-Based Targeting Model](HighValueProjectTutorials/CreatingClusterBasedTargetingModel.md)

  * Find the center of players based on number of clusters.

## Priority Systems

* Creating Probability-Based Priority System

* Creating Regression-Based Priority System
