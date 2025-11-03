# AI Player Systems

Generally, the models can be split into two categories:

* Simple: Uses discrete states (like run, fight and idle).

* Deep: Uses continuous states (like health, distance and damage).

## Data-To-Action Players

* [Creating Simple Data-Based AI Player Model](AIPlayerSystems/CreatingSimpleDataBasedAIPlayerModel.md)
 
  * Matches with real players' general performance.

  * Minimal implementation takes a minimum of 30 minutes using DataPredict™.

* Creating Deep Data-Based AI Player Model

  * Uses real players' environment data so that the AI players mimic real players.

## Data-To-Action-To-Reaction Players

* [Creating Simple Data-Based Reactionary AI Player Model](AIPlayerSystems/CreatingSimpleDataBasedReactionaryAIPlayerModel.md)

  * Minimal implementation takes a minimum of 30 minutes using DataPredict™.

* Creating Deep Data-Based Reactionary AI Player Model

## Data-To-Action Optimization Players

* Creating Simple Reward-Maximization-Based AI Player Model

  * May outcompete real players.

* Creating Deep Reward-Maximization-Based AI Player Model
 
  * May outcompete real players.

  * May exploit bugs and glitches.

## Data-To-Action-To-Reaction Optimization Players

* [Creating Simple Reward-Maximization-Based Reactionary AI Player Model](AIPlayerSystems/CreatingSimpleRewardMaximizationBasedReactionaryAIPlayerModel.md)

  * Breaks mathematical theoretical guarantees due to interference from game designers' control instead of model's own actions. Therefore, it is risky to use.

* Creating Deep Reward-Maximization-Based Reactionary AI Player Model

  * Breaks mathematical theoretical guarantees due to interference from game designers' control instead of model's own actions. Therefore, it is risky to use.
