# High-Value Project Tutorials

### Disclaimer

Before you engage in integrating machine, deep and reinforcement learning models into live projects, I recommend you to have a look at safe practices [here](HighValueProjectTutorials/SafePracticesForLiveProjects).

## Retention Systems

* [Creating Time-To-Leave Prediction Model](HighValueProjectTutorials/CreatingTimeToLeavePredictionModel.md)

* [Creating Probability-To-Leave Prediction Model](HighValueProjectTutorials/CreatingProbabilityToLeavePredictionModel.md)

* [Creating Play Time Maximization Model](HighValueProjectTutorials/CreatingPlayTimeMaximizationModel.md)

  * The model chooses actions or events that maximizes play time.

  * Have higher play time potential due to its ability to exploit and explore than the other two methods, but tend to be risky to use.

* Creating Left-To0-Early-Detection (I forgot the implementation.)

    * Highly exploitable if the player accumulates long session time over many sessions and then suddenly gradually decreases the session time to earn rewards.

## Regression Recommendation System

* [Creating Probability-Based Recommendation Model](HighValueProjectTutorials/CreatingProbabilityBasedRecommendationModel.md)

* Creating Similarity-Based Recommendation Model

* Creating Reward-Maximization-Based Regression Recommendation Model

  * Have higher monetization potential due to its ability to exploit and explore than the other two methods, but tend to be risky to use.

## Binary Recommendation Systems

* [Creating Classification-Based Recommendation Model](HighValueProjectTutorials/CreatingClassificationBasedRecommendationModel.md)

* Creating Reward-Maximization-Based Binary Recommendation Model

  * Have higher monetization potential due to its ability to exploit and explore than the classification-based model, but tend to be risky to use.

## AI Players

* Creating Data-Based AI Players

  * Uses real players' data so that the AI players mimic real players.
 
  * Matches with real players' general performance.

* Creating Reward-Maximization-Based AI Players

  * Allows the creation of AI players that maximizes positive rewards.
 
  * May outcompete real players.

  * May exploit bugs and glitches.

* Creating Data-Based Reactionary AI Players

  * Same as data-based AI players.
 
  * The only difference is that you give counter attacks to players' potential attacks.

  * Best for mixing machine learning with game designers' control.

* Creating Reward-Maximization-Based Reactionary AI Players

  * Same as reward maximization-based AI players.
 
  * The only difference is that you give counter attacks to players' potential attacks.

  * Best for mixing reinforcement learning with game designers' control.

  * Breaks mathematical theoretical guarantees due to inteference from game designers' control instead of model's own actions. Therefore, it is risky to use.

## Adaptive Difficulty Systems

* [Creating Regression-Based Enemy Data Generation Model](HighValueProjectTutorials/CreatingRegressionBasedEnemyDataGenerationModel.md)

  * Everytime a player kills an enemy, both player's combat data and enemy's data are used to train the model.

* [Creating Cluster-Based Enemy Data Generation Model](HighValueProjectTutorials/CreatingClusterBasedEnemyDataGenerationModel.md)

  * Uses players' combat data to generate the center of enemies' data.

* [Creating Reward-Maximization-Based Difficulty Generation Model](HighValueProjectTutorials/CreatingRewardMaximizationBasedDifficultyGenerationModel.md)

  * Everytime an enemy is killed, the positive reward tells the model to "make more enemies similar to this". 

  * If the player ignores or doesn't kill the enemy, the negative reward tells the model that "this enemy is not interesting to the player" or "this enemy is too hard for the player to kill".

  * Have higher play time potential due to its ability to exploit and explore than the other two methods, but tend to be risky to use.

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
