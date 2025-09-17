# High-Value Project Tutorials

### Disclaimer

* Before you engage in integrating machine, deep and reinforcement learning models into live projects, I recommend you to have a look at safe practices [here](HighValueProjectTutorials/SafePracticesForLiveProjects.md).

* The content of this page and its links is licensed under [Terms And Conditions](TermsAndConditions.md).

  * Therefore, making copies or derivatives of this page and its links are not allowed.

  * Commercial use is not allowed.

## Retention Systems

* [Creating Time-To-Leave Prediction Model](HighValueProjectTutorials/CreatingTimeToLeavePredictionModel.md)

* [Creating Probability-To-Leave Prediction Model](HighValueProjectTutorials/CreatingProbabilityToLeavePredictionModel.md)

* [Creating Left-Too-Early Detection Model](HighValueProjectTutorials/CreatingLeftTooEarlyDetectionModel.md)

   * Inverse of probability-to-leave model by detecting outliers.

   * Highly exploitable if the player accumulates long session times over many sessions before suddenly decrease the session times gradually if rewards are involved.

* [Creating Play Time Maximization Model](HighValueProjectTutorials/CreatingPlayTimeMaximizationModel.md)

  * The model chooses actions or events that maximizes play time.

  * Have higher play time potential due to its ability to exploit and explore than the other three methods, but tend to be risky to use.

## Regression Recommendation System

* [Creating Probability-Based Recommendation Model](HighValueProjectTutorials/CreatingProbabilityBasedRecommendationModel.md)

* Creating Similarity-Based Recommendation Model

* Creating Reward-Maximization-Based Regression Recommendation Model

  * Limited to one recommendation.

  * Have higher monetization potential due to its ability to exploit and explore than the other two methods, but tend to be risky to use.

## Binary Recommendation Systems

* [Creating Classification-Based Recommendation Model](HighValueProjectTutorials/CreatingClassificationBasedRecommendationModel.md)

* [Creating Reward-Maximization-Based Binary Recommendation Model](HighValueProjectTutorials/CreatingRewardMaximizationBasedBinaryRecommendationModel.md)

  * Limited to one recommendation at a time.

  * Have higher monetization potential due to its ability to exploit and explore than the classification-based model, but tend to be risky to use.

## Adaptive Difficulty Systems

* [Creating Regression-Based Enemy Data Generation Model](HighValueProjectTutorials/CreatingRegressionBasedEnemyDataGenerationModel.md)

  * Every time a player kills an enemy, both player's combat data and enemy's data are used to train the model.

* [Creating Cluster-Based Enemy Data Generation Model](HighValueProjectTutorials/CreatingClusterBasedEnemyDataGenerationModel.md)

  * Uses players' combat data to generate the center of enemies' data.

* [Creating Reward-Maximization-Based Difficulty Generation Model](HighValueProjectTutorials/CreatingRewardMaximizationBasedDifficultyGenerationModel.md)

  * Every time an enemy is killed, the positive reward tells the model to "make more enemies similar to this". 

  * If the player ignores or doesn't kill the enemy, the negative reward tells the model that "this enemy is not interesting to the player" or "this enemy is too hard for the player to kill".

  * Have higher play time potential due to its ability to exploit and explore than the other two methods, but tend to be risky to use.

## Targeting Systems

* [Creating Cluster-Based Targeting Model](HighValueProjectTutorials/CreatingClusterBasedTargetingModel.md)

  * Find the center of players based on number of clusters.

* Creating Reward-Maximization-Based Targeting Model

  * Might be the most terrible idea out of this list. However, I will not stop game designers from making their games look "smart".

    * The model will likely do a lot of exploration before it can hit a single player. Once that particular location is marked as "reward location", the model will might overfocus on it.

    * The upside is that if your map has terrains and structures, the model may learn to corner and "trap" players to certain locations for easy "kills".

  * Limited to one target at a time.

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

## Quality Assurance

* Creating Reward-Maximization-Based AI Bug Hunter

  * For a given "normal" data, the AI is rewarded based on how far the difference of the collected data compared to current data.

* Creating Curiosity-Based AI Bug Hunter

  * The AI will maximize actions that allows it to explore the game.

## Priority Systems

* Creating Probability-Based Priority System

* Creating Regression-Based Priority System
