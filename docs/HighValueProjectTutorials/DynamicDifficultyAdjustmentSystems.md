# Dynamic Difficulty Adjustment Systems

* [Creating Regression-Based Enemy Data Generation Model](DynamicDifficultyAdjustmentSystems/CreatingRegressionBasedEnemyDataGenerationModel.md)

  * Every time a player kills an enemy, both player's combat data and enemy's data are used to train the model.

  * No need to add new content; the model can use existing content to optimize your games.

  * Minimal implementation takes a minimum of 30 minutes using DataPredict™.

* [Creating Probability-Based Enemy Data Generation Model](DynamicDifficultyAdjustmentSystems/CreatingProbabilityBasedEnemyDataGenerationModel.md)

  * Uses players' combat data (optional) paired with enemies' data to predict likelihood that the player will engage with the enemy if spawned.

  * Uses "Generative One Class Support Vector Machine", which is an unconventional generative method that probably don't exist in the research literature.

  * No need to add new content; the model can use existing content to optimize your games.

  * Minimal implementation takes a minimum of 30 minutes using DataPredict™.

* [Creating Cluster-Based Enemy Data Generation Multiplayer Model](DynamicDifficultyAdjustmentSystems/CreatingClusterBasedEnemyDataGenerationForMultiplayerModel.md)

  * Uses players' combat data to generate the center of enemies' data, creating difficulty suited for the majority of players in a session.

  * No need to add new content; the model can use existing content to optimize your games.

  * Minimal implementation takes a minimum of 30 minutes using DataPredict™.

* [Creating Cluster-Based Enemy Data Generation Singleplayer Model](DynamicDifficultyAdjustmentSystems/CreatingClusterBasedEnemyDataGenerationForSingleplayerModel.md)

  * Uses a single player's defeated enemies' data to generate the center of enemies' data, creating difficulty personal to that player.

  * No need to add new content; the model can use existing content to optimize your games.

  * Minimal implementation takes a minimum of 30 minutes using DataPredict™.

* [Creating Unconditional-Diversity-Based Enemy Data Generation Model](DynamicDifficultyAdjustmentSystems/CreatingUnconditionalDiversityBasedEnemyDataGenerationModel.md)

  * Uses enemies' data as generator's output. The generated enemy data is then used against the discriminator to see if the player will interact with it.
 
    * Think of the discriminator as "fake player" and the discriminator will see the real enemy data that the player has interacted with.

  * No need to add new content; the model can use existing content to optimize your games.

  * More risky than the non-diverse models, but gives more variation in enemy data generation.

  * Minimal implementation takes a minimum of 1 hour using DataPredict™.

* [Creating Conditional-Diversity-Based Enemy Data Generation Model](DynamicDifficultyAdjustmentSystems/CreatingConditionalDiversityBasedEnemyDataGenerationModel.md)

  * Uses players' combat data for generator's input and enemies' data as generator's output. The generated enemy data is then used against the discriminator to see if the player will interact with it.
 
    * Think of the discriminator as "fake player" and the discriminator will see the real enemy data that the player has interacted with.

  * No need to add new content; the model can use existing content to optimize your games.

  * More risky than the non-diverse models, but gives more variation in enemy data generation.

  * Minimal implementation takes a minimum of 2 hour using DataPredict™.

* [Creating Reward-Maximization-Based Difficulty Generation Model](DynamicDifficultyAdjustmentSystems/CreatingRewardMaximizationBasedDifficultyGenerationModel.md)

  * Every time an enemy is killed, the positive reward tells the model to "make more enemies similar to this". 

  * If the player ignores or doesn't kill the enemy, the negative reward tells the model that "this enemy is not interesting to the player" or "this enemy is too hard for the player to kill".

  * No need to add new content; the model can use existing content to optimize your games.

  * Have higher play time potential due to its ability to exploit and explore than the other two models, but tend to be risky to use.

  * Minimal implementation takes a minimum of 2 hours using DataPredict™.