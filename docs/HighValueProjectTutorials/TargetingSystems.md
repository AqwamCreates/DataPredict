# Targeting Systems

* [Creating Distance-Minimization-Based Clustering Targeting Model](TargetingSystems/CreatingDistanceMinimizationBasedClusteringTargetingModel.md)

  * Find the center of players based on number of clusters.

  * Best suited for precise targeting to multiple areas.

  * Minimal implementation takes a minimum of 30 minutes using DataPredict™.

* [Creating Probability-Maximization-Based Clustering Targeting Model](TargetingSystems/CreatingProbabilityMaximizationBasedClusteringTargetingModel.md)

  * Produces clusters that maximizes the likelihood of being hit.

  * Minimal implementation takes a minimum of 30 minutes using DataPredict™.

* [Creating Probability-Based Targeting Model](TargetingSystems/CreatingProbabilityBasedTargetingModel.md)

  * Find the center of players by finding the area with high player density.

  * Can perform both precise and intentionally inaccurate, yet likely-to-hit targeting.

  * Only one cluster.

  * Minimal implementation takes a minimum of 30 minutes using DataPredict™.

* [Creating Reward-Maximization-Based Targeting Model](TargetingSystems/CreatingRewardMaximizationBasedTargetingModel.md)

  * If your map has terrains and structures, the model may learn to corner and "trap" players to certain locations for easy "kills" or "targets".

  * Limited to one target at a time, but can take in multiple player locations.

  * Might be the most terrible idea out of this list. However, I will not stop game designers from making their games look "smart".

    * The model will likely do a lot of exploration before it can hit a single player. Once that particular location is marked as "reward location", the model will might overfocus on it.

  * Minimal implementation takes a minimum of 2 hours using DataPredict™.
