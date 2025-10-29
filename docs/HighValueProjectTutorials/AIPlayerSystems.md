# AI Player Systems

* [Creating Simple Data-Based AI Player Model](AIPlayerSystems/CreatingSimpleDataBasedAIPlayerModel.md)

  * Uses real players' states so that the AI players mimic real players.
 
  * Matches with real players' general performance.

  * Minimal implementation takes a minimum of 30 minutes using DataPredict™.

* [Creating Simple Data-Based Reactionary AI Player Model](AIPlayerSystems/CreatingSimpleDataBasedReactionaryAIPlayerModel.md)

  * Same as above.
 
  * The only difference is that you give counter attacks to players' potential attacks.

  * Best for mixing machine learning with game designers' control.

  * Minimal implementation takes a minimum of 30 minutes using DataPredict™.

* Creating Deep Data-Based AI Player Model

  * Uses real players' environment data so that the AI players mimic real players.

  * Minimal implementation takes a minimum of 30 minutes using DataPredict™.

* Creating Deep Data-Based Reactionary AI Player Model

  * Same as above.

  * The only difference is that you give counter attacks to players' potential attacks.

  * Minimal implementation takes a minimum of 30 minutes using DataPredict™.

* Creating Simple Reward-Maximization-Based AI Player Model

  * Uses real players' states so that the AI players mimic real players.

  * May outcompete real players.

* [Creating Simple Reward-Maximization-Based Reactionary AI Player Model](AIPlayerSystems/CreatingSimpleRewardMaximizationBasedReactionaryAIPlayerModel.md)

  * Same as above.

  * The only difference is that you give counter attacks to players' potential attacks.

  * Minimal implementation takes a minimum of 1 hour using DataPredict™, especially if custom actions are associated with the model's output.

* Creating Deep Reward-Maximization-Based AI Player Model

  * Allows the creation of AI players that maximizes positive rewards.
 
  * May outcompete real players.

  * May exploit bugs and glitches.

* Creating Deep Reward-Maximization-Based Reactionary AI Player Model

  * Same as reward-maximization-based AI players.
 
  * The only difference is that you give counter attacks to players' potential attacks.

  * Best for mixing reinforcement learning with game designers' control.

  * Breaks mathematical theoretical guarantees due to interference from game designers' control instead of model's own actions. Therefore, it is risky to use.
