# AI Player Systems

* [Creating Data-Based Reactionary AI Player Model](AIPlayerSystems/CreatingDataBasedReactionaryAIPlayerModel.md)

  * Same as data-based AI players.
 
  * The only difference is that you give counter attacks to players' potential attacks.

  * Best for mixing machine learning with game designers' control.

* Creating Reward-Maximization-Based Reactionary AI Player Model

  * Same as reward-maximization-based AI players.
 
  * The only difference is that you give counter attacks to players' potential attacks.

  * Best for mixing reinforcement learning with game designers' control.

  * Breaks mathematical theoretical guarantees due to interference from game designers' control instead of model's own actions. Therefore, it is risky to use.

* Creating Data-Based AI Player Model

  * Uses real players' data so that the AI players mimic real players.
 
  * Matches with real players' general performance.

* Creating Reward-Maximization-Based AI Player Model

  * Allows the creation of AI players that maximizes positive rewards.
 
  * May outcompete real players.

  * May exploit bugs and glitches.
