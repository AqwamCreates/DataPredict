# High-Value Project Tutorials - Beat The Retention! Not The Algorithm! Keep Players, Grow Your Game, Forget Competitors!

### Disclaimer

* References that validates the use cases can be found [here](HighValueProjectTutorials/References.md). It also includes my papers.

* The "minimal implementation time" assumes that a junior gameplay machine learning engineer is handling the implementation.

* Since DataPredict is written in native Lua, you can have extra compute per player alongside a single Roblox server by loading the models on players' Roblox client.

  * Phone users: Likely have 4 GB - 8 GB RAM. Variable CPU.
 
  * PC users: Likely have 8 GB - 16 GB RAM. Variable CPU.

* Before you engage in integrating machine, deep and reinforcement learning models into live projects, I recommend you to have a look at safe practices [here](HighValueProjectTutorials/SafePracticesForLiveProjects.md).

* The content of this page and its links are licensed under the DataPredict™ library's [Terms And Conditions](TermsAndConditions.md). This includes the codes shown in the links below.

  * Therefore, creating or redistributing copies or derivatives of this page and its links' contents are not allowed.

  * Commercial use is also limited unless you have a separate license from me.
  
* You can download and read the full list of commercial licensing agreements [here](https://github.com/AqwamCreates/DataPredict/blob/main/docs/DataPredictLibrariesLicensingAgreements.md).

* For information regarding potential license violations and eligibility for a bounty reward, please refer to the [Terms And Conditions Violation Bounty Reward Information](TermsAndConditionsViolationBountyRewardInformation.md).

## Links

* [Retention Systems](HighValueProjectTutorials/RetentionSystems.md)

* [Recommendation Systems](HighValueProjectTutorials/RecommendationSystems.md)

* [Dynamic Difficulty Adjustment Systems](HighValueProjectTutorials/DynamicDifficultyAdjustmentSystems.md)

* [Targeting Systems](HighValueProjectTutorials/TargetingSystems.md)

* [AI Players](HighValueProjectTutorials/AIPlayers.md) (Incomplete)

* [Quality Assurance](HighValueProjectTutorials/QualityAssurance.md) (Incomplete)

* [Priority Systems](HighValueProjectTutorials/PrioritySystems.md) (Incomplete)

* [Anti-Cheat Systems](HighValueProjectTutorials/AntiCheatSystems.md)

* [Placement Systems](HighValueProjectTutorials/PlacementSystems.md)

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

  * Same as reward-maximization-based AI players.
 
  * The only difference is that you give counter attacks to players' potential attacks.

  * Best for mixing reinforcement learning with game designers' control.

  * Breaks mathematical theoretical guarantees due to inteference from game designers' control instead of model's own actions. Therefore, it is risky to use.

## Quality Assurance

* Creating Reward-Maximization-Based AI Bug Hunter

  * For a given "normal" data, the AI is rewarded based on how far the difference of the collected data compared to current data.

* Creating Curiosity-Based AI Bug Hunter

  * The AI will maximize actions that allows it to explore the game.
