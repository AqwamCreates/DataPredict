# High-Value Project Tutorials

> Beat The Retention! Not The Algorithm! 

> Keep Players, Grow Your Game, Forget Competitors! 

> Make Games That Adapts To Players, Not The Other Way Around!

> And Keep The Source Of Retention And Revenue Out Of Competitors' Sight!

## Disclaimer

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

| System                                                                                           | Properties                                                                                         | Minimal Implementation Time |
|--------------------------------------------------------------------------------------------------|----------------------------------------------------------------------------------------------------|-----------------------------|
| [Retention](HighValueProjectTutorials/RetentionSystems.md)                                       | Handles play time maximization and leave predictions.                                              | 30 minutes - 6 hours       |
| [Recommendation](HighValueProjectTutorials/RecommendationSystems.md)                             | Improves the likelihood of selling items.                                                          | 2 hours                   |
| [Dynamic Difficulty Adjustment](HighValueProjectTutorials/DynamicDifficultyAdjustmentSystems.md) | Ensures that the player is likely to interact with the enemies.                                    | 30 minutes - 2 hours       |
| [Team Balancing](HighValueProjectTutorials/TeamBalancingSystems.md)                              | Ensures that teams are balanced based on their combat data.                                        | 30 minutes                 |
| [Targeting](HighValueProjectTutorials/TargetingSystems.md)                                       | Maximizes the likelihood of hitting clusters of players.                                           | 30 minutes - 1 hour        |
| [AI Player](HighValueProjectTutorials/AIPlayerSystems.md)                                        | Replace players if players aren't enough.                                                          | 30 minutes - 1 hour        |
| [Quality Assurance](HighValueProjectTutorials/QualityAssuranceSystems.md) (Incomplete)           | Attempts to break game mechanics to find edge cases and exploits.                                  | To be added soon           |
| [Priority](HighValueProjectTutorials/PrioritySystems.md) (Incomplete)                            | Making sure that the players or items are queued based on their data.                              | To be added soon           |
| [Anti-Cheat](HighValueProjectTutorials/AntiCheatSystems.md)                                      | Handles the separation of unusual players from normal players without changing the game mechanics. | 30 minutes                 |
| [Placement](HighValueProjectTutorials/PlacementSystems.md)                                       | Finds an empty space to place players or items.                                                    | 30 minutes                 |
