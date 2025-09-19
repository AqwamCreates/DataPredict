# Creating Play Time Maximization Ensemble Model

## High-Level Explanation

| First Layer                           | Final Layer                  |
|---------------------------------------|------------------------------|
| Time-To-Leave Prediction Model        | Play Time Maximization Model |
| Probability-To-Leave Prediction Model |                              |

* Should the probability-to-leave be greater than 50%, it activates the "Play Time Maximization Model".

* Once "Play Time Maximization Model" chooses an event that it thinks it will increase play time, it will wait for the event's outcome based on "time-to-leave" value before receiving the rewards and update it.

* Unlike using "Play Time Maximization Model" by itself, introducing "probability-to-leave" value as a trigger allows a more controlled exploration for "Play Time Maximization Model" as the lower "probability-to-leave" gets ignored. As a result, a more risky intervention is only applied when players are likely to leave.

* The first-layer model provides a strong signal about player state. Feeding that state into the final layer means the "Play Time Maximization Model" learns in contextually meaningful situations, which improves its long-term performance.

* The "Time-To-Leave Prediction Model" is in the same layer as "Probability-To-Leave Prediction Model" because we want it to constantly update on how long the player will stay. If we were to put it between the first and final layer, the updates will be too sparse to make accurate wait times for "Play Time Maximization Model".

## Code

```lua
```
