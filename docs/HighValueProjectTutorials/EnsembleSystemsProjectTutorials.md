# Ensemble Systems Project Tutorials

## Retention

| First Layer                           | Final Layer                  |
|---------------------------------------|------------------------------|
| Time-To-Leave Prediction Model        | Play Time Maximization Model |
| Probability-To-Leave Prediction Model |                              |

* Should the probability-to-leave be greater than 50%, it activates the "Play Time Maximization Model".

* Once "Play Time Maximization Model" chooses an event that it thinks it will increase play time, it will wait for the event's outcome based on time-to-leave value before receiving the rewards and update it.

* Unlike using "Play Time Maximization Model" by itself, introducing "probability-to-leave" as a trigger allows a more controlled exploration for "Play Time Maximization Model" as the lower "probability-to-leave" gets ignored. As a result, a more risky intervention is only applied when players are likely to leave.

* The first-layer model provides a strong signal about player state. Feeding that into the final layer means the "Play Time Maximization Model" learns in contextually meaningful situations, which improves its long-term performance.
