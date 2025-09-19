# Ensemble Systems Project Tutorials

## Retention

| First Layer                           | Final Layer                  |
|---------------------------------------|------------------------------|
| Time-To-Leave Prediction Model        | Play Time Maximization Model |
| Probability-To-Leave Prediction Model |                              |

* Should the probability-to-leave be greater than 50%, it activates the "Play Time Maximization Model".

* Once "Play Time Maximization Model" chooses an event that it thinks it will increase play time, it will wait for the event's outcome based on time-to-leave value before receiving the rewards and update it.
