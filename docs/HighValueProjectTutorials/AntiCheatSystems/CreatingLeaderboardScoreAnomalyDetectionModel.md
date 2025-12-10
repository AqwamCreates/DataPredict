# Creating Leaderboard Score Anomaly Detection Model

Hello guys! Today, I will be showing you on how to create a leaderboard score anomaly detection model. To start, we need these two things:

* LocalOutlierProbability (Recommended) / LocalOutlierFactor model.

* Players' leaderboard score data.

## Designing O

### Model

```

-- The kValue determines how many neigbouring data points we want to compare to for each data points. For now, we will check for 3 players.

local AnomalyDetectionModel = DataPredict.Models.LocalOutlierProbability.new({kValue = 3})

-- Theoretically, you can also make it adaptive like this.

AnomalyDetectionModel.kValue = #Players:GetPlayers()

```

### Feature Matrix

```

-- Let's use a multiplayer obby game as our example.

local leaderboardScoreFeatureMatrix = {

  {timeCompleted, distanceTravelled, numberOfPowerUpsCollected}

}

```

###
