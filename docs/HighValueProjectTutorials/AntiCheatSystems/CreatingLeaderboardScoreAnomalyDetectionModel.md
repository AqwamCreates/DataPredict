# Creating Leaderboard Score Anomaly Detection Model

Hello guys! Today, I will be showing you on how to create a leaderboard score anomaly detection model. To start, we need these two things:

* LocalOutlierProbability (Recommended) / LocalOutlierFactor model.

* Players' leaderboard score data.

## Setting Up

### Model

```lua

-- The kValue determines how many neigbouring data points we want to compare to for each data points. For now, we will check against 3 other players.

local AnomalyDetectionModel = DataPredict.Models.LocalOutlierProbability.new({kValue = 3})

-- Theoretically, you can also make it adaptive like this.

AnomalyDetectionModel.kValue = #Players:GetPlayers()

```

### Feature Matrix

```lua

-- Let's use a multiplayer obby game as our example.

local leaderboardScoreFeatureMatrix = {

  {timeCompleted, distanceTravelled, numberOfPowerUpsCollected}

}

```

## Anomaly Detection

```lua

local probabilityValueThreshold = 0.3

local playerArray = {}

local leaderboardScoreFeatureMatrix = {}

local currentIndex = 1

local function onPlayerFinished(Player)

  local leaderBoardScoreFeatureVector = getPlayerLeaderboardScoreFeatureVector(Player)

  leaderboardScoreFeatureMatrix[currentIndex] = leaderBoardScoreFeatureVector

  playerArray[currentIndex] = Player

  currentIndex + 1

end

local function onRoundEnd()

    AnomalyDetectionModel:train(leaderboardScoreFeatureMatrix)

    local probabilityVector = AnomalyDetectionModel:score()

    local probabilityValue

    for playerIndex = #playerArray, 1, -1 do

      -- The probability value here means how likely it is that the data point is "normal" in relative to its neighbours. It is based on kValue.

      probabilityValue = probabilityVector[playerIndex][1]

      -- Above this probability value threshold, we consider them as normal.

      if (probabilityValue >= probabilityValueThreshold) then continue end 

      -- Otherwise, remove this data from the leaderboard.

      table.remove(playerArray, playerIndex)

      table.remove(leaderboardScoreFeatureMatrix, playerIndex)
      
    end

    displayLeaderboardScores(playerArray, leaderboardScoreFeatureMatrix)

    AnomalyDetectionModel:setModelParameters(nil) -- To reset the model.

end

```

That is all for today!
