# Creating Engagement Milestone Detection Model

## High-Level Explanation

| First Layer                           | Final Layer                    |
|---------------------------------------|--------------------------------|
| Probability-To-Leave Prediction Model | Left-Too-Early Detection Model |
| Time-To-Leave Prediction Model        |                                |

* Should the probability-to-leave be greater than 50% or "time-to-leave" is less than 5 seconds, it activates the "Play Time Maximization Model". For the latter metric, even if the "Probability-To-Leave Prediction Model" says the player is unlikely to leave, we still have a chance that the player will leave in near term within a short period of time and the effects of player leaving is generally permanent.

* Once "Play Time Maximization Model" chooses an event that it thinks it will increase play time, it will wait for the event's outcome based on "time-to-leave" value before receiving the rewards and update it.

* Unlike using "Play Time Maximization Model" by itself, introducing "probability-to-leave" value as a trigger allows a more controlled exploration for "Play Time Maximization Model" as the low "probability-to-leave" value and high "time-to-leave" gets ignored. As a result, a more risky intervention is only applied when players are likely to leave.

* The first-layer model provides a strong signal about player state. Feeding that state into the final layer means the "Play Time Maximization Model" learns in contextually meaningful situations, which improves its long-term performance.

* The "Time-To-Leave Prediction Model" is in the same layer as "Probability-To-Leave Prediction Model" because we want it to constantly update on how long the player will stay. If we were to put it between the first and final layer, the updates will be too sparse to predict accurate wait times for "Play Time Maximization Model".

## Code

### Feature Vector And Classes List Design

```lua

local function getPlayerDataVector(Player)

  -- We have five features with one "bias".

  return {
    {
        1,
        numberOfCurrencyAmount,
        numberOfItemsAmount,
        timePlayedInCurrentSession,
        timePlayedInAllSessions,
        healthAmount
    }
  }

end

```

### Creating Time-To-Leave And Probability-To-Leave Prediction Models

```lua

local TimeToLeavePredictionModel = DataPredict.Models.LinearRegression.new({maximumNumberOfIterations = 100, learningRate = 0.01})

local ProbabilityToLeavePredictionModel = DataPredict.Models.LogisticRegression.new({maximumNumberOfIterations = 100, learningRate = 0.01})

local LeftToEarlyPredictionModel = DataPredict.Models.LogisticRegression.new({maximumNumberOfIterations = 100, beta = 1, kernelFunction = "RadialBasisFunction"})

-- The code shown below checks if we already have trained the models previously.

if (TimeToLeavePredictionModelParameters) then TimeToLeavePredictionModel:setModelParameters(TimeToLeavePredictionModelParameters) end

if (ProbabilityToLeavePredictionModelParameters) then ProbabilityToLeavePredictionModel:setModelParameters(ProbabilityToLeavePredictionModelParameters) end

if (LeftToEarlyPredictionModelParameters) then LeftToEarlyPredictionModel:setModelParameters(LeftToEarlyPredictionModelParameters) end

```

### Constructing Play Time Maximization Model

### Player Data Collection

```lua

local playerDataMatrix = {}
  
local recordedTimeArray = {}
  
local snapshotIndex = 1

local function getPlayerDataArray()

  return {1, numberOfCurrencyAmount, numberOfItemsAmount, timePlayedInCurrentSession, timePlayedInAllSessions, healthAmount}

end
  
local function snapshotData(playerDataArray)
  
 playerDataMatrix[snapshotIndex] = playerDataArray
  
  recordedTimeArray[snapshotIndex] = os.time()
  
  snapshotIndex = snapshotIndex + 1

end

```

### On Player Join

```lua

local function run(Player)

    local isPlayerInServer = true

    local rewardValue = 0

    local playerDataArray

    local playerDataVector

    local predictedTimeToLeave

    local predictedProbabilityToLeave

    local eventName

    local eventFunction

    local activatePlayTimeMaximization

    while isPlayerInServer do

        playerDataArray = getPlayerDataArray(Player)

        snapshotData(playerDataArray)

        playerDataVector = {playerDataArray}

        predictedTimeToLeave = TimeToLeavePredictionModel:predict(playerDataArray)[1][1]

        predictedProbabilityToLeave = ProbabilityToLeavePredictionModel:predict(playerDataArray)[1][1]

        activatePlayTimeMaximization = (predictedProbabilityToLeave >= 0.5) or (predictedTimeToLeave <= 5)

        if (activatePlayTimeMaximization) then

          eventName = PlayTimeMaximizationModel:reinforce(playerDataVector, rewardValue)

          eventFunction = eventFunctionDictionary[eventName]

          if (eventFunction) then eventFunction() end

        end

        task.wait(predictedTimeToLeave)

        isPlayerInServer = checkIfPlayerIsInServer(Player)

        if (activatePlayTimeMaximization) then

          -- Player leaving the game is more of a "rarer" and "extremely undesirable" event, therefore a very large negative value is used.\

          rewardValue = (isPlayerInServer and 20) or -100

        end

    end

end

```

### On Player Leave

```lua

local timeToLeaveVector = {}

local probabilityToLeaveVector = {}

for i = 1, snapshotIndex, 1 do

  local timeToLeave = os.time() - recordedTime[i]

  -- To ensure that this does not result in negative probabilityToLeave value if we're using sigmoid function for our logistic regression.

  local clampedTimeToLeave = math.max(timeToLeave, 1)

  local probabilityToLeave = 1 - (1 / clampedTimeToLeave)

  timeToLeaveVector[i] = {timeToLeave}

  probabilityToLeaveVector[i] = {probabilityToLeave}

end

TimeToLeavePredictionModel:train(playerDataMatrix, timeToLeaveVector)

ProbabilityToLeavePredictionModel:train(playerDataMatrix, probabilityToLeaveVector)

LeftToEarlyPredictionModel:train(playerDataMatrix, probabilityToLeaveVector)

-- Just getting our model parameters to save them

TimeToLeavePredictionModelParameters = TimeToLeavePredictionModel:getModelParameters(true)

ProbabilityToLeavePredictionModelParameters = ProbabilityToLeavePredictionModel:getModelParameters(true)

LeftToEarlyPredictionModelParameters = LeftToEarlyPredictionModel:getModelParameters(true)

```
