# Switching Between Low And High Level APIs For Deep Reinforcement Learning

Previously, you have saw that we commonly use reinforce() function. However, that is considered a high-level API function as that functions handle majority of the work. 

But if you wish to have more control over the model, we can use a lower level API function. There are two main functions that you need to remember: update() and episodeUpdate(). I'll show you on how to use those function below. 

Please do note that this code configuration is specific to QLearningNeuralNetwork and other models may need adjustments.

```lua

local QLearningNeuralNetwork = DataPredict.Models.QLearningNeuralNetwork.new()

QLearningNeuralNetwork:addLayer(4, true)

QLearningNeuralNetwork:addLayer(2, false)

QLearningNeuralNetwork:setClassesList({1, 2})

while true do

  local previousEnvironmentVector = {{1, 0, 0, 0, 0}}

  local action = 1

  local hasGameEnded = false

  repeat

    local environmentVector = fetchEnvironmentVector(previousEnvironmentVector, action)

    action = QLearningNeuralNetwork:predict(environmentVector)

    local reward = getReward(environmentVector, action)

    QLearningNeuralNetwork:update(previousEnvironmentVector, reward, action, environmentVector) -- update() is called whenever a step is made.

    previousEnvironmentVector = environmentVector

    hasGameEnded = checkIfGameHasEnded(environmentVector)

  until hasGameEnded

  QLearningNeuralNetwork:episodeUpdate() -- episodeUpdate() is used whenever an episode ends. An episode is the total number of steps that determines when the model should stop training.

end

```
