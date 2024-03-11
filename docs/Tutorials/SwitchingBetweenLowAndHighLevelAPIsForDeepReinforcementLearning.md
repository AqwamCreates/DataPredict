Previously, you have saw that we commonly use reinforce() function. However, that is considered a high-level API function as that functions handle majority of the work. 

But if you wish to have more control over the model, we can use a lower level API function. There are two main functions that you need to remember: update() and episodeUpdate(). I'll show you on how to use those function below. Please do note that this code configuration is specific to QLearningNeuralNetwork and other models may need adjustments.

```lua

local QLearningNeuralNetwork = DataPredict.Models.QLearningNeuralNetwork.new()

QLearningNeuralNetwork:addLayer(5, true)

QLearningNeuralNetwork:addLayer(2, false)

QLearningNeuralNetwork:setClassesList({1, 2})

local maxNumberOfSteps = 100

while true do

  local previousEnvironmentVector = {{0, 0, 0, 0}}

  local action = 1

  for i = 1, maxNumberOfSteps, 1 do

    local environmentVector = fetchEnvironmentVector(action)

    action = QLearningNeuralNetwork:predict(environmentVector)

    local reward = getReward(environmentVector, action)

    QLearningNeuralNetwork:update(previousEnvironmentVector, reward, action, environmentVector)

    previousEnvironmentVector = environmentVector

  end

  QLearningNeuralNetwork:episodeUpdate()

end

```
