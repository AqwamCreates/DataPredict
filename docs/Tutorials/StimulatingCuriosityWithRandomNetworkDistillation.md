# Stimulating Curiosity With Random Network Distillation

Majority of the time, rewards are received from the environment when performing certain actions. These are usually handcrafted by programmers. We call them external rewards.

However, what if there are no external rewards, but we want to encourage our AIs to explore a map?

Then, we need to find a way for our AIs to be "curious".

# The Random Network Distillation

Random Network Distillation uses neural networks to generate internal rewards to stimulate curiousity. At the start, it will encourage exploration. But over time, it sees more than enough similar data and this will discourage exploration.

Under here, this is how you integrate your reinforcement learning algorithms with the random network distillation.

```lua

-- Initializing our RandomNetworkDistillation and QLearningNeuralNetwork

local RandomNetworkDistillation = DataPredict.Others.RandomNetworkDistillation.new()

RandomNetworkDistillation:addLayer(10, true, "LeakyReLU")

RandomNetworkDistillation:addLayer(1, true, "Sigmoid")

local QLearningNeuralNetwork = DataPredict.Model.QLearning.new()

QLearningNeuralNetwork:addLayer(10, true, "LeakyReLU")

QLearningNeuralNetwork:addLayer(4, true, "StableSoftmax")

-- Creating A simple function when receiving environment vector received.

local function onEnvironmentVectorReceived(environmentVector)

  local internalReward = RandomNetworkDistillation:generateReward()

  local action = QLearningNeuralNetwork:reinforce(environmentVector, internalReward)

  return action

end

```

As you can see, creating a random network distillation object is pretty similar to creating neural networks. The difference lies on the number of output it produces, where the random network distillation always produce one output.

That's all for today!
