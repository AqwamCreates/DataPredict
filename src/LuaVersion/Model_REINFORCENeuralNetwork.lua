local ReinforcementLearningNeuralNetworkBaseModel = require("Model_ReinforcementLearningNeuralNetworkBaseModel")

local AqwamMatrixLibrary = require("AqwamMatrixLibrary")

REINFORCENeuralNetworkModel = {}

REINFORCENeuralNetworkModel.__index = REINFORCENeuralNetworkModel

setmetatable(REINFORCENeuralNetworkModel, ReinforcementLearningNeuralNetworkBaseModel)

local function calculateRewardsToGo(rewardHistory, discountFactor)

	local rewardsToGoArray = {}

	local discountedReward = 0

	for h = #rewardHistory, 1, -1 do

		discountedReward = rewardHistory[h] + (discountFactor * discountedReward)

		table.insert(rewardsToGoArray, 1, discountedReward)

	end

	return rewardsToGoArray

end

function VanillaPolicyGradientModel.new(discountFactor)
	
	local NewVanillaPolicyGradientModel = ReinforcementLearningActorCriticBaseModel.new(discountFactor)

	setmetatable(NewVanillaPolicyGradientModel, VanillaPolicyGradientModel)

	local rewardHistory = {}

	local gradientHistory = {}
	
	local valueHistory = {}

	NewVanillaPolicyGradientModel:setUpdateFunction(function(previousFeatureVector, action, rewardValue, currentFeatureVector)

		local allOutputsMatrix = NewVanillaPolicyGradientModel.ActorModel:predict(previousFeatureVector, true)

		local logOutputMatrix = AqwamMatrixLibrary:applyFunction(math.log, allOutputsMatrix)
		
		local CriticModel = NewVanillaPolicyGradientModel.CriticModel

		local previousCriticValue = CriticModel:predict(previousFeatureVector, true)[1][1]

		local currentCriticValue = CriticModel:predict(currentFeatureVector, true)[1][1]

		local advantageValue = rewardValue + (NewVanillaPolicyGradientModel.discountFactor * currentCriticValue) - previousCriticValue

		local gradientMatrix = AqwamMatrixLibrary:multiply(logOutputMatrix, advantageValue)
		
		table.insert(rewardHistory, rewardValue)
		
		table.insert(valueHistory, previousCriticValue)

		table.insert(gradientHistory, gradientMatrix[1])
		
		return advantageValue

	end)

	NewVanillaPolicyGradientModel:setEpisodeUpdateFunction(function()
		
		local rewardToGoArray = calculateRewardsToGo(rewardHistory, NewVanillaPolicyGradientModel.discountFactor)

		local sumGradient = AqwamMatrixLibrary:verticalSum(gradientHistory)
		
		local episodeLength = #rewardHistory
		
		sumGradient = AqwamMatrixLibrary:divide(sumGradient, episodeLength)
		
		local criticLoss = 0
		
		for i, value in ipairs(valueHistory) do
			
			local valueDifference = value - rewardToGoArray[i]
			
			criticLoss = criticLoss + math.pow(valueDifference, 2)
			
		end
		
		criticLoss = criticLoss / episodeLength
		
		criticLoss = {{criticLoss}}
		
		local ActorModel = NewVanillaPolicyGradientModel.ActorModel

		local CriticModel = NewVanillaPolicyGradientModel.CriticModel

		local numberOfFeatures = ActorModel:getTotalNumberOfNeurons(1)

		local numberOfLayers = ActorModel:getNumberOfLayers()

		local numberOfNeuronsAtFinalLayer = ActorModel:getTotalNumberOfNeurons(numberOfLayers)

		local featureVector = AqwamMatrixLibrary:createMatrix(1, numberOfFeatures, 1)

		local actorLossVector = AqwamMatrixLibrary:createMatrix(1, numberOfNeuronsAtFinalLayer, -sumGradient)

		ActorModel:forwardPropagate(featureVector, true)
		CriticModel:forwardPropagate(featureVector, true)

		ActorModel:backPropagate(-actorLossVector, true)
		CriticModel:backPropagate(criticLoss, true)
		
		table.clear(rewardHistory)
		
		table.clear(valueHistory)

		table.clear(gradientHistory)

	end)

	NewVanillaPolicyGradientModel:extendResetFunction(function()

		table.clear(rewardHistory)

		table.clear(valueHistory)

		table.clear(gradientHistory)

	end)
	
	return NewVanillaPolicyGradientModel
	
end

return VanillaPolicyGradientModel
