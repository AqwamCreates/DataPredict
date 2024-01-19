local AqwamMatrixLibrary = require("AqwamMatrixLibrary")

local ReinforcementLearningActorCriticNeuralNetworkBaseModel = require("Model_ReinforcementLearningActorCriticNeuralNetworkBaseModel")

ProximalPolicyOptimizationModel = {}

ProximalPolicyOptimizationModel.__index = ProximalPolicyOptimizationModel

setmetatable(ProximalPolicyOptimizationModel, ReinforcementLearningActorCriticNeuralNetworkBaseModel)

local function calculateProbability(outputMatrix)

	local sumVector = AqwamMatrixLibrary:horizontalSum(outputMatrix)

	local result = AqwamMatrixLibrary:divide(outputMatrix, sumVector)

	return result

end

function ProximalPolicyOptimizationModel:calculateRewardsToGo()

	local rewardsToGoArray = {}

	local discountedReward = 0

	local rewardHistory = self.rewardHistory

	for h = #rewardHistory, 1, -1 do

		discountedReward += rewardHistory[h] + (self.discountFactor * discountedReward)

		table.insert(rewardsToGoArray, 1, discountedReward)

	end

	return rewardsToGoArray

end

function ProximalPolicyOptimizationModel.new(numberOfReinforcementsPerEpisode, epsilon, epsilonDecayFactor, discountFactor)
	
	local NewProximalPolicyOptimizationModel = ReinforcementLearningActorCriticNeuralNetworkBaseModel.new(numberOfReinforcementsPerEpisode, epsilon, epsilonDecayFactor, discountFactor)
	
	setmetatable(NewProximalPolicyOptimizationModel, ProximalPolicyOptimizationModel)
	
	local criticValueHistory = {}
	
	local actionVectorHistory = {}
	
	local advantageValueHistory = {}
	
	NewProximalPolicyOptimizationModel:setUpdateFunction(function(previousFeatureVector, action, rewardValue, currentFeatureVector)
		
		local allOutputsMatrix = NewProximalPolicyOptimizationModel.ActorModel:predict(previousFeatureVector, true)

		local actionProbabilityVector = calculateProbability(allOutputsMatrix)

		local previousCriticValue = NewProximalPolicyOptimizationModel.CriticModel:predict(previousFeatureVector, true)[1][1]

		local currentCriticValue = NewProximalPolicyOptimizationModel.CriticModel:predict(currentFeatureVector, true)[1][1]

		local advantageValue = rewardValue + (NewProximalPolicyOptimizationModel.discountFactor * (currentCriticValue - previousCriticValue))

		table.insert(advantageValueHistory, advantageValue)

		table.insert(criticValueHistory, previousCriticValue)

		table.insert(actionVectorHistory, actionProbabilityVector)
		
	end)
	
	NewProximalPolicyOptimizationModel:setEpisodeUpdateFunction(function()
		
		local rewardsToGoArray = NewProximalPolicyOptimizationModel:calculateRewardsToGo()

		local sumActorLossVector = AqwamMatrixLibrary:createMatrix(1, #NewProximalPolicyOptimizationModel.ClassesList)

		local historyLength = #criticValueHistory

		local sumCriticLoss = 0

		for h = 1, historyLength - 1, 1 do

			local currentActionVector = actionVectorHistory[h + 1]

			local previousActionVector = actionVectorHistory[h]

			local ratioVector = AqwamMatrixLibrary:divide(currentActionVector, previousActionVector)

			local actorLossVector = AqwamMatrixLibrary:multiply(ratioVector, advantageValueHistory[h + 1])

			local criticLoss = math.pow(rewardsToGoArray[h] - criticValueHistory[h], 2)

			sumActorLossVector = AqwamMatrixLibrary:add(sumActorLossVector, actorLossVector)

			sumCriticLoss += criticLoss

		end

		local calculatedActorLossVector = AqwamMatrixLibrary:divide(sumActorLossVector, historyLength)

		local calculatedCriticLossVector = sumCriticLoss / historyLength
		
		local numberOfFeatures, hasBias = NewProximalPolicyOptimizationModel.ActorModel:getLayer(1)

		numberOfFeatures += (hasBias and 1) or 0

		local featureVector = AqwamMatrixLibrary:createMatrix(historyLength, numberOfFeatures, 1)

		NewProximalPolicyOptimizationModel.ActorModel:forwardPropagate(featureVector, true)
		NewProximalPolicyOptimizationModel.CriticModel:forwardPropagate(featureVector, true)

		NewProximalPolicyOptimizationModel.ActorModel:backPropagate(calculatedActorLossVector, true)
		NewProximalPolicyOptimizationModel.CriticModel:backPropagate(calculatedCriticLossVector, true)
		
		table.clear(advantageValueHistory)

		table.clear(criticValueHistory)

		table.clear(actionVectorHistory)
		
	end)
	
	NewProximalPolicyOptimizationModel:extendResetFunction(function()

		table.clear(advantageValueHistory)

		table.clear(criticValueHistory)

		table.clear(actionVectorHistory)
		
	end)
	
	return NewProximalPolicyOptimizationModel
	
end

return ProximalPolicyOptimizationModel
