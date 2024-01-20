local AqwamMatrixLibrary = require(script.Parent.Parent.AqwamMatrixLibraryLinker.Value)

local ReinforcementLearningActorCriticNeuralNetworkBaseModel = require(script.Parent.ReinforcementLearningActorCriticNeuralNetworkBaseModel)

ProximalPolicyOptimizationClipModel = {}

ProximalPolicyOptimizationClipModel.__index = ProximalPolicyOptimizationClipModel

setmetatable(ProximalPolicyOptimizationClipModel, ReinforcementLearningActorCriticNeuralNetworkBaseModel)

local defaultClipRatio = 0.1

local function calculateProbability(outputMatrix)

	local sumVector = AqwamMatrixLibrary:horizontalSum(outputMatrix)

	local result = AqwamMatrixLibrary:divide(outputMatrix, sumVector)

	return result

end

local function calculateRewardsToGo(rewardHistory, discountFactor)

	local rewardsToGoArray = {}

	local discountedReward = 0

	for h = #rewardHistory, 1, -1 do

		discountedReward += rewardHistory[h] + (discountFactor * discountedReward)

		table.insert(rewardsToGoArray, 1, discountedReward)

	end

	return rewardsToGoArray

end

function ProximalPolicyOptimizationClipModel.new(numberOfReinforcementsPerEpisode, epsilon, epsilonDecayFactor, discountFactor, clipRatio)
	
	local NewProximalPolicyOptimizationClipModel = ReinforcementLearningActorCriticNeuralNetworkBaseModel.new(numberOfReinforcementsPerEpisode, epsilon, epsilonDecayFactor, discountFactor)
	
	setmetatable(NewProximalPolicyOptimizationClipModel, ProximalPolicyOptimizationClipModel)
	
	NewProximalPolicyOptimizationClipModel.clipRatio = clipRatio or defaultClipRatio
	
	local rewardHistory = {}
	
	local criticValueHistory = {}
	
	local actionVectorHistory = {}
	
	local advantageValueHistory = {}
	
	NewProximalPolicyOptimizationClipModel:setUpdateFunction(function(previousFeatureVector, action, rewardValue, currentFeatureVector)
		
		local allOutputsMatrix = NewProximalPolicyOptimizationClipModel.ActorModel:predict(previousFeatureVector, true)

		local actionProbabilityVector = calculateProbability(allOutputsMatrix)
		
		local CriticModel = NewProximalPolicyOptimizationClipModel.CriticModel

		local previousCriticValue = CriticModel:predict(previousFeatureVector, true)[1][1]

		local currentCriticValue = CriticModel:predict(currentFeatureVector, true)[1][1]

		local advantageValue = rewardValue + (NewProximalPolicyOptimizationClipModel.discountFactor * (currentCriticValue - previousCriticValue))

		table.insert(advantageValueHistory, advantageValue)

		table.insert(criticValueHistory, previousCriticValue)

		table.insert(actionVectorHistory, actionProbabilityVector)
		
		table.insert(rewardHistory, rewardValue)
		
	end)
	
	NewProximalPolicyOptimizationClipModel:setEpisodeUpdateFunction(function()
		
		local rewardsToGoArray = calculateRewardsToGo(rewardHistory, NewProximalPolicyOptimizationClipModel.discountFactor)

		local sumActorLossVector = AqwamMatrixLibrary:createMatrix(1, #NewProximalPolicyOptimizationClipModel.ClassesList)

		local historyLength = #criticValueHistory

		local sumCriticLoss = 0
		
		local clipFunction = function(value) 
			
			local clipRatio = NewProximalPolicyOptimizationClipModel.clipRatio 
			
			return math.clamp(value, 1 - clipRatio, 1 + clipRatio) 
			
		end

		for h = 1, historyLength - 1, 1 do

			local currentActionVector = actionVectorHistory[h + 1]

			local previousActionVector = actionVectorHistory[h]

			local ratioVector = AqwamMatrixLibrary:divide(currentActionVector, previousActionVector)
			
			local advantageValue = advantageValueHistory[h + 1]
			
			local surrogateLoss1 = AqwamMatrixLibrary:multiply(ratioVector, advantageValue)
			
			local surrogateLoss2Part1 = AqwamMatrixLibrary:applyFunction(clipFunction, ratioVector)
			
			local surrogateLoss2 = AqwamMatrixLibrary:multiply(surrogateLoss2Part1, advantageValue)

			local actorLossVector = AqwamMatrixLibrary:applyFunction(math.min, surrogateLoss1, surrogateLoss2)

			local criticLoss = math.pow(rewardsToGoArray[h] - criticValueHistory[h], 2)

			sumActorLossVector = AqwamMatrixLibrary:add(sumActorLossVector, actorLossVector)

			sumCriticLoss += criticLoss

		end

		local calculatedActorLossVector = AqwamMatrixLibrary:divide(sumActorLossVector, historyLength)

		local calculatedCriticLoss = sumCriticLoss / historyLength
		
		local ActorModel = NewProximalPolicyOptimizationClipModel.ActorModel
		
		local CriticModel = NewProximalPolicyOptimizationClipModel.CriticModel
		
		local numberOfFeatures, hasBias = ActorModel:getLayer(1)

		numberOfFeatures += (hasBias and 1) or 0

		local featureVector = AqwamMatrixLibrary:createMatrix(historyLength, numberOfFeatures, 1)

		ActorModel:forwardPropagate(featureVector, true)
		CriticModel:forwardPropagate(featureVector, true)

		ActorModel:backPropagate(calculatedActorLossVector, true)
		CriticModel:backPropagate(calculatedCriticLoss, true)
		
		table.clear(advantageValueHistory)

		table.clear(criticValueHistory)

		table.clear(rewardHistory)

		table.clear(actionVectorHistory)
		
	end)
	
	NewProximalPolicyOptimizationClipModel:extendResetFunction(function()
		
		table.clear(advantageValueHistory)

		table.clear(criticValueHistory)

		table.clear(rewardHistory)

		table.clear(actionVectorHistory)
		
	end)
	
	return NewProximalPolicyOptimizationClipModel
	
end

return ProximalPolicyOptimizationClipModel
