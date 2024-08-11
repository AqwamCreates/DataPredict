--[[

	--------------------------------------------------------------------

	Aqwam's Machine And Deep Learning Library (DataPredict)

	Author: Aqwam Harish Aiman
	
	YouTube: https://www.youtube.com/channel/UCUrwoxv5dufEmbGsxyEUPZw
	
	LinkedIn: https://www.linkedin.com/in/aqwam-harish-aiman/
	
	--------------------------------------------------------------------
		
	By using this library, you agree to comply with our Terms and Conditions in the link below:
	
	https://github.com/AqwamCreates/DataPredict/blob/main/docs/TermsAndConditions.md
	
	--------------------------------------------------------------------

--]]

local AqwamMatrixLibrary = require("AqwamMatrixLibrary")

local ReinforcementLearningActorCriticBaseModel = require("Model_ReinforcementLearningActorCriticBaseModel")

ProximalPolicyOptimizationModel = {}

ProximalPolicyOptimizationModel.__index = ProximalPolicyOptimizationModel

setmetatable(ProximalPolicyOptimizationModel, ReinforcementLearningActorCriticBaseModel)

local function calculateProbability(outputMatrix)

	local meanVector = AqwamMatrixLibrary:horizontalMean(outputMatrix)

	local standardDeviationVector = AqwamMatrixLibrary:horizontalStandardDeviation(outputMatrix)

	local zScoreVectorPart1 = AqwamMatrixLibrary:subtract(outputMatrix, meanVector)

	local zScoreVector = AqwamMatrixLibrary:divide(zScoreVectorPart1, standardDeviationVector)

	local zScoreSquaredVector = AqwamMatrixLibrary:power(zScoreVector, 2)

	local probabilityVectorPart1 = AqwamMatrixLibrary:multiply(-0.5, zScoreSquaredVector)

	local probabilityVectorPart2 = AqwamMatrixLibrary:applyFunction(math.exp, probabilityVectorPart1)

	local probabilityVectorPart3 = AqwamMatrixLibrary:multiply(standardDeviationVector, math.sqrt(2 * math.pi))

	local probabilityVector = AqwamMatrixLibrary:divide(probabilityVectorPart2, probabilityVectorPart3)

	return probabilityVector

end

local function calculateRewardsToGo(rewardHistory, discountFactor)

	local rewardsToGoArray = {}

	local discountedReward = 0

	for h = #rewardHistory, 1, -1 do

		discountedReward = rewardHistory[h] + (discountFactor * discountedReward)

		table.insert(rewardsToGoArray, 1, discountedReward)

	end

	return rewardsToGoArray

end

function ProximalPolicyOptimizationModel.new(discountFactor)
	
	local NewProximalPolicyOptimizationModel = ReinforcementLearningActorCriticBaseModel.new(discountFactor)
	
	setmetatable(NewProximalPolicyOptimizationModel, ProximalPolicyOptimizationModel)
	
	local rewardHistory = {}
	
	local criticValueHistory = {}
	
	local actionProbabilityVectorHistory = {}
	
	local oldActionProbabilityVectorHistory = {}
	
	local advantageValueHistory = {}
	
	local oldAdvantageValueHistory = {}
	
	NewProximalPolicyOptimizationModel:setUpdateFunction(function(previousFeatureVector, action, rewardValue, currentFeatureVector)
		
		local allOutputsMatrix = NewProximalPolicyOptimizationModel.ActorModel:predict(previousFeatureVector, true)

		local actionProbabilityVector = calculateProbability(allOutputsMatrix)
		
		local CriticModel = NewProximalPolicyOptimizationModel.CriticModel

		local previousCriticValue = CriticModel:predict(previousFeatureVector, true)[1][1]

		local currentCriticValue = CriticModel:predict(currentFeatureVector, true)[1][1]

		local advantageValue = rewardValue + (NewProximalPolicyOptimizationModel.discountFactor * currentCriticValue) - previousCriticValue

		table.insert(advantageValueHistory, advantageValue)

		table.insert(criticValueHistory, previousCriticValue)

		table.insert(actionProbabilityVectorHistory, actionProbabilityVector)
		
		table.insert(rewardHistory, rewardValue)
		
	end)
	
	NewProximalPolicyOptimizationModel:setEpisodeUpdateFunction(function()
		
		local ActorModel = NewProximalPolicyOptimizationModel.ActorModel

		local CriticModel = NewProximalPolicyOptimizationModel.CriticModel
		
		if (#oldActionProbabilityVectorHistory == 0) then 

			oldActionProbabilityVectorHistory = table.clone(actionProbabilityVectorHistory)
			
			oldAdvantageValueHistory = table.clone(advantageValueHistory)
			
			table.clear(advantageValueHistory)

			table.clear(criticValueHistory)

			table.clear(rewardHistory)

			table.clear(actionProbabilityVectorHistory)

			return 

		end
		
		local rewardsToGoArray = calculateRewardsToGo(rewardHistory, NewProximalPolicyOptimizationModel.discountFactor)

		local historyLength = #criticValueHistory
		
		local sumActorLossVector = AqwamMatrixLibrary:createMatrix(1, #ActorModel:getClassesList())
		
		local sumCriticLoss = 0

		for h = 1, historyLength, 1 do

			local currentActionVector = actionProbabilityVectorHistory[h]

			local previousActionVector = oldActionProbabilityVectorHistory[h]

			local ratioVector = AqwamMatrixLibrary:divide(currentActionVector, previousActionVector)

			local actorLossVector = AqwamMatrixLibrary:multiply(ratioVector, oldAdvantageValueHistory[h])

			local criticLoss = rewardsToGoArray[h] - criticValueHistory[h]

			sumActorLossVector = AqwamMatrixLibrary:add(sumActorLossVector, actorLossVector)

			sumCriticLoss += criticLoss

		end

		local calculatedActorLossVector = AqwamMatrixLibrary:divide(sumActorLossVector, historyLength)
		
		calculatedActorLossVector = AqwamMatrixLibrary:multiply(-1, calculatedActorLossVector)

		local calculatedCriticLoss = sumCriticLoss / historyLength
		
		local numberOfFeatures, hasBias = ActorModel:getLayer(1)

		numberOfFeatures += (hasBias and 1) or 0

		local featureVector = AqwamMatrixLibrary:createMatrix(1, numberOfFeatures, 1)

		ActorModel:forwardPropagate(featureVector, true)
		CriticModel:forwardPropagate(featureVector, true)

		ActorModel:backwardPropagate(calculatedActorLossVector, true)
		CriticModel:backwardPropagate(calculatedCriticLoss, true)
		
		oldActionProbabilityVectorHistory = table.clone(actionProbabilityVectorHistory)
		
		oldAdvantageValueHistory = table.clone(advantageValueHistory)
		
		table.clear(advantageValueHistory)

		table.clear(criticValueHistory)

		table.clear(rewardHistory)

		table.clear(actionProbabilityVectorHistory)
		
	end)
	
	NewProximalPolicyOptimizationModel:extendResetFunction(function()
		
		table.clear(advantageValueHistory)
		
		table.clear(oldAdvantageValueHistory)
		
		table.clear(criticValueHistory)
		
		table.clear(rewardHistory)
		
		table.clear(actionProbabilityVectorHistory)
		
		table.clear(oldActionProbabilityVectorHistory)
		
	end)
	
	return NewProximalPolicyOptimizationModel
	
end

return ProximalPolicyOptimizationModel