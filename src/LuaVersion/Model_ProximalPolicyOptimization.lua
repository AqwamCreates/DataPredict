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

local function calculateProbability(vector)

	local zScoreVector, standardDeviationVector = AqwamMatrixLibrary:horizontalZScoreNormalization(vector)

	local squaredZScoreVector = AqwamMatrixLibrary:power(zScoreVector, 2)

	local probabilityVectorPart1 = AqwamMatrixLibrary:multiply(-0.5, squaredZScoreVector)

	local probabilityVectorPart2 = AqwamMatrixLibrary:exponent(probabilityVectorPart1)

	local probabilityVectorPart3 = AqwamMatrixLibrary:multiply(standardDeviationVector, math.sqrt(2 * math.pi))

	local probabilityVector = AqwamMatrixLibrary:divide(probabilityVectorPart2, probabilityVectorPart3)

	return probabilityVector

end

local function calculateRewardToGo(rewardHistory, discountFactor)

	local rewardToGoArray = {}

	local discountedReward = 0

	for h = #rewardHistory, 1, -1 do

		discountedReward = rewardHistory[h] + (discountFactor * discountedReward)

		table.insert(rewardToGoArray, 1, discountedReward)

	end

	return rewardToGoArray

end

function ProximalPolicyOptimizationModel.new(discountFactor)
	
	local NewProximalPolicyOptimizationModel = ReinforcementLearningActorCriticBaseModel.new(discountFactor)
	
	setmetatable(NewProximalPolicyOptimizationModel, ProximalPolicyOptimizationModel)
	
	local rewardValueHistory = {}
	
	local criticValueHistory = {}
	
	local actionProbabilityVectorHistory = {}
	
	local oldActionProbabilityVectorHistory = {}
	
	local advantageValueHistory = {}
	
	local oldAdvantageValueHistory = {}
	
	NewProximalPolicyOptimizationModel:setCategoricalUpdateFunction(function(previousFeatureVector, action, rewardValue, currentFeatureVector)
		
		local CriticModel = NewProximalPolicyOptimizationModel.CriticModel
		
		local actionVector = NewProximalPolicyOptimizationModel.ActorModel:forwardPropagate(previousFeatureVector)

		local actionProbabilityVector = calculateProbability(actionVector)

		local previousCriticValue = CriticModel:forwardPropagate(previousFeatureVector)[1][1]

		local currentCriticValue = CriticModel:forwardPropagate(currentFeatureVector)[1][1]

		local advantageValue = rewardValue + (NewProximalPolicyOptimizationModel.discountFactor * currentCriticValue) - previousCriticValue

		local logActionProbabilityVector = AqwamMatrixLibrary:logarithm(actionProbabilityVector)
		
		table.insert(actionProbabilityVectorHistory, logActionProbabilityVector)

		table.insert(criticValueHistory, previousCriticValue)
		
		table.insert(advantageValueHistory, advantageValue)
		
		table.insert(rewardValueHistory, rewardValue)
		
		return advantageValue
		
	end)
	
	NewProximalPolicyOptimizationModel:setDiagonalGaussianUpdateFunction(function(previousFeatureVector, actionMeanVector, actionStandardDeviationVector, rewardValue, currentFeatureVector)

		local CriticModel = NewProximalPolicyOptimizationModel.CriticModel

		local randomNormalVector = AqwamMatrixLibrary:createRandomNormalMatrix(1, #actionMeanVector[1])

		local actionVectorPart1 = AqwamMatrixLibrary:multiply(actionStandardDeviationVector, randomNormalVector)

		local actionVector = AqwamMatrixLibrary:add(actionMeanVector, actionVectorPart1)

		local zScoreVectorPart1 = AqwamMatrixLibrary:subtract(actionVector, actionMeanVector)

		local zScoreVector = AqwamMatrixLibrary:divide(zScoreVectorPart1, actionStandardDeviationVector)

		local squaredZScoreVector = AqwamMatrixLibrary:power(zScoreVector, 2)

		local logActionProbabilityVectorPart1 = AqwamMatrixLibrary:logarithm(actionStandardDeviationVector)

		local logActionProbabilityVectorPart2 = AqwamMatrixLibrary:multiply(2, logActionProbabilityVectorPart1)

		local logActionProbabilityVectorPart3 = AqwamMatrixLibrary:add(squaredZScoreVector, logActionProbabilityVectorPart2)

		local logActionProbabilityVector = AqwamMatrixLibrary:add(logActionProbabilityVectorPart3, math.log(2 * math.pi))

		local previousCriticValue = CriticModel:forwardPropagate(previousFeatureVector)[1][1]

		local currentCriticValue = CriticModel:forwardPropagate(currentFeatureVector)[1][1]

		local advantageValue = rewardValue + (NewProximalPolicyOptimizationModel.discountFactor * currentCriticValue) - previousCriticValue
		
		table.insert(actionProbabilityVectorHistory, logActionProbabilityVector)

		table.insert(criticValueHistory, previousCriticValue)

		table.insert(advantageValueHistory, advantageValue)

		table.insert(rewardValueHistory, rewardValue)

		return advantageValue

	end)
	
	NewProximalPolicyOptimizationModel:setEpisodeUpdateFunction(function()
		
		local ActorModel = NewProximalPolicyOptimizationModel.ActorModel

		local CriticModel = NewProximalPolicyOptimizationModel.CriticModel
		
		if (#oldActionProbabilityVectorHistory == 0) then 

			oldActionProbabilityVectorHistory = table.clone(actionProbabilityVectorHistory)
			
			oldAdvantageValueHistory = table.clone(advantageValueHistory)
			
			table.clear(actionProbabilityVectorHistory)
			
			table.clear(criticValueHistory)
			
			table.clear(advantageValueHistory)

			table.clear(rewardValueHistory)

			return 

		end
		
		local rewardToGoArray = calculateRewardToGo(rewardValueHistory, NewProximalPolicyOptimizationModel.discountFactor)

		local historyLength = #criticValueHistory
		
		local sumActorLossVector = AqwamMatrixLibrary:createMatrix(1, #actionProbabilityVectorHistory[1], 0)
		
		local sumCriticLoss = 0

		for h = 1, historyLength, 1 do

			local ratioVector = AqwamMatrixLibrary:divide(actionProbabilityVectorHistory[h], oldActionProbabilityVectorHistory[h])

			local actorLossVector = AqwamMatrixLibrary:multiply(ratioVector, oldAdvantageValueHistory[h])

			local criticLoss = criticValueHistory[h] - rewardToGoArray[h]

			sumActorLossVector = AqwamMatrixLibrary:add(sumActorLossVector, actorLossVector)

			sumCriticLoss = sumCriticLoss + criticLoss

		end
		
		sumActorLossVector = AqwamMatrixLibrary:divide(sumActorLossVector, -historyLength)
		
		sumCriticLoss = sumCriticLoss / historyLength

		local numberOfFeatures = ActorModel:getTotalNumberOfNeurons(1)

		local featureVector = AqwamMatrixLibrary:createMatrix(1, numberOfFeatures, 1)

		ActorModel:forwardPropagate(featureVector, true, true)
		
		CriticModel:forwardPropagate(featureVector, true, true)

		ActorModel:backwardPropagate(sumActorLossVector, true)
		
		CriticModel:backwardPropagate(sumCriticLoss, true)
		
		oldActionProbabilityVectorHistory = table.clone(actionProbabilityVectorHistory)
		
		oldAdvantageValueHistory = table.clone(advantageValueHistory)
		
		table.clear(actionProbabilityVectorHistory)

		table.clear(criticValueHistory)

		table.clear(advantageValueHistory)

		table.clear(rewardValueHistory)
		
	end)
	
	NewProximalPolicyOptimizationModel:setResetFunction(function()
		
		table.clear(actionProbabilityVectorHistory)
		
		table.clear(oldActionProbabilityVectorHistory)
		
		table.clear(criticValueHistory)
		
		table.clear(advantageValueHistory)
		
		table.clear(oldAdvantageValueHistory)
		
		table.clear(rewardValueHistory)
		
	end)
	
	return NewProximalPolicyOptimizationModel
	
end

return ProximalPolicyOptimizationModel