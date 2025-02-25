--[[

	--------------------------------------------------------------------

	Aqwam's Machine And Deep Learning Library (DataPredict)

	Author: Aqwam Harish Aiman
	
	Email: aqwam.harish.aiman@gmail.com
	
	YouTube: https://www.youtube.com/channel/UCUrwoxv5dufEmbGsxyEUPZw
	
	LinkedIn: https://www.linkedin.com/in/aqwam-harish-aiman/
	
	--------------------------------------------------------------------
		
	By using this library, you agree to comply with our Terms and Conditions in the link below:
	
	https://github.com/AqwamCreates/DataPredict/blob/main/docs/TermsAndConditions.md
	
	--------------------------------------------------------------------
	
	DO NOT REMOVE THIS TEXT!
	
	--------------------------------------------------------------------

--]]

local AqwamTensorLibrary = require("AqwamTensorLibrary")

local ReinforcementLearningActorCriticBaseModel = require("Model_ReinforcementLearningActorCriticBaseModel")

ProximalPolicyOptimizationModel = {}

ProximalPolicyOptimizationModel.__index = ProximalPolicyOptimizationModel

setmetatable(ProximalPolicyOptimizationModel, ReinforcementLearningActorCriticBaseModel)

local defaultLambda = 0

local function calculateProbability(valueVector)

	local zScoreVector, standardDeviationVector = AqwamTensorLibrary:zScoreNormalization(valueVector, 2)

	local squaredZScoreVector = AqwamTensorLibrary:power(zScoreVector, 2)

	local probabilityVectorPart1 = AqwamTensorLibrary:multiply(-0.5, squaredZScoreVector)

	local probabilityVectorPart2 = AqwamTensorLibrary:exponent(probabilityVectorPart1)

	local probabilityVectorPart3 = AqwamTensorLibrary:multiply(standardDeviationVector, math.sqrt(2 * math.pi))

	local probabilityVector = AqwamTensorLibrary:divide(probabilityVectorPart2, probabilityVectorPart3)

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

function ProximalPolicyOptimizationModel.new(parameterDictionary)
	
	parameterDictionary = parameterDictionary or {}
	
	local NewProximalPolicyOptimizationModel = ReinforcementLearningActorCriticBaseModel.new(parameterDictionary)
	
	setmetatable(NewProximalPolicyOptimizationModel, ProximalPolicyOptimizationModel)
	
	NewProximalPolicyOptimizationModel:setName("ProximalPolicyOptimization")
	
	NewProximalPolicyOptimizationModel.lambda = parameterDictionary.lambda or defaultLambda
	
	local rewardValueHistory = {}
	
	local criticValueHistory = {}
	
	local actionProbabilityVectorHistory = {}
	
	local oldActionProbabilityVectorHistory = {}
	
	local advantageValueHistory = {}
	
	local oldAdvantageValueHistory = {}
	
	NewProximalPolicyOptimizationModel:setCategoricalUpdateFunction(function(previousFeatureVector, action, rewardValue, currentFeatureVector, terminalStateValue)
		
		local CriticModel = NewProximalPolicyOptimizationModel.CriticModel
		
		local actionVector = NewProximalPolicyOptimizationModel.ActorModel:forwardPropagate(previousFeatureVector)

		local actionProbabilityVector = calculateProbability(actionVector)

		local previousCriticValue = CriticModel:forwardPropagate(previousFeatureVector)[1][1]

		local currentCriticValue = CriticModel:forwardPropagate(currentFeatureVector)[1][1]

		local advantageValue = rewardValue + (NewProximalPolicyOptimizationModel.discountFactor * (1 - terminalStateValue) * currentCriticValue) - previousCriticValue

		local logActionProbabilityVector = AqwamTensorLibrary:logarithm(actionProbabilityVector)
		
		table.insert(actionProbabilityVectorHistory, logActionProbabilityVector)

		table.insert(criticValueHistory, previousCriticValue)
		
		table.insert(advantageValueHistory, advantageValue)
		
		table.insert(rewardValueHistory, rewardValue)
		
		return advantageValue
		
	end)
	
	NewProximalPolicyOptimizationModel:setDiagonalGaussianUpdateFunction(function(previousFeatureVector, actionMeanVector, actionStandardDeviationVector, actionNoiseVector, rewardValue, currentFeatureVector, terminalStateValue)

		if (not actionNoiseVector) then actionNoiseVector = AqwamTensorLibrary:createRandomNormalTensor({1, #actionMeanVector[1]}) end
		
		local CriticModel = NewProximalPolicyOptimizationModel.CriticModel

		local actionVectorPart1 = AqwamTensorLibrary:multiply(actionStandardDeviationVector, actionNoiseVector)

		local actionVector = AqwamTensorLibrary:add(actionMeanVector, actionVectorPart1)

		local zScoreVectorPart1 = AqwamTensorLibrary:subtract(actionVector, actionMeanVector)

		local zScoreVector = AqwamTensorLibrary:divide(zScoreVectorPart1, actionStandardDeviationVector)

		local squaredZScoreVector = AqwamTensorLibrary:power(zScoreVector, 2)

		local logActionProbabilityVectorPart1 = AqwamTensorLibrary:logarithm(actionStandardDeviationVector)

		local logActionProbabilityVectorPart2 = AqwamTensorLibrary:multiply(2, logActionProbabilityVectorPart1)

		local logActionProbabilityVectorPart3 = AqwamTensorLibrary:add(squaredZScoreVector, logActionProbabilityVectorPart2)

		local logActionProbabilityVector = AqwamTensorLibrary:add(logActionProbabilityVectorPart3, math.log(2 * math.pi))

		local previousCriticValue = CriticModel:forwardPropagate(previousFeatureVector)[1][1]

		local currentCriticValue = CriticModel:forwardPropagate(currentFeatureVector)[1][1]

		local advantageValue = rewardValue + (NewProximalPolicyOptimizationModel.discountFactor * (1 - terminalStateValue) * currentCriticValue) - previousCriticValue
		
		table.insert(actionProbabilityVectorHistory, logActionProbabilityVector)

		table.insert(criticValueHistory, previousCriticValue)

		table.insert(advantageValueHistory, advantageValue)

		table.insert(rewardValueHistory, rewardValue)

		return advantageValue

	end)
	
	NewProximalPolicyOptimizationModel:setEpisodeUpdateFunction(function(terminalStateValue)
		
		local discountFactor = NewProximalPolicyOptimizationModel.discountFactor
		
		local lambda = NewProximalPolicyOptimizationModel.lambda
	
		if (lambda ~= 0) then

			local generalizedAdvantageEstimationValue = 0

			local generalizedAdvantageEstimationHistory = {}

			for t = #advantageValueHistory, 1, -1 do

				generalizedAdvantageEstimationValue = advantageValueHistory[t] + (discountFactor * lambda * generalizedAdvantageEstimationValue)

				table.insert(generalizedAdvantageEstimationHistory, 1, generalizedAdvantageEstimationValue) -- Insert at the beginning to maintain order

			end

			advantageValueHistory = generalizedAdvantageEstimationHistory

		end
		
		if (#oldActionProbabilityVectorHistory == 0) then 

			oldActionProbabilityVectorHistory = table.clone(actionProbabilityVectorHistory)
			
			oldAdvantageValueHistory = table.clone(advantageValueHistory)
			
			table.clear(actionProbabilityVectorHistory)
			
			table.clear(criticValueHistory)
			
			table.clear(advantageValueHistory)

			table.clear(rewardValueHistory)

			return nil

		end
		
		if (#actionProbabilityVectorHistory ~= #oldActionProbabilityVectorHistory) then error("The number of updates does not equal to the number of old updates!") end
		
		local ActorModel = NewProximalPolicyOptimizationModel.ActorModel

		local CriticModel = NewProximalPolicyOptimizationModel.CriticModel
		
		local rewardToGoArray = calculateRewardToGo(rewardValueHistory, discountFactor)

		local historyLength = #criticValueHistory
		
		local sumActorLossVector = AqwamTensorLibrary:createTensor({1, #actionProbabilityVectorHistory[1]}, 0)
		
		local sumCriticLoss = 0

		for h = 1, historyLength, 1 do

			local ratioVector = AqwamTensorLibrary:divide(actionProbabilityVectorHistory[h], oldActionProbabilityVectorHistory[h])

			local actorLossVector = AqwamTensorLibrary:multiply(ratioVector, oldAdvantageValueHistory[h])

			local criticLoss = criticValueHistory[h] - rewardToGoArray[h]

			sumActorLossVector = AqwamTensorLibrary:add(sumActorLossVector, actorLossVector)

			sumCriticLoss = sumCriticLoss + criticLoss

		end
		
		sumActorLossVector = AqwamTensorLibrary:divide(sumActorLossVector, -historyLength)
		
		sumCriticLoss = sumCriticLoss / historyLength

		local numberOfFeatures = ActorModel:getTotalNumberOfNeurons(1)

		local featureVector = AqwamTensorLibrary:createTensor({1, numberOfFeatures}, 1)

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