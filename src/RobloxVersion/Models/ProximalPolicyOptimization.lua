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

local AqwamTensorLibrary = require(script.Parent.Parent.AqwamTensorLibraryLinker.Value)

local DeepReinforcementLearningActorCriticBaseModel = require(script.Parent.DeepReinforcementLearningActorCriticBaseModel)

ProximalPolicyOptimizationModel = {}

ProximalPolicyOptimizationModel.__index = ProximalPolicyOptimizationModel

setmetatable(ProximalPolicyOptimizationModel, DeepReinforcementLearningActorCriticBaseModel)

local defaultClipRatio = 0.3

local defaultLambda = 0

local function calculateCategoricalProbability(valueVector)

	local highestActionValue = AqwamTensorLibrary:findMaximumValue(valueVector)

	local subtractedZVector = AqwamTensorLibrary:subtract(valueVector, highestActionValue)

	local exponentValueVector = AqwamTensorLibrary:applyFunction(math.exp, subtractedZVector)

	local exponentValueSumVector = AqwamTensorLibrary:sum(exponentValueVector, 2)

	local targetActionVector = AqwamTensorLibrary:divide(exponentValueVector, exponentValueSumVector)

	return targetActionVector

end

local function calculateDiagonalGaussianProbability(meanVector, standardDeviationVector, noiseVector)

	local valueVectorPart1 = AqwamTensorLibrary:multiply(standardDeviationVector, noiseVector)

	local valueVector = AqwamTensorLibrary:add(meanVector, valueVectorPart1)

	local zScoreVectorPart1 = AqwamTensorLibrary:subtract(valueVector, meanVector)

	local zScoreVector = AqwamTensorLibrary:divide(zScoreVectorPart1, standardDeviationVector)

	local squaredZScoreVector = AqwamTensorLibrary:power(zScoreVector, 2)

	local logValueVectorPart1 = AqwamTensorLibrary:logarithm(standardDeviationVector)

	local logValueVectorPart2 = AqwamTensorLibrary:multiply(2, logValueVectorPart1)

	local logValueVectorPart3 = AqwamTensorLibrary:add(squaredZScoreVector, logValueVectorPart2)

	local logValueVector = AqwamTensorLibrary:add(logValueVectorPart3, math.log(2 * math.pi))

	return logValueVector

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

	local NewProximalPolicyOptimizationModel = DeepReinforcementLearningActorCriticBaseModel.new(parameterDictionary)

	setmetatable(NewProximalPolicyOptimizationModel, ProximalPolicyOptimizationModel)

	NewProximalPolicyOptimizationModel:setName("ProximalPolicyOptimization")

	NewProximalPolicyOptimizationModel.lambda = parameterDictionary.lambda or defaultLambda

	NewProximalPolicyOptimizationModel.CurrentActorModelParameters = parameterDictionary.CurrentActorModelParameters

	NewProximalPolicyOptimizationModel.OldActorModelParameters = parameterDictionary.OldActorModelParameters

	local featureVectorHistory = {}

	local ratioActionProbabiltyVectorHistory = {}

	local rewardValueHistory = {}

	local criticValueHistory = {}

	local advantageValueHistory = {}

	NewProximalPolicyOptimizationModel:setCategoricalUpdateFunction(function(previousFeatureVector, action, rewardValue, currentFeatureVector, terminalStateValue)

		local ActorModel = NewProximalPolicyOptimizationModel.ActorModel

		local CriticModel = NewProximalPolicyOptimizationModel.CriticModel

		NewProximalPolicyOptimizationModel.CurrentActorModelParameters = ActorModel:getModelParameters(true)

		ActorModel:setModelParameters(NewProximalPolicyOptimizationModel.OldActorModelParameters, true)

		local oldPolicyActionVector = ActorModel:forwardPropagate(previousFeatureVector)

		NewProximalPolicyOptimizationModel.OldActorModelParameters = ActorModel:getModelParameters(true)

		local oldPolicyActionProbabilityVector = calculateCategoricalProbability(oldPolicyActionVector)

		ActorModel:setModelParameters(NewProximalPolicyOptimizationModel.CurrentActorModelParameters, true)

		local currentPolicyActionVector = ActorModel:forwardPropagate(previousFeatureVector)

		local currentPolicyActionProbabilityVector = calculateCategoricalProbability(currentPolicyActionVector)

		local ratioActionProbabiltyVector = AqwamTensorLibrary:divide(currentPolicyActionProbabilityVector, oldPolicyActionProbabilityVector)

		local previousCriticValue = CriticModel:forwardPropagate(previousFeatureVector)[1][1]

		local currentCriticValue = CriticModel:forwardPropagate(currentFeatureVector)[1][1]

		local advantageValue = rewardValue + (NewProximalPolicyOptimizationModel.discountFactor * (1 - terminalStateValue) * currentCriticValue) - previousCriticValue

		table.insert(featureVectorHistory, previousFeatureVector)

		table.insert(ratioActionProbabiltyVectorHistory, ratioActionProbabiltyVector)

		table.insert(rewardValueHistory, rewardValue)

		table.insert(criticValueHistory, previousCriticValue)

		table.insert(advantageValueHistory, advantageValue)

		return advantageValue

	end)

	NewProximalPolicyOptimizationModel:setDiagonalGaussianUpdateFunction(function(previousFeatureVector, actionMeanVector, actionStandardDeviationVector, actionNoiseVector, rewardValue, currentFeatureVector, terminalStateValue)

		if (not actionNoiseVector) then actionNoiseVector = AqwamTensorLibrary:createRandomNormalTensor({1, #actionMeanVector[1]}) end

		local ActorModel = NewProximalPolicyOptimizationModel.ActorModel

		local CriticModel = NewProximalPolicyOptimizationModel.CriticModel

		NewProximalPolicyOptimizationModel.CurrentActorModelParameters = ActorModel:getModelParameters(true)

		ActorModel:setModelParameters(NewProximalPolicyOptimizationModel.OldActorModelParameters, true)

		local oldPolicyActionMeanVector = ActorModel:forwardPropagate(previousFeatureVector)

		NewProximalPolicyOptimizationModel.OldActorModelParameters = ActorModel:getModelParameters(true)

		local oldPolicyActionProbabilityVector = calculateDiagonalGaussianProbability(oldPolicyActionMeanVector, actionStandardDeviationVector, actionNoiseVector)

		local currentPolicyActionProbabilityVector = calculateDiagonalGaussianProbability(actionMeanVector, actionStandardDeviationVector, actionNoiseVector)

		local ratioActionProbabiltyVector = AqwamTensorLibrary:divide(currentPolicyActionProbabilityVector, oldPolicyActionProbabilityVector)

		local previousCriticValue = CriticModel:forwardPropagate(previousFeatureVector)[1][1]

		local currentCriticValue = CriticModel:forwardPropagate(currentFeatureVector)[1][1]

		local advantageValue = rewardValue + (NewProximalPolicyOptimizationModel.discountFactor * (1 - terminalStateValue) * currentCriticValue) - previousCriticValue

		table.insert(featureVectorHistory, previousFeatureVector)

		table.insert(ratioActionProbabiltyVectorHistory, ratioActionProbabiltyVector)

		table.insert(rewardValueHistory, rewardValue)

		table.insert(criticValueHistory, previousCriticValue)

		table.insert(advantageValueHistory, advantageValue)

		return advantageValue

	end)

	NewProximalPolicyOptimizationModel:setEpisodeUpdateFunction(function(terminalStateValue)

		local ActorModel = NewProximalPolicyOptimizationModel.ActorModel

		local CriticModel = NewProximalPolicyOptimizationModel.CriticModel

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

		local rewardToGoArray = calculateRewardToGo(rewardValueHistory, discountFactor)

		NewProximalPolicyOptimizationModel.OldActorModelParameters = NewProximalPolicyOptimizationModel.CurrentActorModelParameters

		ActorModel:setModelParameters(NewProximalPolicyOptimizationModel.CurrentActorModelParameters, true)

		for h, featureVector in ipairs(featureVectorHistory) do

			local ratioActionProbabilityVector = ratioActionProbabiltyVectorHistory[h]

			local advantageValue = advantageValueHistory[h]

			local actorLossVector = AqwamTensorLibrary:multiply(ratioActionProbabilityVector, advantageValue)

			local criticLoss = criticValueHistory[h] - rewardToGoArray[h]

			actorLossVector = AqwamTensorLibrary:unaryMinus(actorLossVector)

			ActorModel:forwardPropagate(featureVector, true)

			CriticModel:forwardPropagate(featureVector, true)

			ActorModel:update(actorLossVector, true)

			CriticModel:update(criticLoss, true)

		end

		NewProximalPolicyOptimizationModel.CurrentActorModelParameters = ActorModel:getModelParameters(true)

		table.clear(featureVectorHistory)

		table.clear(ratioActionProbabiltyVectorHistory)

		table.clear(rewardValueHistory)

		table.clear(criticValueHistory)

		table.clear(advantageValueHistory)

	end)

	NewProximalPolicyOptimizationModel:setResetFunction(function()

		table.clear(featureVectorHistory)

		table.clear(ratioActionProbabiltyVectorHistory)

		table.clear(rewardValueHistory)

		table.clear(criticValueHistory)

		table.clear(advantageValueHistory)

	end)

	return NewProximalPolicyOptimizationModel

end

return ProximalPolicyOptimizationModel
