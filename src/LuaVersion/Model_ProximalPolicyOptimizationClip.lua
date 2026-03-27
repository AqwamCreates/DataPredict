--[[

	--------------------------------------------------------------------

	Aqwam's Machine, Deep And Reinforcement Learning Library (DataPredict)

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

local DeepReinforcementLearningActorCriticBaseModel = require("Model_DeepReinforcementLearningActorCriticBaseModel")

local ProximalPolicyOptimizationClipModel = {}

ProximalPolicyOptimizationClipModel.__index = ProximalPolicyOptimizationClipModel

setmetatable(ProximalPolicyOptimizationClipModel, DeepReinforcementLearningActorCriticBaseModel)

local defaultEpsilon = 0.3

local defaultLambda = 0

local defaultUseLogProbabilities = true

local function rateAverageModelParameters(averagingRate, TargetModelParameters, PrimaryModelParameters)

	local averagingRateComplement = 1 - averagingRate

	for layer = 1, #TargetModelParameters, 1 do

		local TargetModelParametersPart = AqwamTensorLibrary:multiply(averagingRate, TargetModelParameters[layer])

		local PrimaryModelParametersPart = AqwamTensorLibrary:multiply(averagingRateComplement, PrimaryModelParameters[layer])

		TargetModelParameters[layer] = AqwamTensorLibrary:add(TargetModelParametersPart, PrimaryModelParametersPart)

	end

	return TargetModelParameters

end

local function calculateCategoricalProbability(valueTensor)

	local highestActionValue = AqwamTensorLibrary:findMaximumValue(valueTensor)

	local subtractedZVector = AqwamTensorLibrary:subtract(valueTensor, highestActionValue)

	local exponentActionVector = AqwamTensorLibrary:applyFunction(math.exp, subtractedZVector)

	local exponentActionSumVector = AqwamTensorLibrary:sum(exponentActionVector, 2)

	local targetActionTensor = AqwamTensorLibrary:divide(exponentActionVector, exponentActionSumVector)

	return targetActionTensor

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

	logValueVector = AqwamTensorLibrary:multiply(-0.5, logValueVector)

	return logValueVector

end

local function calculateDiagonalGaussianProbabilityGradient(meanVector, standardDeviationVector, noiseVector)

	local actionVectorPart1 = AqwamTensorLibrary:multiply(standardDeviationVector, noiseVector)

	local actionVector = AqwamTensorLibrary:add(meanVector, actionVectorPart1)

	local actionProbabilityGradientVectorPart1 = AqwamTensorLibrary:subtract(actionVector, meanVector)

	local actionProbabilityGradientVectorPart2 = AqwamTensorLibrary:power(standardDeviationVector, 2)

	local actionProbabilityGradientVector = AqwamTensorLibrary:divide(actionProbabilityGradientVectorPart1, actionProbabilityGradientVectorPart2)

	return actionProbabilityGradientVector

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

local function calculateActorLossValue(ratioActionProbabilityValue, advantageValue, actorGradientValue, epsilon)
	
	local upperRatioActionProbabilityValue = 1 + epsilon
	
	local lowerRatioActionProbabilityValue = 1 - epsilon
	
	local clippedRatioActionProbabilityValue = math.clamp(ratioActionProbabilityValue, lowerRatioActionProbabilityValue, upperRatioActionProbabilityValue)
	
	local unclippedAdvantageValue = ratioActionProbabilityValue * advantageValue
	
	local clippedAdvantageValue = clippedRatioActionProbabilityValue * advantageValue
	
	local isUnclippedAdvantageValueIsUsed = (unclippedAdvantageValue <= clippedAdvantageValue)
	
	local isRatioActionProbabilityValueNotClipped = (ratioActionProbabilityValue >= lowerRatioActionProbabilityValue) and (ratioActionProbabilityValue <= upperRatioActionProbabilityValue)
	
	if (isUnclippedAdvantageValueIsUsed) or (isRatioActionProbabilityValueNotClipped) then return -(ratioActionProbabilityValue * advantageValue * actorGradientValue) end
	
	return 0

end

function ProximalPolicyOptimizationClipModel.new(parameterDictionary)

	parameterDictionary = parameterDictionary or {}

	local NewProximalPolicyOptimizationClipModel = DeepReinforcementLearningActorCriticBaseModel.new(parameterDictionary)

	setmetatable(NewProximalPolicyOptimizationClipModel, ProximalPolicyOptimizationClipModel)

	NewProximalPolicyOptimizationClipModel:setName("ProximalPolicyOptimizationClip")
	
	NewProximalPolicyOptimizationClipModel.epsilon = parameterDictionary.epsilon or defaultEpsilon

	NewProximalPolicyOptimizationClipModel.lambda = parameterDictionary.lambda or defaultLambda

	NewProximalPolicyOptimizationClipModel.useLogProbabilities = NewProximalPolicyOptimizationClipModel:getValueOrDefaultValue(parameterDictionary.useLogProbabilities, defaultUseLogProbabilities)

	NewProximalPolicyOptimizationClipModel.OldActorModelParameters = parameterDictionary.OldActorModelParameters

	local featureVectorHistory = {}
	
	local ratioActionProbabilityVectorHistory = {}

	local actorGradientVectorHistory = {}

	local rewardValueHistory = {}

	local criticValueHistory = {}

	local advantageValueHistory = {}

	NewProximalPolicyOptimizationClipModel:setCategoricalUpdateFunction(function(previousFeatureVector, previousAction, rewardValue, currentFeatureVector, currentAction, terminalStateValue)
		
		local ActorModel = NewProximalPolicyOptimizationClipModel.ActorModel

		local CriticModel = NewProximalPolicyOptimizationClipModel.CriticModel

		local CurrentActorModelParameters = ActorModel:getModelParameters(true)

		local OldModelParameters = NewProximalPolicyOptimizationClipModel.OldActorModelParameters or CurrentActorModelParameters

		ActorModel:setModelParameters(OldModelParameters, true)

		local oldPolicyActionVector = ActorModel:forwardPropagate(previousFeatureVector)

		ActorModel:setModelParameters(CurrentActorModelParameters, true)

		local currentPolicyActionVector = ActorModel:forwardPropagate(previousFeatureVector)

		local oldPolicyActionProbabilityVector = calculateCategoricalProbability(oldPolicyActionVector)

		local currentPolicyActionProbabilityVector = calculateCategoricalProbability(currentPolicyActionVector)

		local ClassesList = ActorModel:getClassesList()

		local previousActionIndex = table.find(ClassesList, previousAction)
		
		local unwrappedCurrentPolicyActionProbabilityVector = currentPolicyActionProbabilityVector[1]

		local currentPolicyActionProbability = unwrappedCurrentPolicyActionProbabilityVector[previousActionIndex]

		local oldPolicyActionProbability = oldPolicyActionProbabilityVector[1][previousActionIndex]

		local ratioActionProbability

		if (NewProximalPolicyOptimizationClipModel.useLogProbabilities) then

			ratioActionProbability = math.exp(math.log(currentPolicyActionProbability) - math.log(oldPolicyActionProbability))

		else

			ratioActionProbability = currentPolicyActionProbability / oldPolicyActionProbability

		end
		
		local ratioActionProbabilityVector = AqwamTensorLibrary:createTensor({1, #ClassesList}, ratioActionProbability)

		local previousActionProbabilityGradientVector = {}

		for i, _ in ipairs(ClassesList) do

			previousActionProbabilityGradientVector[i] = (((i == previousActionIndex) and 1) or 0) - unwrappedCurrentPolicyActionProbabilityVector[i]

		end
		
		previousActionProbabilityGradientVector = {previousActionProbabilityGradientVector}

		local previousCriticValue = CriticModel:forwardPropagate(previousFeatureVector)[1][1]

		local currentCriticValue = CriticModel:forwardPropagate(currentFeatureVector)[1][1]

		local advantageValue = rewardValue + (NewProximalPolicyOptimizationClipModel.discountFactor * (1 - terminalStateValue) * currentCriticValue) - previousCriticValue
		
		table.insert(featureVectorHistory, previousFeatureVector)
		
		table.insert(ratioActionProbabilityVectorHistory, ratioActionProbabilityVector)

		table.insert(actorGradientVectorHistory, previousActionProbabilityGradientVector)

		table.insert(rewardValueHistory, rewardValue)

		table.insert(criticValueHistory, previousCriticValue)

		table.insert(advantageValueHistory, advantageValue)

		return advantageValue

	end)

	NewProximalPolicyOptimizationClipModel:setDiagonalGaussianUpdateFunction(function(previousFeatureVector, previousActionMeanVector, previousActionStandardDeviationVector, previousActionNoiseVector, rewardValue, currentFeatureVector, currentActionMeanVector, terminalStateValue)

		if (not previousActionNoiseVector) then previousActionNoiseVector = AqwamTensorLibrary:createRandomNormalTensor({1, #previousActionMeanVector[1]}) end
		
		local epsilon = NewProximalPolicyOptimizationClipModel.epsilon

		local ActorModel = NewProximalPolicyOptimizationClipModel.ActorModel

		local CriticModel = NewProximalPolicyOptimizationClipModel.CriticModel

		local CurrentActorModelParameters = ActorModel:getModelParameters(true)

		local OldModelParameters = NewProximalPolicyOptimizationClipModel.OldActorModelParameters or CurrentActorModelParameters

		ActorModel:setModelParameters(OldModelParameters, true)

		local oldPolicyActionMeanVector = ActorModel:forwardPropagate(previousFeatureVector)

		ActorModel:setModelParameters(CurrentActorModelParameters, true)

		local oldPolicyActionProbabilityVector = calculateDiagonalGaussianProbability(oldPolicyActionMeanVector, previousActionStandardDeviationVector, previousActionNoiseVector)

		local currentPolicyActionProbabilityVector = calculateDiagonalGaussianProbability(previousActionMeanVector, previousActionStandardDeviationVector, previousActionNoiseVector)

		local ratioActionProbabiltyVector

		if (NewProximalPolicyOptimizationClipModel.useLogProbabilities) then

			ratioActionProbabiltyVector = AqwamTensorLibrary:applyFunction(math.exp, AqwamTensorLibrary:subtract(currentPolicyActionProbabilityVector, oldPolicyActionProbabilityVector))

		else

			currentPolicyActionProbabilityVector = AqwamTensorLibrary:applyFunction(math.exp, currentPolicyActionProbabilityVector)

			oldPolicyActionProbabilityVector = AqwamTensorLibrary:applyFunction(math.exp, oldPolicyActionProbabilityVector)

			ratioActionProbabiltyVector = AqwamTensorLibrary:divide(currentPolicyActionProbabilityVector, oldPolicyActionProbabilityVector)

		end
		
		local previousActionProbabilityGradientVector = calculateDiagonalGaussianProbabilityGradient(previousActionMeanVector, previousActionStandardDeviationVector, previousActionNoiseVector)

		local previousCriticValue = CriticModel:forwardPropagate(previousFeatureVector)[1][1]

		local currentCriticValue = CriticModel:forwardPropagate(currentFeatureVector)[1][1]

		local advantageValue = rewardValue + (NewProximalPolicyOptimizationClipModel.discountFactor * (1 - terminalStateValue) * currentCriticValue) - previousCriticValue
		
		table.insert(featureVectorHistory, previousFeatureVector)
		
		table.insert(ratioActionProbabilityVectorHistory, ratioActionProbabiltyVector)

		table.insert(actorGradientVectorHistory, previousActionProbabilityGradientVector)

		table.insert(rewardValueHistory, rewardValue)

		table.insert(criticValueHistory, previousCriticValue)

		table.insert(advantageValueHistory, advantageValue)

		return advantageValue

	end)

	NewProximalPolicyOptimizationClipModel:setEpisodeUpdateFunction(function(terminalStateValue)

		local ActorModel = NewProximalPolicyOptimizationClipModel.ActorModel

		local CriticModel = NewProximalPolicyOptimizationClipModel.CriticModel
		
		local epsilon = NewProximalPolicyOptimizationClipModel.epsilon

		local discountFactor = NewProximalPolicyOptimizationClipModel.discountFactor

		local lambda = NewProximalPolicyOptimizationClipModel.lambda
		
		local ClassesList = ActorModel:getClassesList()
		
		local numberOfClasses = #ClassesList
		
		local outputDimensionSizeArray = {1, numberOfClasses}
		
		local epsilonVector = AqwamTensorLibrary:createTensor(outputDimensionSizeArray, epsilon)

		local CurrentActorModelParameters = ActorModel:getModelParameters(true)
		
		NewProximalPolicyOptimizationClipModel.OldActorModelParameters = CurrentActorModelParameters

		if (lambda ~= 0) then

			local generalizedAdvantageEstimationValue = 0

			local generalizedAdvantageEstimationHistory = {}

			for t = #advantageValueHistory, 1, -1 do

				generalizedAdvantageEstimationValue = advantageValueHistory[t] + (discountFactor * lambda * generalizedAdvantageEstimationValue)

				table.insert(generalizedAdvantageEstimationHistory, 1, generalizedAdvantageEstimationValue)

			end

			advantageValueHistory = generalizedAdvantageEstimationHistory

		end

		local rewardToGoArray = calculateRewardToGo(rewardValueHistory, discountFactor)

		for h, featureVector in ipairs(featureVectorHistory) do
			
			local ratioActionProbabilityVector = ratioActionProbabilityVectorHistory[h]

			local actorGradientVector = actorGradientVectorHistory[h]

			local advantageValue = advantageValueHistory[h]

			local criticLoss = criticValueHistory[h] - rewardToGoArray[h]
			
			local advantageVector = AqwamTensorLibrary:createTensor(outputDimensionSizeArray, advantageValue)
			
			local actorLossVector = AqwamTensorLibrary:applyFunction(calculateActorLossValue, ratioActionProbabilityVector, advantageVector, actorGradientVector, epsilonVector)

			ActorModel:forwardPropagate(featureVector, true)

			CriticModel:forwardPropagate(featureVector, true)

			ActorModel:update(actorLossVector, true)

			CriticModel:update(criticLoss, true)

		end

		table.clear(featureVectorHistory)
		
		table.clear(ratioActionProbabilityVectorHistory)

		table.clear(actorGradientVectorHistory)

		table.clear(rewardValueHistory)

		table.clear(criticValueHistory)

		table.clear(advantageValueHistory)

	end)

	NewProximalPolicyOptimizationClipModel:setResetFunction(function()

		table.clear(featureVectorHistory)
		
		table.clear(ratioActionProbabilityVectorHistory)

		table.clear(actorGradientVectorHistory)

		table.clear(rewardValueHistory)

		table.clear(criticValueHistory)

		table.clear(advantageValueHistory)

	end)

	return NewProximalPolicyOptimizationClipModel

end

return ProximalPolicyOptimizationClipModel
