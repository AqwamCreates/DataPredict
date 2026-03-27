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

local AqwamTensorLibrary = require(script.Parent.Parent.AqwamTensorLibraryLinker.Value)

local DeepReinforcementLearningActorCriticBaseModel = require(script.Parent.DeepReinforcementLearningActorCriticBaseModel)

local ProximalPolicyOptimizationModel = {}

ProximalPolicyOptimizationModel.__index = ProximalPolicyOptimizationModel

setmetatable(ProximalPolicyOptimizationModel, DeepReinforcementLearningActorCriticBaseModel)

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

local function calculateActionVector(meanVector, standardDeviationVector, noiseVector)

	local actionVectoPart1 = AqwamTensorLibrary:multiply(standardDeviationVector, noiseVector)

	local actionVector = AqwamTensorLibrary:add(meanVector, actionVectoPart1)

	return actionVector

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

function ProximalPolicyOptimizationModel.new(parameterDictionary)

	parameterDictionary = parameterDictionary or {}

	local NewProximalPolicyOptimizationModel = DeepReinforcementLearningActorCriticBaseModel.new(parameterDictionary)

	setmetatable(NewProximalPolicyOptimizationModel, ProximalPolicyOptimizationModel)

	NewProximalPolicyOptimizationModel:setName("ProximalPolicyOptimization")

	NewProximalPolicyOptimizationModel.lambda = parameterDictionary.lambda or defaultLambda
	
	NewProximalPolicyOptimizationModel.useLogProbabilities = NewProximalPolicyOptimizationModel:getValueOrDefaultValue(parameterDictionary.useLogProbabilities, defaultUseLogProbabilities)

	NewProximalPolicyOptimizationModel.OldActorModelParameters = parameterDictionary.OldActorModelParameters

	local featureVectorHistory = {}

	local actorGradientVectorHistory = {}

	local rewardValueHistory = {}

	local criticValueHistory = {}

	local advantageValueHistory = {}

	NewProximalPolicyOptimizationModel:setCategoricalUpdateFunction(function(previousFeatureVector, previousAction, rewardValue, currentFeatureVector, currentAction, terminalStateValue)

		local ActorModel = NewProximalPolicyOptimizationModel.ActorModel

		local CriticModel = NewProximalPolicyOptimizationModel.CriticModel
		
		local CurrentActorModelParameters = ActorModel:getModelParameters(true)
		
		local OldModelParameters = NewProximalPolicyOptimizationModel.OldActorModelParameters or CurrentActorModelParameters

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
		
		if (NewProximalPolicyOptimizationModel.useLogProbabilities) then
			
			ratioActionProbability = math.exp(math.log(currentPolicyActionProbability) - math.log(oldPolicyActionProbability))
			
		else
			
			ratioActionProbability = currentPolicyActionProbability / oldPolicyActionProbability
			
		end
		
		local previousActionProbabilityGradientVector = {}

		for i, _ in ipairs(ClassesList) do

			previousActionProbabilityGradientVector[i] = (((i == previousActionIndex) and 1) or 0) - unwrappedCurrentPolicyActionProbabilityVector[i]

		end

		previousActionProbabilityGradientVector = {previousActionProbabilityGradientVector}
		
		previousActionProbabilityGradientVector = AqwamTensorLibrary:multiply(previousActionProbabilityGradientVector, ratioActionProbability)

		local previousCriticValue = CriticModel:forwardPropagate(previousFeatureVector)[1][1]

		local currentCriticValue = CriticModel:forwardPropagate(currentFeatureVector)[1][1]

		local advantageValue = rewardValue + (NewProximalPolicyOptimizationModel.discountFactor * (1 - terminalStateValue) * currentCriticValue) - previousCriticValue

		table.insert(featureVectorHistory, previousFeatureVector)

		table.insert(actorGradientVectorHistory, previousActionProbabilityGradientVector)

		table.insert(rewardValueHistory, rewardValue)

		table.insert(criticValueHistory, previousCriticValue)

		table.insert(advantageValueHistory, advantageValue)

		return advantageValue

	end)

	NewProximalPolicyOptimizationModel:setDiagonalGaussianUpdateFunction(function(previousFeatureVector, previousActionMeanVector, previousActionStandardDeviationVector, previousActionNoiseVector, rewardValue, currentFeatureVector, currentActionMeanVector, terminalStateValue)

		if (not previousActionNoiseVector) then previousActionNoiseVector = AqwamTensorLibrary:createRandomNormalTensor({1, #previousActionMeanVector[1]}) end

		local ActorModel = NewProximalPolicyOptimizationModel.ActorModel

		local CriticModel = NewProximalPolicyOptimizationModel.CriticModel

		local CurrentActorModelParameters = ActorModel:getModelParameters(true)
		
		local OldModelParameters = NewProximalPolicyOptimizationModel.OldActorModelParameters or CurrentActorModelParameters

		ActorModel:setModelParameters(OldModelParameters, true)

		local oldPolicyActionMeanVector = ActorModel:forwardPropagate(previousFeatureVector)
		
		ActorModel:setModelParameters(CurrentActorModelParameters, true)

		local oldPolicyActionProbabilityVector = calculateDiagonalGaussianProbability(oldPolicyActionMeanVector, previousActionStandardDeviationVector, previousActionNoiseVector)

		local currentPolicyActionProbabilityVector = calculateDiagonalGaussianProbability(previousActionMeanVector, previousActionStandardDeviationVector, previousActionNoiseVector)
		
		local ratioActionProbabiltyVector
		
		if (NewProximalPolicyOptimizationModel.useLogProbabilities) then

			ratioActionProbabiltyVector = AqwamTensorLibrary:applyFunction(math.exp, AqwamTensorLibrary:subtract(currentPolicyActionProbabilityVector, oldPolicyActionProbabilityVector))

		else
			
			currentPolicyActionProbabilityVector = AqwamTensorLibrary:applyFunction(math.exp, currentPolicyActionProbabilityVector)
			
			oldPolicyActionProbabilityVector = AqwamTensorLibrary:applyFunction(math.exp, oldPolicyActionProbabilityVector)

			ratioActionProbabiltyVector = AqwamTensorLibrary:divide(currentPolicyActionProbabilityVector, oldPolicyActionProbabilityVector)

		end
		
		local previousActionProbabilityGradientVector = calculateDiagonalGaussianProbabilityGradient(previousActionMeanVector, previousActionStandardDeviationVector, previousActionNoiseVector)
		
		previousActionProbabilityGradientVector = AqwamTensorLibrary:multiply(previousActionProbabilityGradientVector, ratioActionProbabiltyVector)
		
		local previousCriticValue = CriticModel:forwardPropagate(previousFeatureVector)[1][1]

		local currentCriticValue = CriticModel:forwardPropagate(currentFeatureVector)[1][1]

		local advantageValue = rewardValue + (NewProximalPolicyOptimizationModel.discountFactor * (1 - terminalStateValue) * currentCriticValue) - previousCriticValue

		table.insert(featureVectorHistory, previousFeatureVector)

		table.insert(actorGradientVectorHistory, previousActionProbabilityGradientVector)

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
		
		local CurrentActorModelParameters = ActorModel:getModelParameters(true)

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

			local actorGradientVector = actorGradientVectorHistory[h]

			local advantageValue = advantageValueHistory[h]

			local actorLossVector = AqwamTensorLibrary:multiply(actorGradientVector, advantageValue)

			local criticLoss = criticValueHistory[h] - rewardToGoArray[h]

			actorLossVector = AqwamTensorLibrary:unaryMinus(actorLossVector)

			ActorModel:forwardPropagate(featureVector, true)

			CriticModel:forwardPropagate(featureVector, true)

			ActorModel:update(actorLossVector, true)

			CriticModel:update(criticLoss, true)

		end

		table.clear(featureVectorHistory)

		table.clear(actorGradientVectorHistory)

		table.clear(rewardValueHistory)

		table.clear(criticValueHistory)

		table.clear(advantageValueHistory)
		
		NewProximalPolicyOptimizationModel.OldActorModelParameters = CurrentActorModelParameters

	end)

	NewProximalPolicyOptimizationModel:setResetFunction(function()

		table.clear(featureVectorHistory)

		table.clear(actorGradientVectorHistory)

		table.clear(rewardValueHistory)

		table.clear(criticValueHistory)

		table.clear(advantageValueHistory)

	end)

	return NewProximalPolicyOptimizationModel

end

return ProximalPolicyOptimizationModel
