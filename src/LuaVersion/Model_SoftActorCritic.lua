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

local SoftActorCriticModel = {}

SoftActorCriticModel.__index = SoftActorCriticModel

setmetatable(SoftActorCriticModel, DeepReinforcementLearningActorCriticBaseModel)

local defaultAlpha = 0.1

local defaultAveragingRate = 0.995

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

function SoftActorCriticModel.new(parameterDictionary)
	
	parameterDictionary = parameterDictionary or {}
	
	local NewSoftActorCritic = DeepReinforcementLearningActorCriticBaseModel.new(parameterDictionary)
	
	setmetatable(NewSoftActorCritic, SoftActorCriticModel)
	
	NewSoftActorCritic:setName("SoftActorCritic")
	
	NewSoftActorCritic.alpha = parameterDictionary.alpha or defaultAlpha
	
	NewSoftActorCritic.averagingRate = parameterDictionary.averagingRate or defaultAveragingRate
	
	NewSoftActorCritic.CriticModelParametersArray = parameterDictionary.CriticModelParametersArray or {}
	
	NewSoftActorCritic:setCategoricalUpdateFunction(function(previousFeatureVector, previousAction, rewardValue, currentFeatureVector, currentAction, terminalStateValue)
		
		local ActorModel = NewSoftActorCritic.ActorModel

		local previousActionVector = ActorModel:forwardPropagate(previousFeatureVector, true)
		
		local currentActionVector = ActorModel:forwardPropagate(currentFeatureVector, true)

		local previousActionProbabilityVector = calculateCategoricalProbability(previousActionVector)
		
		local currentActionProbabilityVector = calculateCategoricalProbability(currentActionVector)
		
		local ClassesList = ActorModel:getClassesList()
		
		local previousClassIndex = table.find(ClassesList, previousAction)

		local previousActionProbabilityGradientVector = {}

		for i, _ in ipairs(ClassesList) do

			previousActionProbabilityGradientVector[i] = (((i == previousClassIndex) and 1) or 0) - previousActionProbabilityVector[1][i]

		end
		
		previousActionProbabilityGradientVector = {previousActionProbabilityGradientVector}

		local previousLogActionProbabilityVector = AqwamTensorLibrary:logarithm(previousActionProbabilityVector)
		
		local currentLogActionProbabilityVector = AqwamTensorLibrary:logarithm(currentActionProbabilityVector)
		
		return NewSoftActorCritic:update(previousFeatureVector, previousActionVector, previousLogActionProbabilityVector, previousActionProbabilityGradientVector, rewardValue, currentFeatureVector, currentActionVector, currentLogActionProbabilityVector, currentAction, terminalStateValue)
		
	end)
	
	NewSoftActorCritic:setDiagonalGaussianUpdateFunction(function(previousFeatureVector, previousActionMeanVector, previousActionStandardDeviationVector, previousActionNoiseVector, rewardValue, currentFeatureVector, currentActionMeanVector, terminalStateValue)
		
		if (not previousActionNoiseVector) then previousActionNoiseVector = AqwamTensorLibrary:createRandomNormalTensor({1, #previousActionMeanVector[1]}) end
		
		local dimensionSizeArray = AqwamTensorLibrary:getDimensionSizeArray(previousActionNoiseVector)
		
		local currentActionNoiseVector = AqwamTensorLibrary:createRandomUniformTensor(dimensionSizeArray)
		
		local previousActionVector = calculateActionVector(previousActionMeanVector, previousActionStandardDeviationVector, previousActionNoiseVector)
		
		local currentActionVector = calculateActionVector(currentActionMeanVector, previousActionStandardDeviationVector, currentActionNoiseVector)
		
		local previousLogActionProbabilityVector = calculateDiagonalGaussianProbability(previousActionMeanVector, previousActionStandardDeviationVector, previousActionNoiseVector)
		
		local currentLogActionProbabilityVector = calculateDiagonalGaussianProbability(currentActionMeanVector, previousActionStandardDeviationVector, currentActionNoiseVector)
		
		local previousActionProbabilityGradientVector = calculateDiagonalGaussianProbabilityGradient(previousActionMeanVector, previousActionStandardDeviationVector, previousActionNoiseVector)
		
		return NewSoftActorCritic:update(previousFeatureVector, previousActionVector, previousLogActionProbabilityVector, previousActionProbabilityGradientVector, rewardValue, currentFeatureVector, currentActionVector, currentLogActionProbabilityVector, nil, terminalStateValue)
		
	end)
	
	NewSoftActorCritic:setEpisodeUpdateFunction(function(terminalStateValue) end)
	
	NewSoftActorCritic:setResetFunction(function() end)
	
	return NewSoftActorCritic
	
end

function SoftActorCriticModel:update(previousFeatureVector, previousActionVector, previousLogActionProbabilityVector, previousActionProbabilityGradientVector, rewardValue, currentFeatureVector, currentActionVector, currentLogActionProbabilityVector, currentAction, terminalStateValue)
	
	local CriticModelParametersArray = self.CriticModelParametersArray
	
	local CriticModel = self.CriticModel

	local ActorModel = self.ActorModel

	local alpha = self.alpha
	
	local averagingRate = self.averagingRate
	
	local PreviousCriticModelParametersArray = {}
	
	local currentLogActionProbabilityValue
	
	if (currentAction) then
		
		local ClassesList = ActorModel:getClassesList()
		
		local actionIndex = table.find(ClassesList, currentAction)
		
		currentLogActionProbabilityValue = currentLogActionProbabilityVector[1][actionIndex]
		
	else
		
		currentLogActionProbabilityValue = AqwamTensorLibrary:sum(currentLogActionProbabilityVector)
		
	end

	local currentCriticValueArray = {}
	
	local concatenatedCurrentFeatureAndActionVector = AqwamTensorLibrary:concatenate(currentFeatureVector, currentActionVector, 2)

	for i = 1, 2, 1 do 

		CriticModel:setModelParameters(CriticModelParametersArray[i])

		currentCriticValueArray[i] = CriticModel:forwardPropagate(concatenatedCurrentFeatureAndActionVector)[1][1] 
		
		local CriticModelParameters = CriticModel:getModelParameters(true)
		
		PreviousCriticModelParametersArray[i] = CriticModelParameters

	end

	local minimumCurrentCriticValue = math.min(table.unpack(currentCriticValueArray))
	
	local yValuePart1 = (1 - terminalStateValue) * (minimumCurrentCriticValue - (alpha * currentLogActionProbabilityValue))

	local yValue = rewardValue + (self.discountFactor * yValuePart1)

	local temporalDifferenceErrorVector = AqwamTensorLibrary:createTensor({1, 2}, 0)
	
	local concatenatedPreviousFeatureAndActionVector = AqwamTensorLibrary:concatenate(previousFeatureVector, previousActionVector, 2)
	
	local previousCriticValueArray = {}

	for i = 1, 2, 1 do 

		CriticModel:setModelParameters(PreviousCriticModelParametersArray[i], true)

		local previousCriticValue = CriticModel:forwardPropagate(concatenatedPreviousFeatureAndActionVector, true)[1][1] 

		local criticLoss = previousCriticValue - yValue
		
		previousCriticValueArray[i] = previousCriticValue

		temporalDifferenceErrorVector[1][i] = -criticLoss -- We perform gradient descent here, so the critic loss is negated so that it can be used as temporal difference value.

		CriticModel:update(criticLoss, true)
		
		local TargetModelParameters = CriticModel:getModelParameters(true)
		
		CriticModelParametersArray[i] = rateAverageModelParameters(averagingRate, TargetModelParameters, PreviousCriticModelParametersArray[i])

	end
	
	local minimumPreviousCriticValue = math.min(table.unpack(previousCriticValueArray))

	local actorLossVectorPart1 = AqwamTensorLibrary:multiply(alpha, previousLogActionProbabilityVector)

	local actorLossVectorPart2 = AqwamTensorLibrary:subtract(minimumPreviousCriticValue, actorLossVectorPart1)
	
	local actorLossVector = AqwamTensorLibrary:multiply(actorLossVectorPart2, previousActionProbabilityGradientVector)
	
	actorLossVector = AqwamTensorLibrary:unaryMinus(actorLossVector)

	ActorModel:forwardPropagate(previousFeatureVector, true)

	ActorModel:update(actorLossVector, true)
	
	return temporalDifferenceErrorVector
	
end

function SoftActorCriticModel:setCriticModelParameters1(CriticModelParameters1, doNotDeepCopy)

	if (doNotDeepCopy) then

		self.CriticModelParametersArray[1] = CriticModelParameters1

	else

		self.CriticModelParametersArray[1] = self:deepCopyTable(CriticModelParameters1)

	end

end

function SoftActorCriticModel:setCriticModelParameters2(CriticModelParameters2, doNotDeepCopy)

	if (doNotDeepCopy) then

		self.CriticModelParametersArray[2] = CriticModelParameters2

	else

		self.CriticModelParametersArray[2] = self:deepCopyTable(CriticModelParameters2)

	end

end

function SoftActorCriticModel:getCriticModelParameters1(doNotDeepCopy)

	if (doNotDeepCopy) then

		return self.CriticModelParametersArray[1]

	else

		return self:deepCopyTable(self.CriticModelParametersArray[1])

	end

end

function SoftActorCriticModel:getCriticModelParameters2(doNotDeepCopy)

	if (doNotDeepCopy) then

		return self.CriticModelParametersArray[2]

	else

		return self:deepCopyTable(self.CriticModelParametersArray[2])

	end

end

return SoftActorCriticModel
