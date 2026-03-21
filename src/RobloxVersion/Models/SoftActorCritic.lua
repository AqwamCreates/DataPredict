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
	
	NewSoftActorCritic.PrimaryCriticModelParametersArray = parameterDictionary.PrimaryCriticModelParametersArray or {}
	
	NewSoftActorCritic.TargetCriticModelParametersArray = parameterDictionary.TargetCriticModelParametersArray or {}
	
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
	
	local CriticModel = self.CriticModel

	local ActorModel = self.ActorModel

	local alpha = self.alpha
	
	local averagingRate = self.averagingRate
	
	local PrimaryCriticModelParametersArray = self.PrimaryCriticModelParametersArray

	local TargetCriticModelParametersArray = self.TargetCriticModelParametersArray
	
	PrimaryCriticModelParametersArray[1] = PrimaryCriticModelParametersArray[1] or CriticModel:generateLayers()
	
	PrimaryCriticModelParametersArray[2] = PrimaryCriticModelParametersArray[2] or CriticModel:generateLayers()
	
	TargetCriticModelParametersArray[1] = TargetCriticModelParametersArray[1] or PrimaryCriticModelParametersArray[1]
	
	TargetCriticModelParametersArray[2] = TargetCriticModelParametersArray[2] or PrimaryCriticModelParametersArray[2]
	
	local currentLogActionProbabilityValue
	
	if (currentAction) then
		
		local ClassesList = ActorModel:getClassesList()
		
		local actionIndex = table.find(ClassesList, currentAction)
		
		currentLogActionProbabilityValue = currentLogActionProbabilityVector[1][actionIndex]
		
	else
		
		currentLogActionProbabilityValue = AqwamTensorLibrary:sum(currentLogActionProbabilityVector)
		
	end

	local targetCurrentCriticValueArray = {}
	
	local concatenatedCurrentFeatureAndActionVector = AqwamTensorLibrary:concatenate(currentFeatureVector, currentActionVector, 2)

	for i = 1, 2, 1 do 

		CriticModel:setModelParameters(TargetCriticModelParametersArray[i])

		targetCurrentCriticValueArray[i] = CriticModel:forwardPropagate(concatenatedCurrentFeatureAndActionVector)[1][1]

	end

	local minimumTargetCurrentCriticValue = math.min(table.unpack(targetCurrentCriticValueArray))
	
	local yValuePart1 = (1 - terminalStateValue) * (minimumTargetCurrentCriticValue - (alpha * currentLogActionProbabilityValue))

	local yValue = rewardValue + (self.discountFactor * yValuePart1)

	local temporalDifferenceErrorVector = AqwamTensorLibrary:createTensor({1, 2}, 0)
	
	local concatenatedPreviousFeatureAndActionVector = AqwamTensorLibrary:concatenate(previousFeatureVector, previousActionVector, 2)
	
	local previousCriticValueArray = {}

	for i = 1, 2, 1 do

		CriticModel:setModelParameters(PrimaryCriticModelParametersArray[i], true)

		local primaryPreviousCriticValue = CriticModel:forwardPropagate(concatenatedPreviousFeatureAndActionVector, true)[1][1] 

		local criticLoss = 2 * (primaryPreviousCriticValue - yValue)
		
		previousCriticValueArray[i] = primaryPreviousCriticValue

		temporalDifferenceErrorVector[1][i] = -criticLoss -- We perform gradient descent here, so the critic loss is negated so that it can be used as temporal difference value.

		CriticModel:update(criticLoss, true)
		
		local TargetModelParameters = TargetCriticModelParametersArray[i]
		
		local PrimaryModelParameters = CriticModel:getModelParameters(true)
		
		PrimaryCriticModelParametersArray[i] = PrimaryModelParameters
		
		TargetCriticModelParametersArray[i] = rateAverageModelParameters(averagingRate, TargetModelParameters, PrimaryModelParameters)

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

function SoftActorCriticModel:setPrimaryCriticModelParameters1(PrimaryCriticModelParameters1, doNotDeepCopy)

	if (doNotDeepCopy) then

		self.PrimaryCriticModelParametersArray[1] = PrimaryCriticModelParameters1

	else

		self.PrimaryCriticModelParametersArray[1] = self:deepCopyTable(PrimaryCriticModelParameters1)

	end

end

function SoftActorCriticModel:setPrimaryCriticModelParameters2(PrimaryCriticModelParameters2, doNotDeepCopy)

	if (doNotDeepCopy) then

		self.PrimaryCriticModelParametersArray[2] = PrimaryCriticModelParameters2

	else

		self.PrimaryCriticModelParametersArray[2] = self:deepCopyTable(PrimaryCriticModelParameters2)

	end

end

function SoftActorCriticModel:getPrimaryCriticModelParameters1(doNotDeepCopy)

	if (doNotDeepCopy) then

		return self.PrimaryCriticModelParametersArray[1]

	else

		return self:deepCopyTable(self.PrimaryCriticModelParametersArray[1])

	end

end

function SoftActorCriticModel:getPrimaryCriticModelParameters2(doNotDeepCopy)

	if (doNotDeepCopy) then

		return self.PrimaryCriticModelParametersArray[2]

	else

		return self:deepCopyTable(self.PrimaryCriticModelParametersArray[2])

	end

end

function SoftActorCriticModel:getTargetCriticModelParameters1(doNotDeepCopy)

	if (doNotDeepCopy) then

		return self.TargetCriticModelParametersArray[1]

	else

		return self:deepCopyTable(self.TargetCriticModelParametersArray[1])

	end

end

function SoftActorCriticModel:getTargetCriticModelParameters2(doNotDeepCopy)

	if (doNotDeepCopy) then

		return self.TargetCriticModelParametersArray[2]

	else

		return self:deepCopyTable(self.TargetCriticModelParametersArray[2])

	end

end

return SoftActorCriticModel
