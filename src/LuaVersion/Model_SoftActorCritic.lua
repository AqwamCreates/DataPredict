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

SoftActorCriticModel = {}

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

	local subtractedZTensor = AqwamTensorLibrary:subtract(valueTensor, highestActionValue)

	local exponentActionTensor = AqwamTensorLibrary:applyFunction(math.exp, subtractedZTensor)

	local exponentActionSumTensor = AqwamTensorLibrary:sum(exponentActionTensor, 2)

	local targetActionTensor = AqwamTensorLibrary:divide(exponentActionTensor, exponentActionSumTensor)

	return targetActionTensor

end

local function calculateDiagonalGaussianProbability(actionMeanTensor, actionStandardDeviationTensor, actionNoiseTensor)

	local actionTensorPart1 = AqwamTensorLibrary:multiply(actionStandardDeviationTensor, actionNoiseTensor)

	local actionTensor = AqwamTensorLibrary:add(actionMeanTensor, actionTensorPart1)

	local zScoreTensorPart1 = AqwamTensorLibrary:subtract(actionTensor, actionMeanTensor)

	local zScoreTensor = AqwamTensorLibrary:divide(zScoreTensorPart1, actionStandardDeviationTensor)

	local squaredZScoreTensor = AqwamTensorLibrary:power(zScoreTensor, 2)

	local logActionProbabilityTensorPart1 = AqwamTensorLibrary:logarithm(actionStandardDeviationTensor)

	local logActionProbabilityTensorPart2 = AqwamTensorLibrary:multiply(2, logActionProbabilityTensorPart1)

	local logActionProbabilityTensorPart3 = AqwamTensorLibrary:add(squaredZScoreTensor, logActionProbabilityTensorPart2)

	local logActionProbabilityTensorPart4 = AqwamTensorLibrary:add(logActionProbabilityTensorPart3, math.log(2 * math.pi))

	local logActionProbabilityTensor = AqwamTensorLibrary:multiply(-0.5, logActionProbabilityTensorPart4)

	return logActionProbabilityTensor

end

function SoftActorCriticModel.new(parameterDictionary)
	
	parameterDictionary = parameterDictionary or {}
	
	local NewSoftActorCritic = DeepReinforcementLearningActorCriticBaseModel.new(parameterDictionary)
	
	setmetatable(NewSoftActorCritic, SoftActorCriticModel)
	
	NewSoftActorCritic:setName("SoftActorCritic")
	
	NewSoftActorCritic.alpha = parameterDictionary.alpha or defaultAlpha
	
	NewSoftActorCritic.averagingRate = parameterDictionary.averagingRate or defaultAveragingRate
	
	NewSoftActorCritic.CriticModelParametersArray = parameterDictionary.CriticModelParametersArray or {}
	
	NewSoftActorCritic:setCategoricalUpdateFunction(function(previousFeatureVector, action, rewardValue, currentFeatureVector, terminalStateValue)
		
		local ActorModel = NewSoftActorCritic.ActorModel
		
		local CriticModel = NewSoftActorCritic.CriticModel

		local previousActionVector = ActorModel:forwardPropagate(previousFeatureVector, true)
		
		local currentActionVector = ActorModel:forwardPropagate(currentFeatureVector, true)

		local previousActionProbabilityVector = calculateCategoricalProbability(previousActionVector)
		
		local currentActionProbabilityVector = calculateCategoricalProbability(currentActionVector)

		local previousLogActionProbabilityVector = AqwamTensorLibrary:logarithm(previousActionProbabilityVector)
		
		local currentLogActionProbabilityVector = AqwamTensorLibrary:logarithm(currentActionProbabilityVector)
		
		return NewSoftActorCritic:update(previousFeatureVector, previousLogActionProbabilityVector, currentLogActionProbabilityVector, action, rewardValue, currentFeatureVector, terminalStateValue)
		
	end)
	
	NewSoftActorCritic:setDiagonalGaussianUpdateFunction(function(previousFeatureVector, actionMeanVector, actionStandardDeviationVector, actionNoiseVector, rewardValue, currentFeatureVector, terminalStateValue)
		
		if (not actionNoiseVector) then actionNoiseVector = AqwamTensorLibrary:createRandomNormalTensor({1, #actionMeanVector[1]}) end
		
		local currentActionMeanVector = NewSoftActorCritic.ActorModel:forwardPropagate(currentFeatureVector, true)
		
		local dimensionSizeArray = AqwamTensorLibrary:getDimensionSizeArray(actionNoiseVector)
		
		local currentActionNoiseVector = AqwamTensorLibrary:createRandomUniformTensor(dimensionSizeArray)
		
		local previousLogActionProbabilityVector = calculateDiagonalGaussianProbability(actionMeanVector, actionStandardDeviationVector, actionNoiseVector)
		
		local currentLogActionProbabilityVector = calculateDiagonalGaussianProbability(currentActionMeanVector, actionStandardDeviationVector, currentActionNoiseVector)
		
		return NewSoftActorCritic:update(previousFeatureVector, previousLogActionProbabilityVector, currentLogActionProbabilityVector, nil, rewardValue, currentFeatureVector, terminalStateValue)
		
	end)
	
	NewSoftActorCritic:setEpisodeUpdateFunction(function(terminalStateValue) end)
	
	NewSoftActorCritic:setResetFunction(function() end)
	
	return NewSoftActorCritic
	
end

function SoftActorCriticModel:update(previousFeatureVector, previousLogActionProbabilityVector, currentLogActionProbabilityVector, action, rewardValue, currentFeatureVector, terminalStateValue)
	
	local CriticModelParametersArray = self.CriticModelParametersArray
	
	local CriticModel = self.CriticModel

	local ActorModel = self.ActorModel

	local alpha = self.alpha
	
	local averagingRate = self.averagingRate
	
	local PreviousCriticModelParametersArray = {}
	
	local previousLogActionProbabilityValue
	
	if (action) then
		
		local ClassesList = ActorModel:getClassesList()
		
		local actionIndex = table.find(ClassesList, action)
		
		previousLogActionProbabilityValue = previousLogActionProbabilityVector[1][actionIndex]
		
	else
		
		previousLogActionProbabilityValue = AqwamTensorLibrary:sum(previousLogActionProbabilityVector)
		
	end

	local currentCriticValueArray = {}

	for i = 1, 2, 1 do 

		CriticModel:setModelParameters(CriticModelParametersArray[i])

		currentCriticValueArray[i] = CriticModel:forwardPropagate(currentFeatureVector)[1][1] 
		
		local CriticModelParameters = CriticModel:getModelParameters(true)
		
		PreviousCriticModelParametersArray[i] = CriticModelParameters

	end

	local minimumCurrentCriticValue = math.min(table.unpack(currentCriticValueArray))
	
	local yValuePart1 = (1 - terminalStateValue) * (minimumCurrentCriticValue - (alpha * previousLogActionProbabilityValue))

	local yValue = rewardValue + (self.discountFactor * yValuePart1)

	local temporalDifferenceErrorVector = AqwamTensorLibrary:createTensor({1, 2}, 0)
	
	local previousCriticValueArray = {}

	for i = 1, 2, 1 do 

		CriticModel:setModelParameters(PreviousCriticModelParametersArray[i], true)

		local previousCriticValue = CriticModel:forwardPropagate(previousFeatureVector, true)[1][1] 

		local criticLoss = previousCriticValue - yValue

		temporalDifferenceErrorVector[1][i] = -criticLoss -- We perform gradient descent here, so the critic loss is negated so that it can be used as temporal difference value.
		
		previousCriticValueArray[i] = previousCriticValue

		CriticModel:update(criticLoss, true)
		
		local TargetModelParameters = CriticModel:getModelParameters(true)
		
		CriticModelParametersArray[i] = rateAverageModelParameters(averagingRate, TargetModelParameters, PreviousCriticModelParametersArray[i])

	end
	
	local minimumCurrentCriticValue = math.min(table.unpack(previousCriticValueArray))

	local actorLossVector = AqwamTensorLibrary:multiply(alpha, previousLogActionProbabilityVector)

	actorLossVector = AqwamTensorLibrary:subtract(minimumCurrentCriticValue, actorLossVector)
	
	actorLossVector = AqwamTensorLibrary:unaryMinus(actorLossVector)

	ActorModel:forwardPropagate(previousFeatureVector, true)

	ActorModel:update(actorLossVector, true)
	
	return temporalDifferenceErrorVector
	
end

function SoftActorCriticModel:setCrtiticModelParameters1(CriticModelParameters1, doNotDeepCopy)

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
