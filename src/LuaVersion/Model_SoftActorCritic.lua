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

SoftActorCriticModel = {}

SoftActorCriticModel.__index = SoftActorCriticModel

setmetatable(SoftActorCriticModel, ReinforcementLearningActorCriticBaseModel)

local defaultAlpha = 0.1

local defaultAveragingRate = 0.995

local function calculateProbability(valueVector)
	
	local maximumValue = AqwamTensorLibrary:findMaximumValue(valueVector)
	
	local zValueVector = AqwamTensorLibrary:subtract(valueVector, maximumValue)
	
	local exponentVector = AqwamTensorLibrary:exponent(zValueVector)
	
	local sumExponentValue = AqwamTensorLibrary:sum(exponentVector)
	
	local probabilityVector = AqwamTensorLibrary:divide(exponentVector, sumExponentValue)

	return probabilityVector

end

local function rateAverageModelParameters(averagingRate, TargetModelParameters, PrimaryModelParameters)

	local averagingRateComplement = 1 - averagingRate

	for layer = 1, #TargetModelParameters, 1 do

		local TargetModelParametersPart = AqwamTensorLibrary:multiply(averagingRate, TargetModelParameters[layer])

		local PrimaryModelParametersPart = AqwamTensorLibrary:multiply(averagingRateComplement, PrimaryModelParameters[layer])

		TargetModelParameters[layer] = AqwamTensorLibrary:add(TargetModelParametersPart, PrimaryModelParametersPart)

	end

	return TargetModelParameters

end

local function calculateLogActionProbabilityVector(actionMeanVector, actionStandardDeviationVector, actionNoiseVector)
	
	local actionVectorPart1 = AqwamTensorLibrary:multiply(actionStandardDeviationVector, actionNoiseVector)

	local actionVector = AqwamTensorLibrary:add(actionMeanVector, actionVectorPart1)

	local zScoreVectorPart1 = AqwamTensorLibrary:subtract(actionVector, actionMeanVector)

	local zScoreVector = AqwamTensorLibrary:divide(zScoreVectorPart1, actionStandardDeviationVector)

	local squaredZScoreVector = AqwamTensorLibrary:power(zScoreVector, 2)

	local logActionProbabilityVectorPart1 = AqwamTensorLibrary:logarithm(actionStandardDeviationVector)

	local logActionProbabilityVectorPart2 = AqwamTensorLibrary:multiply(2, logActionProbabilityVectorPart1)

	local logActionProbabilityVectorPart3 = AqwamTensorLibrary:add(squaredZScoreVector, logActionProbabilityVectorPart2)

	local logActionProbabilityVector = AqwamTensorLibrary:add(logActionProbabilityVectorPart3, math.log(2 * math.pi))
	
	return logActionProbabilityVector
	
end

function SoftActorCriticModel.new(parameterDictionary)
	
	parameterDictionary = parameterDictionary or {}
	
	local NewSoftActorCritic = ReinforcementLearningActorCriticBaseModel.new(parameterDictionary)
	
	setmetatable(NewSoftActorCritic, SoftActorCriticModel)
	
	NewSoftActorCritic:setName("SoftActorCritic")
	
	NewSoftActorCritic.alpha = parameterDictionary.alpha or defaultAlpha
	
	NewSoftActorCritic.averagingRate = parameterDictionary.averagingRate or defaultAveragingRate
	
	NewSoftActorCritic.CriticModelParametersArray = parameterDictionary.CriticModelParametersArray or {}
	
	NewSoftActorCritic:setCategoricalUpdateFunction(function(previousFeatureVector, action, rewardValue, currentFeatureVector, terminalStateValue)
		
		local CriticModel = NewSoftActorCritic.CriticModel

		local previousActionVector = NewSoftActorCritic.ActorModel:forwardPropagate(previousFeatureVector, true)
		
		local currentActionVector = NewSoftActorCritic.ActorModel:forwardPropagate(currentFeatureVector, true)

		local previousActionProbabilityVector = calculateProbability(previousActionVector)
		
		local currentActionProbabilityVector = calculateProbability(currentActionVector)

		local previousLogActionProbabilityVector = AqwamTensorLibrary:logarithm(previousActionProbabilityVector)
		
		local currentLogActionProbabilityVector = AqwamTensorLibrary:logarithm(currentActionProbabilityVector)
		
		return NewSoftActorCritic:backwardPropagate(previousFeatureVector, previousLogActionProbabilityVector, currentLogActionProbabilityVector, action, rewardValue, currentFeatureVector, terminalStateValue)
		
	end)
	
	NewSoftActorCritic:setDiagonalGaussianUpdateFunction(function(previousFeatureVector, actionMeanVector, actionStandardDeviationVector, actionNoiseVector, rewardValue, currentFeatureVector, terminalStateValue)
		
		if (not actionNoiseVector) then actionNoiseVector = AqwamTensorLibrary:createRandomNormalTensor({1, #actionMeanVector[1]}) end
		
		local currentActionMeanVector = NewSoftActorCritic.ActorModel:forwardPropagate(currentFeatureVector, true)
		
		local dimensionSizeArray = AqwamTensorLibrary:getDimensionSizeArray(actionNoiseVector)
		
		local currentActionNoiseVector = AqwamTensorLibrary:createRandomUniformTensor(dimensionSizeArray)
		
		local previousLogActionProbabilityVector = calculateLogActionProbabilityVector(actionMeanVector, actionStandardDeviationVector, actionNoiseVector)
		
		local currentLogActionProbabilityVector = calculateLogActionProbabilityVector(currentActionMeanVector, actionStandardDeviationVector, currentActionNoiseVector)
		
		return NewSoftActorCritic:backwardPropagate(previousFeatureVector, previousLogActionProbabilityVector, currentLogActionProbabilityVector, nil, rewardValue, currentFeatureVector, terminalStateValue)
		
	end)
	
	NewSoftActorCritic:setEpisodeUpdateFunction(function(terminalStateValue) end)
	
	NewSoftActorCritic:setResetFunction(function() end)
	
	return NewSoftActorCritic
	
end

function SoftActorCriticModel:backwardPropagate(previousFeatureVector, previousLogActionProbabilityVector, currentLogActionProbabilityVector, action, rewardValue, currentFeatureVector, terminalStateValue)
	
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

		local criticLoss = yValue - previousCriticValue

		temporalDifferenceErrorVector[1][i] = criticLoss
		
		previousCriticValueArray[i] = previousCriticValue

		CriticModel:backwardPropagate(criticLoss, true)
		
		local TargetModelParameters = CriticModel:getModelParameters(true)
		
		CriticModelParametersArray[i] = rateAverageModelParameters(averagingRate, TargetModelParameters, PreviousCriticModelParametersArray[i])

	end
	
	local minimumCurrentCriticValue = math.min(table.unpack(previousCriticValueArray))

	local actorLossVector = AqwamTensorLibrary:multiply(alpha, previousLogActionProbabilityVector)

	actorLossVector = AqwamTensorLibrary:subtract(minimumCurrentCriticValue, actorLossVector)
	
	actorLossVector = AqwamTensorLibrary:unaryMinus(actorLossVector)

	ActorModel:forwardPropagate(previousFeatureVector, true)

	ActorModel:backwardPropagate(actorLossVector, true)
	
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