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
		
		local CriticModel = NewSoftActorCritic.CriticModel

		local previousActionVector = ActorModel:forwardPropagate(previousFeatureVector, true)
		
		local currentActionVector = ActorModel:forwardPropagate(currentFeatureVector, true)

		local previousActionProbabilityVector = calculateCategoricalProbability(previousActionVector)
		
		local currentActionProbabilityVector = calculateCategoricalProbability(currentActionVector)

		local previousLogActionProbabilityVector = AqwamTensorLibrary:logarithm(previousActionProbabilityVector)
		
		local currentLogActionProbabilityVector = AqwamTensorLibrary:logarithm(currentActionProbabilityVector)
		
		return NewSoftActorCritic:update(previousFeatureVector, previousLogActionProbabilityVector, currentLogActionProbabilityVector, previousAction, rewardValue, currentFeatureVector, terminalStateValue)
		
	end)
	
	NewSoftActorCritic:setDiagonalGaussianUpdateFunction(function(previousFeatureVector, previousActionMeanVector, previousActionStandardDeviationVector, previousActionNoiseVector, rewardValue, currentFeatureVector, currentActionMeanVector, terminalStateValue)
		
		if (not previousActionNoiseVector) then previousActionNoiseVector = AqwamTensorLibrary:createRandomNormalTensor({1, #previousActionMeanVector[1]}) end
		
		local dimensionSizeArray = AqwamTensorLibrary:getDimensionSizeArray(previousActionNoiseVector)
		
		local currentActionNoiseVector = AqwamTensorLibrary:createRandomUniformTensor(dimensionSizeArray)
		
		local previousLogActionProbabilityVector = calculateDiagonalGaussianProbability(previousActionMeanVector, previousActionStandardDeviationVector, previousActionStandardDeviationVector)
		
		local currentLogActionProbabilityVector = calculateDiagonalGaussianProbability(currentActionMeanVector, previousActionStandardDeviationVector, currentActionNoiseVector)
		
		return NewSoftActorCritic:update(previousFeatureVector, previousLogActionProbabilityVector, currentLogActionProbabilityVector, nil, rewardValue, currentFeatureVector, terminalStateValue)
		
	end)
	
	NewSoftActorCritic:setEpisodeUpdateFunction(function(terminalStateValue) end)
	
	NewSoftActorCritic:setResetFunction(function() end)
	
	return NewSoftActorCritic
	
end

function SoftActorCriticModel:update(previousFeatureVector, previousLogActionProbabilityVector, currentLogActionProbabilityVector, previousAction, rewardValue, currentFeatureVector, terminalStateValue)
	
	local CriticModelParametersArray = self.CriticModelParametersArray
	
	local CriticModel = self.CriticModel

	local ActorModel = self.ActorModel

	local alpha = self.alpha
	
	local averagingRate = self.averagingRate
	
	local PreviousCriticModelParametersArray = {}
	
	local previousLogActionProbabilityValue
	
	if (previousAction) then
		
		local ClassesList = ActorModel:getClassesList()
		
		local actionIndex = table.find(ClassesList, previousAction)
		
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
