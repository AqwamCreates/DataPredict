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

local ReinforcementLearningBaseQuickSetup = require("QuickSetup_ReinforcementLearningBaseQuickSetup")

local CategoricalPolicyBaseQuickSetup = {}

CategoricalPolicyBaseQuickSetup.__index = CategoricalPolicyBaseQuickSetup

setmetatable(CategoricalPolicyBaseQuickSetup, ReinforcementLearningBaseQuickSetup)

local defaultActionSelectionFunction = "Maximum"

local defaultEpsilon = 0

local defaultTemperature = 1

local defaultCValue = 1

local RandomObject = Random.new()

local function selectIndexWithHighestValue(valueVector)
	
	local selectedIndex = 1
	
	local highestValue = -math.huge
	
	for index, value in ipairs(valueVector[1]) do

		if (value > highestValue) then

			highestValue = value

			selectedIndex = index

		end

	end
	
	return selectedIndex
	
end

local function calculateStableProbability(valueVector, temperature)

	local maximumValue = AqwamTensorLibrary:findMaximumValue(valueVector)

	local zValueVector = AqwamTensorLibrary:subtract(valueVector, maximumValue)
	
	local temperatureZValueVector = AqwamTensorLibrary:divide(zValueVector, temperature)

	local exponentVector = AqwamTensorLibrary:exponent(temperatureZValueVector)

	local sumExponentValue = AqwamTensorLibrary:sum(exponentVector)

	local probabilityVector = AqwamTensorLibrary:divide(exponentVector, sumExponentValue)

	return probabilityVector

end

local function calculateProbability(valueVector, temperature)

	local temperatureZValueVector = AqwamTensorLibrary:divide(valueVector, temperature)

	local exponentVector = AqwamTensorLibrary:exponent(temperatureZValueVector)

	local sumExponentValue = AqwamTensorLibrary:sum(exponentVector)

	local probabilityVector = AqwamTensorLibrary:divide(exponentVector, sumExponentValue)

	return probabilityVector

end

local function sample(probabilityVector)
	
	local unwrappedProbabilityVector = probabilityVector[1]

	local totalProbability = 0

	for _, probability in ipairs(unwrappedProbabilityVector) do

		totalProbability = totalProbability + probability

	end

	local randomProbability = math.random() * totalProbability

	local cumulativeProbability = 0

	for index, probability in ipairs(unwrappedProbabilityVector) do

		cumulativeProbability = cumulativeProbability + probability

		if (cumulativeProbability >= randomProbability) then return index end

	end

	return #unwrappedProbabilityVector

end

local function calculateUpperConfidenceBound(actionVector, cValue, selectedActionCountVector, currentNumberOfReinforcements)
	
	local naturalLogCurrentNumberOfReinforcements = math.log(currentNumberOfReinforcements)
	
	local upperConfidenceBoundVector1 = AqwamTensorLibrary:divide(naturalLogCurrentNumberOfReinforcements, selectedActionCountVector)
	
	local upperConfidenceBoundVector2 = AqwamTensorLibrary:multiply(cValue, upperConfidenceBoundVector1)
	
	local upperConfidenceBoundVector = AqwamTensorLibrary:add(actionVector, upperConfidenceBoundVector2)
	
	return upperConfidenceBoundVector
	
end

function CategoricalPolicyBaseQuickSetup:selectAction(actionVector, selectedActionCountVector, currentEpsilon, EpsilonValueScheduler, currentNumberOfReinforcements)
	
	local actionSelectionFunction = self.actionSelectionFunction
	
	local randomProbability = RandomObject:NextNumber()
	
	local actionIndex
	
	currentEpsilon = currentEpsilon or self.epsilon
	
	selectedActionCountVector = selectedActionCountVector or {table.create(#actionVector[1], 0)}
	
	if (randomProbability <= currentEpsilon) then
		
		actionIndex = RandomObject:NextInteger(1, #actionVector[1])
	
	elseif (actionSelectionFunction == "Maximum") then
		
		actionIndex = selectIndexWithHighestValue(actionVector)
	
	elseif (actionSelectionFunction == "StableSoftmaxSampling") or (actionSelectionFunction == "StableBoltzmannSampling") then
		
		local stableActionProbabilityVector = calculateStableProbability(actionVector, self.temperature)
		
		actionIndex = sample(stableActionProbabilityVector)
		
	elseif (actionSelectionFunction == "SoftmaxSampling") or (actionSelectionFunction == "BoltzmannSampling") then

		local actionProbabilityVector = calculateProbability(actionVector, self.temperature)

		actionIndex = sample(actionProbabilityVector)
		
	elseif (actionSelectionFunction == "UpperConfidenceBound") then
		
		local actionUpperConfidenceBoundVector = calculateUpperConfidenceBound(actionVector, self.cValue, selectedActionCountVector, currentNumberOfReinforcements)
		
		actionIndex = selectIndexWithHighestValue(actionUpperConfidenceBoundVector)
		
	else
		
		error("Invalid action selection function.")
		
	end
	
	selectedActionCountVector[1][actionIndex] = selectedActionCountVector[1][actionIndex] + 1
	
	if (EpsilonValueScheduler) then currentEpsilon = EpsilonValueScheduler:calculate(currentEpsilon) end
	
	return actionIndex, selectedActionCountVector, currentEpsilon
	
end

function CategoricalPolicyBaseQuickSetup.new(parameterDictionary)
	
	parameterDictionary = parameterDictionary or {}
	
	local NewCategoricalPolicyBaseQuickSetup = ReinforcementLearningBaseQuickSetup.new(parameterDictionary)
	
	setmetatable(NewCategoricalPolicyBaseQuickSetup, CategoricalPolicyBaseQuickSetup)
	
	NewCategoricalPolicyBaseQuickSetup:setName("CategoricalPolicyBaseQuickSetup")
	
	NewCategoricalPolicyBaseQuickSetup:setClassName("CategoricalPolicyQuickSetup")
	
	local epsilon = parameterDictionary.epsilon or defaultEpsilon
	
	NewCategoricalPolicyBaseQuickSetup.actionSelectionFunction = parameterDictionary.actionSelectionFunction or defaultActionSelectionFunction
	
	NewCategoricalPolicyBaseQuickSetup.epsilon = epsilon or defaultEpsilon
	
	NewCategoricalPolicyBaseQuickSetup.temperature = parameterDictionary.temperature or defaultTemperature
	
	NewCategoricalPolicyBaseQuickSetup.cValue = parameterDictionary.cValue or defaultCValue
	
	NewCategoricalPolicyBaseQuickSetup.EpsilonValueScheduler = parameterDictionary.EpsilonValueScheduler
	
	NewCategoricalPolicyBaseQuickSetup.selectedActionCountVector = parameterDictionary.selectedActionCountVector
	
	NewCategoricalPolicyBaseQuickSetup.currentEpsilon = parameterDictionary.currentEpsilon or epsilon
	
	return NewCategoricalPolicyBaseQuickSetup
	
end

return CategoricalPolicyBaseQuickSetup
