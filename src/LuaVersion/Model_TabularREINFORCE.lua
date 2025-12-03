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

local TabularReinforcementLearningBaseModel = require("Model_TabularReinforcementLearningBaseModel")

local TabularREINFORCEModel = {}

TabularREINFORCEModel.__index = TabularREINFORCEModel

setmetatable(TabularREINFORCEModel, TabularReinforcementLearningBaseModel)

local function calculateProbability(valueVector)

	local maximumValue = AqwamTensorLibrary:findMaximumValue(valueVector)

	local zValueVector = AqwamTensorLibrary:subtract(valueVector, maximumValue)

	local exponentVector = AqwamTensorLibrary:exponent(zValueVector)

	local sumExponentValue = AqwamTensorLibrary:sum(exponentVector)

	local probabilityVector = AqwamTensorLibrary:divide(exponentVector, sumExponentValue)

	return probabilityVector

end

local function calculateRewardToGo(rewardValueHistory, discountFactor)

	local rewardToGoArray = {}

	local discountedReward = 0

	for h = #rewardValueHistory, 1, -1 do

		discountedReward = rewardValueHistory[h] + (discountFactor * discountedReward)

		table.insert(rewardToGoArray, 1, discountedReward)

	end

	return rewardToGoArray

end

function TabularREINFORCEModel.new(parameterDictionary)

	local NewTabularREINFORCEModel = TabularReinforcementLearningBaseModel.new(parameterDictionary)

	setmetatable(NewTabularREINFORCEModel, TabularREINFORCEModel)

	NewTabularREINFORCEModel:setName("TabularREINFORCE")

	local stateValueArray = {}

	local actionProbabilityGradientVectorHistory = {}

	local rewardValueHistory = {}

	NewTabularREINFORCEModel:setCategoricalUpdateFunction(function(previousStateValue, previousAction, rewardValue, currentStateValue, currentAction, terminalStateValue)

		local actionVector = NewTabularREINFORCEModel.Model:predict(previousStateValue, true)

		local actionProbabilityVector = calculateProbability(actionVector)
		
		local StatesList = NewTabularREINFORCEModel:getStatesList()

		local ActionsList = NewTabularREINFORCEModel:getActionsList()
		
		local stateIndex = table.find(StatesList, previousStateValue)
		
		local actionIndex = table.find(StatesList, currentAction)

		local actionProbabilityGradientVector = {}

		for i, _ in ipairs(ActionsList) do

			actionProbabilityGradientVector[i] = (((i == actionIndex) and 1) or 0) - actionProbabilityVector[1][i]

		end

		actionProbabilityGradientVector = {actionProbabilityGradientVector}

		table.insert(stateValueArray, previousStateValue)

		table.insert(actionProbabilityGradientVectorHistory, actionProbabilityGradientVector)

		table.insert(rewardValueHistory, rewardValue)

	end)

	NewTabularREINFORCEModel:setEpisodeUpdateFunction(function(terminalStateValue)

		local Model = NewTabularREINFORCEModel.Model
		
		local learningRate = NewTabularREINFORCEModel.learningRate

		local rewardToGoArray = calculateRewardToGo(rewardValueHistory, NewTabularREINFORCEModel.discountFactor)

		for h, actionProbabilityGradientVector in ipairs(actionProbabilityGradientVectorHistory) do

			local lossVector = AqwamTensorLibrary:multiply(actionProbabilityGradientVector, rewardToGoArray[h])
			
			Model:getOutputMatrix(stateValueArray[h], true)

			Model:update(lossVector, true)

		end

		table.clear(stateValueArray)

		table.clear(actionProbabilityGradientVectorHistory)

		table.clear(rewardValueHistory)

	end)

	NewTabularREINFORCEModel:setResetFunction(function()

		table.clear(stateValueArray)

		table.clear(actionProbabilityGradientVectorHistory)

		table.clear(rewardValueHistory)

	end)

	return NewTabularREINFORCEModel

end

return TabularREINFORCEModel
