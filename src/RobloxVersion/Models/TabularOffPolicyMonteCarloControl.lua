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

local DeepReinforcementLearningBaseModel = require(script.Parent.DeepReinforcementLearningBaseModel)

TabularOffPolicyMonteCarloControlModel = {}

TabularOffPolicyMonteCarloControlModel.__index = TabularOffPolicyMonteCarloControlModel

setmetatable(TabularOffPolicyMonteCarloControlModel, DeepReinforcementLearningBaseModel)

local defaultTargetPolicyFunction = "StableSoftmax"

local targetPolicyFunctionList = {

	["Greedy"] = function (actionVector)

		local targetActionVector = AqwamTensorLibrary:createTensor({1, #actionVector[1]}, 0)

		local highestActionValue = -math.huge

		local indexWithHighestActionValue

		for i, actionValue in ipairs(actionVector[1]) do

			if (actionValue > highestActionValue) then

				highestActionValue = actionValue

				indexWithHighestActionValue = i

			end

		end

		targetActionVector[1][indexWithHighestActionValue] = highestActionValue

		return targetActionVector

	end,

	["Softmax"] = function (actionVector) -- Apparently Lua doesn't really handle very small values such as math.exp(-1000), so I added a more stable computation exp(a) / exp(b) -> exp (a - b).

		local exponentActionVector = AqwamTensorLibrary:applyFunction(math.exp, actionVector)

		local exponentActionSumVector = AqwamTensorLibrary:sum(exponentActionVector, 2)

		local targetActionVector = AqwamTensorLibrary:divide(exponentActionVector, exponentActionSumVector)

		return targetActionVector

	end,

	["StableSoftmax"] = function (actionVector)

		local highestActionValue = AqwamTensorLibrary:findMaximumValue(actionVector)

		local subtractedZVector = AqwamTensorLibrary:subtract(actionVector, highestActionValue)

		local exponentActionVector = AqwamTensorLibrary:applyFunction(math.exp, subtractedZVector)

		local exponentActionSumVector = AqwamTensorLibrary:sum(exponentActionVector, 2)

		local targetActionVector = AqwamTensorLibrary:divide(exponentActionVector, exponentActionSumVector)

		return targetActionVector

	end,

}

function TabularOffPolicyMonteCarloControlModel.new(parameterDictionary)

	local NewTabularOffPolicyMonteCarloControlModel = DeepReinforcementLearningBaseModel.new(parameterDictionary)

	setmetatable(NewTabularOffPolicyMonteCarloControlModel, TabularOffPolicyMonteCarloControlModel)

	NewTabularOffPolicyMonteCarloControlModel:setName("TabularOffPolicyMonteCarloControl")

	NewTabularOffPolicyMonteCarloControlModel.targetPolicyFunction = parameterDictionary.targetPolicyFunction or defaultTargetPolicyFunction

	local stateHistory = {}

	local actionVectorHistory = {}

	local rewardValueHistory = {}

	NewTabularOffPolicyMonteCarloControlModel:setCategoricalUpdateFunction(function(previousState, action, rewardValue, currentState, terminalStateValue)

		local actionVector = NewTabularOffPolicyMonteCarloControlModel:predict({{previousState}}, true)

		table.insert(stateHistory, previousState)

		table.insert(actionVectorHistory, actionVector)

		table.insert(rewardValueHistory, rewardValue)

	end)

	NewTabularOffPolicyMonteCarloControlModel:setEpisodeUpdateFunction(function(terminalStateValue)

		local Model = NewTabularOffPolicyMonteCarloControlModel.Model

		local targetPolicyFunction = targetPolicyFunctionList[NewTabularOffPolicyMonteCarloControlModel.targetPolicyFunction]

		local discountFactor = NewTabularOffPolicyMonteCarloControlModel.discountFactor

		local numberOfActions = #actionVectorHistory[1]

		local outputDimensionSizeArray = {1, numberOfActions}

		local cVector = AqwamTensorLibrary:createTensor(outputDimensionSizeArray, 0) 

		local weightVector = AqwamTensorLibrary:createTensor(outputDimensionSizeArray, 1)

		local discountedReward = 0
		
		local StatesList = NewTabularOffPolicyMonteCarloControlModel:getStatesList()
		
		local ActionsList = NewTabularOffPolicyMonteCarloControlModel:getActionsList()
		
		local ModelParameters = NewTabularOffPolicyMonteCarloControlModel.ModelParameters
		
		for h = #actionVectorHistory, 1, -1 do

			discountedReward = rewardValueHistory[h] + (discountFactor * discountedReward)

			cVector = AqwamTensorLibrary:add(cVector, weightVector)

			local actionVector = actionVectorHistory[h]

			local lossVectorPart1 = AqwamTensorLibrary:divide(weightVector, cVector)

			local lossVectorPart2 = AqwamTensorLibrary:subtract(discountedReward, actionVector)

			local lossVector = AqwamTensorLibrary:multiply(lossVectorPart1, lossVectorPart2)

			local targetActionVector = targetPolicyFunction(actionVector)

			local actionRatioVector = AqwamTensorLibrary:divide(targetActionVector, actionVector)

			weightVector = AqwamTensorLibrary:multiply(weightVector, actionRatioVector)
			
			local stateIndex = table.find(StatesList, stateHistory[h])
			
			ModelParameters[stateIndex] = AqwamTensorLibrary:add({ModelParameters[stateIndex]}, lossVector)[1]

		end

		table.clear(stateHistory)

		table.clear(actionVectorHistory)

		table.clear(rewardValueHistory)

	end)

	NewTabularOffPolicyMonteCarloControlModel:setResetFunction(function()

		table.clear(stateHistory)

		table.clear(actionVectorHistory)

		table.clear(rewardValueHistory)

	end)

	return NewTabularOffPolicyMonteCarloControlModel

end

return TabularOffPolicyMonteCarloControlModel
