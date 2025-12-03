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

local TabularReinforcementLearningBaseModel = require(script.Parent.TabularReinforcementLearningBaseModel)

local TabularMonteCarloControlModel = {}

TabularMonteCarloControlModel.__index = TabularMonteCarloControlModel

setmetatable(TabularMonteCarloControlModel, TabularReinforcementLearningBaseModel)

local function safeguardedDivisionAndUnaryFunction(nominator, denominator)
	
	if (denominator == 0) then return 0 end
	
	return -(nominator / denominator)
	
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

function TabularMonteCarloControlModel.new(parameterDictionary)

	local NewTabularMonteCarloControlModel = TabularReinforcementLearningBaseModel.new(parameterDictionary)
	
	setmetatable(NewTabularMonteCarloControlModel, TabularMonteCarloControlModel)
	
	NewTabularMonteCarloControlModel:setName("TabularMonteCarloControl")
	
	local stateValueHistory = {}
	
	local actionHistory = {}
	
	local rewardValueHistory = {}
	
	NewTabularMonteCarloControlModel:setCategoricalUpdateFunction(function(previousStateValue, previousAction, rewardValue, currentStateValue, currentAction, terminalStateValue)
		
		table.insert(stateValueHistory, previousStateValue)
		
		table.insert(actionHistory, previousAction)
		
		table.insert(rewardValueHistory, rewardValue)

	end)
	
	NewTabularMonteCarloControlModel:setEpisodeUpdateFunction(function(terminalStateValue)
		
		local StatesList = NewTabularMonteCarloControlModel:getStatesList()
		
		local ActionsList = NewTabularMonteCarloControlModel:getActionsList()
		
		local numberOfStates = #StatesList
		
		local numberOfActions = #ActionsList
		
		local dimensionSizeArray = {numberOfStates, numberOfActions}
		
		local returnMatrix = AqwamTensorLibrary:createTensor(dimensionSizeArray, 0)
		
		local countMatrix = AqwamTensorLibrary:createTensor(dimensionSizeArray, 0)
		
		local rewardToGoArray = calculateRewardToGo(rewardValueHistory, NewTabularMonteCarloControlModel.discountFactor)
		
		for h, state in ipairs(stateValueHistory) do
			
			local stateIndex = table.find(StatesList, state)
			
			local actionIndex = table.find(ActionsList, actionHistory[h])
			
			returnMatrix[stateIndex][actionIndex] = returnMatrix[stateIndex][actionIndex] + rewardToGoArray[h]
			
			countMatrix[stateIndex][actionIndex] = countMatrix[stateIndex][actionIndex] + 1
			
		end
		
		local lossMatrix = AqwamTensorLibrary:applyFunction(safeguardedDivisionAndUnaryFunction, returnMatrix, countMatrix)
		
		NewTabularMonteCarloControlModel.Model:gradientDescent(lossMatrix)
		
		table.clear(stateValueHistory)
		
		table.clear(actionHistory)
		
		table.clear(rewardValueHistory)
		
	end)
	
	NewTabularMonteCarloControlModel:setResetFunction(function()
		
		table.clear(stateValueHistory)
		
		table.clear(actionHistory)
		
		table.clear(rewardValueHistory)
		
	end)
	
	return NewTabularMonteCarloControlModel

end

return TabularMonteCarloControlModel
