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

TabularMonteCarloControlModel = {}

TabularMonteCarloControlModel.__index = TabularMonteCarloControlModel

setmetatable(TabularMonteCarloControlModel, TabularReinforcementLearningBaseModel)

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
	
	local stateHistory = {}
	
	local rewardValueHistory = {}
	
	NewTabularMonteCarloControlModel:setCategoricalUpdateFunction(function(previousState, action, rewardValue, currentState, terminalStateValue)
		
		table.insert(stateHistory, previousState)
		
		table.insert(rewardValueHistory, rewardValue)

	end)
	
	NewTabularMonteCarloControlModel:setEpisodeUpdateFunction(function(terminalStateValue)
		
		local Model = NewTabularMonteCarloControlModel.Model
		
		local rewardToGoArray = calculateRewardToGo(rewardValueHistory, NewTabularMonteCarloControlModel.discountFactor)
		
		for h, featureVector in ipairs(stateHistory) do
			
			local averageRewardToGo = rewardToGoArray[h] / h
			
			Model:forwardPropagate(featureVector, true)

			Model:update(averageRewardToGo, true)
			
		end
		
		table.clear(stateHistory)
		
		table.clear(rewardValueHistory)
		
	end)
	
	NewTabularMonteCarloControlModel:setResetFunction(function()
		
		table.clear(stateHistory)
		
		table.clear(rewardValueHistory)
		
	end)
	
	return NewTabularMonteCarloControlModel

end

return TabularMonteCarloControlModel
