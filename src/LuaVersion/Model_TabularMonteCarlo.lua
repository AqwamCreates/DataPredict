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
	
	local stateValueHistory = {}
	
	local actionHistory = {}
	
	local rewardValueHistory = {}
	
	NewTabularMonteCarloControlModel:setCategoricalUpdateFunction(function(previousStateValue, action, rewardValue, currentStateValue, terminalStateValue)
		
		table.insert(stateValueHistory, previousStateValue)
		
		table.insert(actionHistory, action)
		
		table.insert(rewardValueHistory, rewardValue)

	end)
	
	NewTabularMonteCarloControlModel:setEpisodeUpdateFunction(function(terminalStateValue)
		
		local rewardToGoArray = calculateRewardToGo(rewardValueHistory, NewTabularMonteCarloControlModel.discountFactor)
		
		local StatesList = NewTabularMonteCarloControlModel:getStatesList()
		
		local ActionsList = NewTabularMonteCarloControlModel:getActionsList()
		
		local ModelParameters = NewTabularMonteCarloControlModel.ModelParameters
		
		local numberOfStates = #StatesList
		
		local numberOfActions = #ActionsList
		
		local returnsMatrix = AqwamTensorLibrary:createTensor({numberOfStates, numberOfActions}, 0)
		
		local countMatrix = AqwamTensorLibrary:createTensor({numberOfStates, numberOfActions}, 0)
		
		for h, state in ipairs(stateValueHistory) do
			
			local action = actionHistory[h]
			
			local averageRewardToGo = rewardToGoArray[h]
			
			local stateIndex = table.find(StatesList, state)
			
			local actionIndex = table.find(ActionsList, action)
			
			returnsMatrix[stateIndex][actionIndex] = returnsMatrix[stateIndex][actionIndex] + averageRewardToGo
			
			countMatrix[stateIndex][actionIndex] = countMatrix[stateIndex][actionIndex] + 1
			
		end
		
		for stateIndex, _ in ipairs(StatesList) do
			
			for actionIndex, _ in ipairs(ActionsList) do
				
				local count = countMatrix[stateIndex][actionIndex]
				
				if (count ~= 0) then
					
					ModelParameters[stateIndex][actionIndex] = returnsMatrix[stateIndex][actionIndex] / count
					
				end
				
			end
			
		end
		
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
