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

local TabularNStepQLearningModel = {}

TabularNStepQLearningModel.__index = TabularNStepQLearningModel

setmetatable(TabularNStepQLearningModel, TabularReinforcementLearningBaseModel)

local defaultNStep = 3

function TabularNStepQLearningModel.new(parameterDictionary)
	
	parameterDictionary = parameterDictionary or {}

	local NewTabularNStepQLearningModel = TabularReinforcementLearningBaseModel.new(parameterDictionary)
	
	setmetatable(NewTabularNStepQLearningModel, TabularNStepQLearningModel)
	
	NewTabularNStepQLearningModel:setName("TabularNStepQLearning")
	
	NewTabularNStepQLearningModel.nStep = parameterDictionary.nStep or defaultNStep
	
	NewTabularNStepQLearningModel.replayBufferArray = parameterDictionary.replayBufferArray or {}
	
	NewTabularNStepQLearningModel:setCategoricalUpdateFunction(function(previousStateValue, previousAction, rewardValue, currentStateValue, currentAction, terminalStateValue)
		
		local nStep = NewTabularNStepQLearningModel.nStep
		
		local replayBufferArray = NewTabularNStepQLearningModel.replayBufferArray
		
		table.insert(replayBufferArray, {previousStateValue, previousAction, rewardValue, terminalStateValue})
		
		local currentNStep = #replayBufferArray
		
		if (currentNStep < nStep) and (terminalStateValue == 0) then return 0 end
		
		if (currentNStep > nStep) then 
			
			table.remove(replayBufferArray, 1)
			
			currentNStep = currentNStep - 1
			
		end
		
		local Model = NewTabularNStepQLearningModel.Model
		
		local discountFactor = NewTabularNStepQLearningModel.discountFactor
		
		local ActionsList = NewTabularNStepQLearningModel:getActionsList()

		local returnValue = 0
		
		local experience

		local rewardValueAtStepI

		local terminalStateValueAtStepI

		for i = currentNStep, 1, -1 do

			experience = replayBufferArray[i]

			rewardValueAtStepI = experience[3]

			terminalStateValueAtStepI = experience[4]

			returnValue = rewardValueAtStepI + (discountFactor * (1 - terminalStateValueAtStepI) * returnValue)

		end
		
		local firstExperience = replayBufferArray[1]

		local _, maxQValue = Model:predict(currentStateValue)
		
		local lastQVector = Model:getOutputMatrix(firstExperience[1], true)
		
		local bootstrapValue = math.pow(discountFactor, currentNStep) * maxQValue[1][1]	

		local nStepTarget = returnValue + bootstrapValue

		local actionIndex = table.find(ActionsList, firstExperience[2])

		local lastValue = lastQVector[1][actionIndex]

		local temporalDifferenceError = nStepTarget - lastValue

		Model:update(-temporalDifferenceError, true)
		
		return temporalDifferenceError

	end)
	
	NewTabularNStepQLearningModel:setEpisodeUpdateFunction(function(terminalStateValue) 
		
		table.clear(NewTabularNStepQLearningModel.replayBufferArray)
		
	end)

	NewTabularNStepQLearningModel:setResetFunction(function()
		
		table.clear(NewTabularNStepQLearningModel.replayBufferArray)
		
	end)

	return NewTabularNStepQLearningModel

end

return TabularNStepQLearningModel
