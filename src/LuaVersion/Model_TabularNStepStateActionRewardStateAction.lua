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

local TabularNStepStateActionRewardStateActionModel = {}

TabularNStepStateActionRewardStateActionModel.__index = TabularNStepStateActionRewardStateActionModel

setmetatable(TabularNStepStateActionRewardStateActionModel, TabularReinforcementLearningBaseModel)

local defaultNStep = 3

function TabularNStepStateActionRewardStateActionModel.new(parameterDictionary)
	
	parameterDictionary = parameterDictionary or {}

	local NewTabularNStepStateActionRewardStateActionModel = TabularReinforcementLearningBaseModel.new(parameterDictionary)
	
	setmetatable(NewTabularNStepStateActionRewardStateActionModel, TabularNStepStateActionRewardStateActionModel)
	
	NewTabularNStepStateActionRewardStateActionModel:setName("TabularNStepStateActionRewardStateAction")
	
	NewTabularNStepStateActionRewardStateActionModel.nStep = parameterDictionary.nStep or defaultNStep
	
	NewTabularNStepStateActionRewardStateActionModel.replayBufferArray = parameterDictionary.replayBufferArray or {}
	
	NewTabularNStepStateActionRewardStateActionModel:setCategoricalUpdateFunction(function(previousStateValue, previousAction, rewardValue, currentStateValue, currentAction, terminalStateValue)
		
		local nStep = NewTabularNStepStateActionRewardStateActionModel.nStep
		
		local replayBufferArray = NewTabularNStepStateActionRewardStateActionModel.replayBufferArray
		
		table.insert(replayBufferArray, {previousStateValue, previousAction, rewardValue, terminalStateValue})
		
		local currentNStep = #replayBufferArray
		
		if (currentNStep < nStep) and (terminalStateValue == 0) then return 0 end
		
		if (currentNStep > nStep) then 
			
			table.remove(replayBufferArray, 1)
			
			currentNStep = currentNStep - 1
			
		end
		
		local Model = NewTabularNStepStateActionRewardStateActionModel.Model
		
		local discountFactor = NewTabularNStepStateActionRewardStateActionModel.discountFactor
		
		local ActionsList = NewTabularNStepStateActionRewardStateActionModel:getActionsList()

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
		
		local currentQVector = Model:predict(currentStateValue, true)

		local lastQVector = Model:getOutputMatrix(firstExperience[1], true)
		
		local currentActionIndex = table.find(ActionsList, currentAction)
		
		local bootstrapValue = math.pow(discountFactor, currentNStep) * currentQVector[1][currentActionIndex]	

		local nStepTarget = returnValue + bootstrapValue
		
		local previousActionIndex = table.find(ActionsList, previousAction)

		local lastValue = lastQVector[1][previousActionIndex]

		local temporalDifferenceError = nStepTarget - lastValue

		Model:update(-temporalDifferenceError, true)
		
		return temporalDifferenceError

	end)
	
	NewTabularNStepStateActionRewardStateActionModel:setEpisodeUpdateFunction(function(terminalStateValue) 
		
		table.clear(NewTabularNStepStateActionRewardStateActionModel.replayBufferArray)
		
	end)

	NewTabularNStepStateActionRewardStateActionModel:setResetFunction(function()
		
		table.clear(NewTabularNStepStateActionRewardStateActionModel.replayBufferArray)
		
	end)

	return NewTabularNStepStateActionRewardStateActionModel

end

return TabularNStepStateActionRewardStateActionModel
