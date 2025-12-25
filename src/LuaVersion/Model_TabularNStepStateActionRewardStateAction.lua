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

local TabularNStepExpectedStateActionRewardStateActionModel = {}

TabularNStepExpectedStateActionRewardStateActionModel.__index = TabularNStepExpectedStateActionRewardStateActionModel

setmetatable(TabularNStepExpectedStateActionRewardStateActionModel, TabularReinforcementLearningBaseModel)

local defaultEpsilon = 0.5

local defaultNStep = 3

function TabularNStepExpectedStateActionRewardStateActionModel.new(parameterDictionary)
	
	parameterDictionary = parameterDictionary or {}

	local NewTabularNStepExpectedStateActionRewardStateActionModel = TabularReinforcementLearningBaseModel.new(parameterDictionary)
	
	setmetatable(NewTabularNStepExpectedStateActionRewardStateActionModel, TabularNStepExpectedStateActionRewardStateActionModel)
	
	NewTabularNStepExpectedStateActionRewardStateActionModel:setName("TabularNStepExpectedStateActionRewardStateAction")
	
	NewTabularNStepExpectedStateActionRewardStateActionModel.epsilon = parameterDictionary.epsilon or defaultEpsilon
	
	NewTabularNStepExpectedStateActionRewardStateActionModel.nStep = parameterDictionary.nStep or defaultNStep
	
	NewTabularNStepExpectedStateActionRewardStateActionModel.replayBufferArray = parameterDictionary.replayBufferArray or {}
	
	NewTabularNStepExpectedStateActionRewardStateActionModel:setCategoricalUpdateFunction(function(previousStateValue, previousAction, rewardValue, currentStateValue, currentAction, terminalStateValue)
		
		local nStep = NewTabularNStepExpectedStateActionRewardStateActionModel.nStep
		
		local replayBufferArray = NewTabularNStepExpectedStateActionRewardStateActionModel.replayBufferArray
		
		table.insert(replayBufferArray, {previousStateValue, previousAction, rewardValue, terminalStateValue})
		
		local currentNStep = #replayBufferArray
		
		if (currentNStep < nStep) and (terminalStateValue == 0) then return 0 end
		
		if (currentNStep > nStep) then 
			
			table.remove(replayBufferArray, 1)
			
			currentNStep = currentNStep - 1
			
		end
		
		local Model = NewTabularNStepExpectedStateActionRewardStateActionModel.Model
		
		local discountFactor = NewTabularNStepExpectedStateActionRewardStateActionModel.discountFactor
		
		local epsilon = NewTabularNStepExpectedStateActionRewardStateActionModel.epsilon
		
		local ActionsList = NewTabularNStepExpectedStateActionRewardStateActionModel:getActionsList()
		
		local numberOfActions = #ActionsList

		local expectedQValue = 0

		local numberOfGreedyActions = 0

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
		
		local targetVector = Model:predict(currentStateValue, true)
		
		local lastQVector = Model:getOutputMatrix(firstExperience[1], true)
		
		local maxQValue = AqwamTensorLibrary:findMaximumValue(targetVector)

		local actionIndex = table.find(ActionsList, previousAction)

		local unwrappedTargetVector = targetVector[1]

		for i = 1, numberOfActions, 1 do

			if (unwrappedTargetVector[i] == maxQValue) then

				numberOfGreedyActions = numberOfGreedyActions + 1

			end

		end

		local nonGreedyActionProbability = epsilon / numberOfActions

		local greedyActionProbability = ((1 - epsilon) / numberOfGreedyActions) + nonGreedyActionProbability

		local actionProbability

		for _, qValue in ipairs(unwrappedTargetVector) do

			actionProbability = ((qValue == maxQValue) and greedyActionProbability) or nonGreedyActionProbability

			expectedQValue = expectedQValue + (qValue * actionProbability)

		end
		
		local bootstrapValue = math.pow(discountFactor, currentNStep) * expectedQValue

		local nStepTarget = returnValue + bootstrapValue
		
		local previousActionIndex = table.find(ActionsList, previousAction)

		local lastValue = lastQVector[1][previousActionIndex]

		local temporalDifferenceError = nStepTarget - lastValue

		Model:update(-temporalDifferenceError, true)
		
		return temporalDifferenceError

	end)
	
	NewTabularNStepExpectedStateActionRewardStateActionModel:setEpisodeUpdateFunction(function(terminalStateValue) 
		
		table.clear(NewTabularNStepExpectedStateActionRewardStateActionModel.replayBufferArray)
		
	end)

	NewTabularNStepExpectedStateActionRewardStateActionModel:setResetFunction(function()
		
		table.clear(NewTabularNStepExpectedStateActionRewardStateActionModel.replayBufferArray)
		
	end)

	return NewTabularNStepExpectedStateActionRewardStateActionModel

end

return TabularNStepExpectedStateActionRewardStateActionModel
