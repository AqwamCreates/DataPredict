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

local DeepNStepStateActionRewardStateActionModel = {}

DeepNStepStateActionRewardStateActionModel.__index = DeepNStepStateActionRewardStateActionModel

setmetatable(DeepNStepStateActionRewardStateActionModel, DeepReinforcementLearningBaseModel)

local defaultNStep = 3

function DeepNStepStateActionRewardStateActionModel.new(parameterDictionary)
	
	parameterDictionary = parameterDictionary or {}

	local NewDeepNStepStateActionRewardStateActionModel = DeepReinforcementLearningBaseModel.new(parameterDictionary)
	
	setmetatable(NewDeepNStepStateActionRewardStateActionModel, DeepNStepStateActionRewardStateActionModel)
	
	NewDeepNStepStateActionRewardStateActionModel:setName("DeepNStepStateActionRewardStateAction")
	
	NewDeepNStepStateActionRewardStateActionModel.nStep = parameterDictionary.nStep or defaultNStep
	
	NewDeepNStepStateActionRewardStateActionModel.replayBufferArray = parameterDictionary.replayBufferArray or {}
	
	NewDeepNStepStateActionRewardStateActionModel:setCategoricalUpdateFunction(function(previousFeatureVector, previousAction, rewardValue, currentFeatureVector, currentAction, terminalStateValue)
		
		local nStep = NewDeepNStepStateActionRewardStateActionModel.nStep

		local replayBufferArray = NewDeepNStepStateActionRewardStateActionModel.replayBufferArray

		table.insert(replayBufferArray, {previousFeatureVector, previousAction, rewardValue, terminalStateValue})

		local currentNStep = #replayBufferArray

		if (currentNStep < nStep) and (terminalStateValue == 0) then return 0 end

		if (currentNStep > nStep) then 

			table.remove(replayBufferArray, 1)

			currentNStep = currentNStep - 1

		end
		
		if (currentNStep < nStep) and (terminalStateValue == 0) then return 0 end

		if (currentNStep > nStep) then 

			table.remove(replayBufferArray, 1)

			currentNStep = currentNStep - 1

		end

		local Model = NewDeepNStepStateActionRewardStateActionModel.Model

		local discountFactor = NewDeepNStepStateActionRewardStateActionModel.discountFactor

		local ClassesList = Model:getClassesList()

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

		local currentQVector = Model:predict(currentFeatureVector, true)

		local lastQVector = Model:forwardPropagate(firstExperience[1], true)
		
		local currentActionIndex = table.find(ClassesList, currentAction)
		
		local previousActionIndex = table.find(ClassesList, firstExperience[2])

		local bootstrapValue = math.pow(discountFactor, currentNStep) * currentQVector[1][currentActionIndex]	

		local nStepTarget = returnValue + bootstrapValue

		local lastValue = lastQVector[1][previousActionIndex]

		local temporalDifferenceError = nStepTarget - lastValue
		
		local outputDimensionSizeArray = {1, #ClassesList}

		local negatedTemporalDifferenceErrorVector = AqwamTensorLibrary:createTensor(outputDimensionSizeArray, 0)

		negatedTemporalDifferenceErrorVector[1][previousActionIndex] = -temporalDifferenceError -- The original non-deep Q-Learning version performs gradient ascent. But the neural network performs gradient descent. So, we need to negate the error vector to make the neural network to perform gradient ascent.

		Model:update(negatedTemporalDifferenceErrorVector, true)
		
		return temporalDifferenceError

	end)
	
	NewDeepNStepStateActionRewardStateActionModel:setEpisodeUpdateFunction(function(terminalStateValue)
		
		table.clear(NewDeepNStepStateActionRewardStateActionModel.replayBufferArray)
		
	end)

	NewDeepNStepStateActionRewardStateActionModel:setResetFunction(function()
		
		table.clear(NewDeepNStepStateActionRewardStateActionModel.replayBufferArray)
		
	end)

	return NewDeepNStepStateActionRewardStateActionModel

end

return DeepNStepStateActionRewardStateActionModel
