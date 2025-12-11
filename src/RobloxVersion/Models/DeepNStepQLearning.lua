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

local DeepNStepQLearningModel = {}

DeepNStepQLearningModel.__index = DeepNStepQLearningModel

setmetatable(DeepNStepQLearningModel, DeepReinforcementLearningBaseModel)

local defaultNStep = 3

function DeepNStepQLearningModel.new(parameterDictionary)
	
	parameterDictionary = parameterDictionary or {}

	local NewDeepNStepQLearningModel = DeepReinforcementLearningBaseModel.new(parameterDictionary)
	
	setmetatable(NewDeepNStepQLearningModel, DeepNStepQLearningModel)
	
	NewDeepNStepQLearningModel:setName("DeepNStepQLearning")
	
	NewDeepNStepQLearningModel.nStep = parameterDictionary.nStep or defaultNStep
	
	NewDeepNStepQLearningModel.replayBufferArray = parameterDictionary.replayBufferArray or {}
	
	NewDeepNStepQLearningModel:setCategoricalUpdateFunction(function(previousFeatureVector, previousAction, rewardValue, currentFeatureVector, currentAction, terminalStateValue)
		
		local nStep = NewDeepNStepQLearningModel.nStep

		local replayBufferArray = NewDeepNStepQLearningModel.replayBufferArray

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

		local Model = NewDeepNStepQLearningModel.Model

		local discountFactor = NewDeepNStepQLearningModel.discountFactor

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

		local _, maxQValue = Model:predict(currentFeatureVector)

		local lastQVector = Model:forwardPropagate(firstExperience[1], true)

		local bootstrapValue = math.pow(discountFactor, currentNStep) * maxQValue[1][1]	

		local nStepTarget = returnValue + bootstrapValue

		local actionIndex = table.find(ClassesList, firstExperience[2])

		local lastValue = lastQVector[1][actionIndex]

		local temporalDifferenceError = nStepTarget - lastValue
		
		local outputDimensionSizeArray = {1, #ClassesList}

		local temporalDifferenceErrorVector = AqwamTensorLibrary:createTensor(outputDimensionSizeArray, 0)

		temporalDifferenceErrorVector[1][actionIndex] = temporalDifferenceError
		
		local negatedTemporalDifferenceErrorVector = AqwamTensorLibrary:unaryMinus(temporalDifferenceErrorVector) -- The original non-deep Q-Learning version performs gradient ascent. But the neural network performs gradient descent. So, we need to negate the error vector to make the neural network to perform gradient ascent.

		Model:update(negatedTemporalDifferenceErrorVector, true)
		
		return temporalDifferenceErrorVector

	end)
	
	NewDeepNStepQLearningModel:setEpisodeUpdateFunction(function(terminalStateValue)
		
		table.clear(NewDeepNStepQLearningModel.replayBufferArray)
		
	end)

	NewDeepNStepQLearningModel:setResetFunction(function()
		
		table.clear(NewDeepNStepQLearningModel.replayBufferArray)
		
	end)

	return NewDeepNStepQLearningModel

end

return DeepNStepQLearningModel
