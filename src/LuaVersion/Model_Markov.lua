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

local BaseModel = require("Model_BaseModel")

MarkovModel = {}

MarkovModel.__index = MarkovModel

setmetatable(MarkovModel, BaseModel)

local defaultLearningRate = 0.1

function MarkovModel.new(parameterDictionary)
	
	parameterDictionary = parameterDictionary or {}
	
	local NewMarkovModel = BaseModel.new(parameterDictionary)
	
	setmetatable(NewMarkovModel, MarkovModel)
	
	NewMarkovModel:setName("Markov")

	NewMarkovModel:setClassName("Markov")
	
	local isHidden = parameterDictionary.isHidden
	
	local StatesList = parameterDictionary.StatesList or {}
	
	local ObservationsList = parameterDictionary.ObservationsList or {}
	
	if (type(isHidden) ~= "boolean") then isHidden = (#ObservationsList > 0) and (ObservationsList ~= StatesList) end
	
	NewMarkovModel.learningRate = parameterDictionary.learningRate or defaultLearningRate
	
	NewMarkovModel.isHidden = isHidden
	
	NewMarkovModel.StatesList = StatesList
	
	NewMarkovModel.ObservationsList = ObservationsList
	
	NewMarkovModel.TransitionProbabilityOptimizer = parameterDictionary.TransitionProbabilityOptimizer
	
	NewMarkovModel.EmissionProbabilityOptimizer = parameterDictionary.EmissionProbabilityOptimizer
	
	NewMarkovModel.ModelParameters = parameterDictionary.ModelParameters
	
	return NewMarkovModel
	
end

function MarkovModel:setLearningRate(learningRate)

	self.learningRate = learningRate

end

function MarkovModel:getLearningRate()

	return self.learningRate

end

function MarkovModel:train(previousStateVector, currentStateVector, currentObservationStateVector)
	
	local numberOfData = #previousStateVector
	
	if (numberOfData ~= #currentStateVector) then error("The number of data in the previous state vector is not equal to the number of data in the current state vector.") end
	
	if (currentObservationStateVector) then
		
		if (numberOfData ~= #currentObservationStateVector) then error("The number of data in the previous state vector is not equal to the number of data in the current observation state vector.") end
		
	end
	
	local learningRate = self.learningRate
	
	local isHidden = self.isHidden
	
	local StatesList = self.StatesList
	
	local ObservationsList = self.ObservationsList
	
	local TransitionProbabilityOptimizer = self.TransitionProbabilityOptimizer
	
	local EmissionProbabilityOptimizer = self.EmissionProbabilityOptimizer
	
	local ModelParameters = self.ModelParameters or {}
	
	local numberOfStates = #StatesList
	
	local numberOfObservations = #ObservationsList
	
	local transitionProbabilityMatrix = ModelParameters[1] or self:initializeMatrixBasedOnMode({numberOfStates, numberOfStates})
	
	local emissionProbabilityMatrix
	
	if (isHidden) then
		
		emissionProbabilityMatrix = ModelParameters[2] or self:initializeMatrixBasedOnMode({numberOfStates, numberOfObservations})
		
	end
	
	local previousState
	
	local currentState
	
	local currentObservationState
	
	local previousStateIndex
	
	local currentStateIndex
	
	local observationStateIndex
	
	local unwrappedPreviousStateTransitionProbabilityVector
	
	local targetTransitionProbabilityValue
	
	local transitionProbabilityChangeValue
	
	local transitionProbabilityChangeVector
	
	local newTransitionProbabilityVector
	
	local sumNewTransitionProbability
	
	local unwrappedCurrentStateEmissionProbabilityVector
	
	local targetStateEmissionProbabilityValue
	
	local stateEmissionProbabilityChangeVector
	
	local newStateEmissionProbabilityVector
	
	local sumNewStateEmissionProbability
	
	for i, unwrappedPreviousStateVector in ipairs(previousStateVector) do
		
		previousState = unwrappedPreviousStateVector[1]
		
		currentState = currentStateVector[i][1]
		
		previousStateIndex = table.find(StatesList, previousState)
		
		currentStateIndex = table.find(StatesList, currentState)

		if (previousStateIndex) and (currentStateIndex) then
			
			unwrappedPreviousStateTransitionProbabilityVector = transitionProbabilityMatrix[previousStateIndex]
			
			transitionProbabilityChangeVector = {}
			
			for j, previousStateTransitionProbabilityValue in ipairs(unwrappedPreviousStateTransitionProbabilityVector) do
				
				targetTransitionProbabilityValue = ((j == currentStateIndex) and 1) or 0
				
				transitionProbabilityChangeVector[j] = {targetTransitionProbabilityValue - previousStateTransitionProbabilityValue}
				
			end
			
			transitionProbabilityChangeVector = {transitionProbabilityChangeVector}
			
			if (TransitionProbabilityOptimizer) then

				transitionProbabilityChangeVector = TransitionProbabilityOptimizer:calculate(learningRate, transitionProbabilityChangeVector)

			else

				transitionProbabilityChangeVector = AqwamTensorLibrary:multiply(learningRate, transitionProbabilityChangeVector)

			end
			
			newTransitionProbabilityVector = AqwamTensorLibrary:add({unwrappedPreviousStateTransitionProbabilityVector}, transitionProbabilityChangeVector)
			
			sumNewTransitionProbability = AqwamTensorLibrary:sum(newTransitionProbabilityVector)
			
			if (sumNewTransitionProbability ~= 0) then
				
				newTransitionProbabilityVector = AqwamTensorLibrary:divide(newTransitionProbabilityVector, sumNewTransitionProbability)

				transitionProbabilityMatrix[previousStateIndex] = newTransitionProbabilityVector[1]
				
			end
			
		end
		
		if (isHidden) then
			
			currentObservationState = currentObservationStateVector[i][1]

			if (currentObservationState) then

				observationStateIndex = table.find(ObservationsList, currentObservationState)

				if (currentStateIndex) and (observationStateIndex) then

					unwrappedCurrentStateEmissionProbabilityVector = emissionProbabilityMatrix[currentStateIndex]
					
					stateEmissionProbabilityChangeVector = {}

					for j, currentStateEmissionProbabilityValue in ipairs(unwrappedCurrentStateEmissionProbabilityVector) do

						targetStateEmissionProbabilityValue = ((j == observationStateIndex) and 1) or 0
						
						stateEmissionProbabilityChangeVector[j] = targetStateEmissionProbabilityValue - currentStateEmissionProbabilityValue

					end
					
					stateEmissionProbabilityChangeVector = {stateEmissionProbabilityChangeVector}
					
					if (EmissionProbabilityOptimizer) then

						stateEmissionProbabilityChangeVector = EmissionProbabilityOptimizer:calculate(learningRate, stateEmissionProbabilityChangeVector)

					else

						stateEmissionProbabilityChangeVector = AqwamTensorLibrary:multiply(learningRate, stateEmissionProbabilityChangeVector)

					end
					
					newStateEmissionProbabilityVector = AqwamTensorLibrary:add({unwrappedCurrentStateEmissionProbabilityVector}, stateEmissionProbabilityChangeVector)
					
					sumNewStateEmissionProbability = AqwamTensorLibrary:sum(newStateEmissionProbabilityVector)
					
					if (sumNewStateEmissionProbability ~= 0) then
						
						newStateEmissionProbabilityVector = AqwamTensorLibrary:divide(newStateEmissionProbabilityVector, sumNewStateEmissionProbability)

						emissionProbabilityMatrix[currentStateIndex] = newStateEmissionProbabilityVector[1]
						
					end
					
				end
			end
			
		end
		
	end
	
	self.ModelParameters = {transitionProbabilityMatrix, emissionProbabilityMatrix}
	
end

function MarkovModel:predict(stateVector, returnOriginalOutput)
	
	local isHidden = self.isHidden
	
	local StatesList = self.StatesList
	
	local ObservationsList = self.ObservationsList
	
	local ModelParameters = self.ModelParameters
	
	local numberOfStates = #StatesList

	local numberOfObservations = #ObservationsList
	
	local transitionProbabilityMatrix

	local emissionProbabilityMatrix
	
	if (not ModelParameters) then
		
		transitionProbabilityMatrix = self:initializeMatrixBasedOnMode({numberOfStates, numberOfStates})
		
		if (isHidden) then
			
			emissionProbabilityMatrix = self:initializeMatrixBasedOnMode({numberOfStates, numberOfObservations})
			
		end
		
		self.ModelParameters = {transitionProbabilityMatrix, emissionProbabilityMatrix}
		
	else
		
		transitionProbabilityMatrix = ModelParameters[1]
		
		emissionProbabilityMatrix = ModelParameters[2]
		
	end
	
	local resultTensor = {}
	
	local selectedMatrix = (isHidden and emissionProbabilityMatrix) or transitionProbabilityMatrix
	
	for i, unwrappedStateVector in ipairs(stateVector) do
		
		local state = unwrappedStateVector[1]
		
		local stateIndex = table.find(StatesList, state)
		
		if (not stateIndex) then error("State \"" .. state ..  "\" does not exist in the states list.") end
		
		resultTensor[i] = selectedMatrix[stateIndex]
		
	end
	
	if (returnOriginalOutput) then return resultTensor end
	
	local outputVector = {}
	
	local maximumValueVector = {}
	
	local SelectedList = (isHidden and ObservationsList) or StatesList
	
	for i, unwrappedResultVector in ipairs(resultTensor) do
		
		local maximumValue = math.max(table.unpack(unwrappedResultVector))
		
		local outputStateIndex = table.find(unwrappedResultVector, maximumValue)
		
		local outputState = SelectedList[outputStateIndex] 
		
		if (not outputState) then error("Output state for index " .. outputStateIndex .. " does not exist in the list.") end
		
		outputVector[i] = {outputState}
		
		maximumValueVector[i] = {maximumValue}
		
	end

	return outputVector, maximumValueVector

end

function MarkovModel:setStatesList(StatesList)
	
	self.StatesList = StatesList
	
end

function MarkovModel:getStatesList()
	
	return self.StatesList
	
end

function MarkovModel:setObservationsList(ObservationsList)
	
	self.ObservationsList = ObservationsList
	
end

function MarkovModel:getObservationsList()
	
	return self.ObservationsList
	
end

return MarkovModel
