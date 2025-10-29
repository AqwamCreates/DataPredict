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

local BaseModel = require(script.Parent.BaseModel)

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
	
	NewMarkovModel.ModelParameters = parameterDictionary.ModelParameters
	
	return NewMarkovModel
	
end

function MarkovModel:setLearningRate(learningRate)

	self.learningRate = learningRate

end

function MarkovModel:getLearningRate()

	return self.learningRate

end

function MarkovModel:train(previousStateVector, currentStateVector, observationStateVector)
	
	local learningRate = self.learningRate
	
	local isHidden = self.isHidden
	
	local StatesList = self.StatesList
	
	local ObservationsList = self.ObservationsList
	
	local ModelParameters = self.ModelParameters or {}
	
	local numberOfStates = #StatesList
	
	local numberOfObservations = #ObservationsList
	
	local transitionProbabilityMatrix = ModelParameters[1] or self:initializeMatrixBasedOnMode({numberOfStates, numberOfStates})
	
	local emissionProbabilityMatrix
	
	if (isHidden) then
		
		emissionProbabilityMatrix = ModelParameters[2] or self:initializeMatrixBasedOnMode({numberOfStates, numberOfObservations})
		
	end
	
	local learningRateComplement = 1 - learningRate
	
	local previousState
	
	local currentState
	
	local observationState
	
	local previousStateIndex
	
	local currentStateIndex
	
	local observationStateIndex
	
	local unwrappedPreviousStateTransitionProbabilityVector
	
	local targetTransitionProbabilityValue
	
	local newTransitionProbabilityValue
	
	local sumProbability
	
	local unwrappedCurrentStateEmissionVector
	
	local currentStateEmissionValue
	
	local targetStateEmissionProbabilityValue
	
	local newStateEmissionProbabilityValue
	
	for i, unwrappedPreviousStateVector in ipairs(previousStateVector) do
		
		previousState = unwrappedPreviousStateVector[1]
		
		currentState = currentStateVector[i][1]
		
		previousStateIndex = table.find(StatesList, previousState)
		
		currentStateIndex = table.find(StatesList, currentState)

		if (previousStateIndex) and (currentStateIndex) then
			
			unwrappedPreviousStateTransitionProbabilityVector = transitionProbabilityMatrix[previousStateIndex]
			
			sumProbability = 0
			
			for j, previousStateTransitionProbabilityValue in ipairs(unwrappedPreviousStateTransitionProbabilityVector) do
				
				targetTransitionProbabilityValue = ((j == currentStateIndex) and 1) or 0
				
				newTransitionProbabilityValue = previousStateTransitionProbabilityValue + learningRate * (targetTransitionProbabilityValue - previousStateTransitionProbabilityValue)
				
				unwrappedPreviousStateTransitionProbabilityVector[j] = newTransitionProbabilityValue
				
				sumProbability = sumProbability + newTransitionProbabilityValue
				
			end
			
			for j, probability in ipairs(unwrappedPreviousStateTransitionProbabilityVector) do unwrappedPreviousStateTransitionProbabilityVector[j] = probability / sumProbability end
			
		end
		
		if (isHidden) then
			
			observationState = observationStateVector[i][1]

			if (observationState) then

				observationStateIndex = table.find(ObservationsList, observationState)

				if (currentStateIndex) and (observationStateIndex) then

					unwrappedCurrentStateEmissionVector = emissionProbabilityMatrix[currentStateIndex]
					
					sumProbability = 0

					for j, currentStateEmissionProbabilityValue in ipairs(unwrappedCurrentStateEmissionVector) do

						targetStateEmissionProbabilityValue = ((j == observationStateIndex) and 1) or 0

						newStateEmissionProbabilityValue = currentStateEmissionProbabilityValue + learningRate * (targetStateEmissionProbabilityValue - currentStateEmissionProbabilityValue)

						unwrappedCurrentStateEmissionVector[j] = newStateEmissionProbabilityValue

						sumProbability = sumProbability + newStateEmissionProbabilityValue

					end

					for j, probability in ipairs(unwrappedCurrentStateEmissionVector) do unwrappedCurrentStateEmissionVector[j] = probability / sumProbability end

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
	
	for i, wrappedState in ipairs(stateVector) do
		
		local state = wrappedState[1]
		
		local stateIndex = table.find(StatesList, state)
		
		if (not stateIndex) then error("State \"" .. state ..  "\" does not exist in the states list.") end
		
		resultTensor[i] = selectedMatrix[stateIndex]
		
	end
	
	if (returnOriginalOutput) then return resultTensor end
	
	local outputVector = {}
	
	local maximumValueVector = {}
	
	local SelectedList = (isHidden and ObservationsList) or StatesList
	
	for i, resultVector in ipairs(resultTensor) do
		
		local maximumValue = math.max(table.unpack(resultVector))
		
		local outputStateIndex = table.find(resultVector, maximumValue)
		
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
