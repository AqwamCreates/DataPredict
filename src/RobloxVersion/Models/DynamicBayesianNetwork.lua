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

local BaseModel = require(script.Parent.BaseModel)

DynamicBayesianNetworkModel = {}

DynamicBayesianNetworkModel.__index = DynamicBayesianNetworkModel

setmetatable(DynamicBayesianNetworkModel, BaseModel)

local defaultMode = "Hybrid"

local defaultUseLogProbabilities = false

function DynamicBayesianNetworkModel.new(parameterDictionary)

	parameterDictionary = parameterDictionary or {}

	local NewDynamicBayesianNetworkModel = BaseModel.new(parameterDictionary)

	setmetatable(NewDynamicBayesianNetworkModel, DynamicBayesianNetworkModel)

	NewDynamicBayesianNetworkModel:setName("DynamicBayesianNetwork")

	local isHidden = parameterDictionary.isHidden

	local StatesList = parameterDictionary.StatesList or {}

	local ObservationsList = parameterDictionary.ObservationsList or {}

	if (type(isHidden) ~= "boolean") then isHidden = (#ObservationsList > 0) and (ObservationsList ~= StatesList) end
	
	NewDynamicBayesianNetworkModel.mode = parameterDictionary.mode or defaultMode
	
	NewDynamicBayesianNetworkModel.isHidden = isHidden

	NewDynamicBayesianNetworkModel.useLogProbabilities = NewDynamicBayesianNetworkModel:getValueOrDefaultValue(parameterDictionary.useLogProbabilities, defaultUseLogProbabilities)

	NewDynamicBayesianNetworkModel.StatesList = StatesList

	NewDynamicBayesianNetworkModel.ObservationsList = ObservationsList

	NewDynamicBayesianNetworkModel.TransitionProbabilityOptimizer = parameterDictionary.TransitionProbabilityOptimizer

	NewDynamicBayesianNetworkModel.EmissionProbabilityOptimizer = parameterDictionary.EmissionProbabilityOptimizer

	NewDynamicBayesianNetworkModel.ModelParameters = parameterDictionary.ModelParameters

	return NewDynamicBayesianNetworkModel

end

-- State matrix are basically (row) one hot encoding in which the state values are active.

function DynamicBayesianNetworkModel:train(previousStateMatrix, currentStateMatrix, currentObservationStateMatrix)
	
	local StatesList = self.StatesList

	local ObservationsList = self.ObservationsList
	
	local numberOfStates = #StatesList

	local numberOfObservations = #ObservationsList
	
	local numberOfData = #previousStateMatrix

	if (numberOfData ~= #currentStateMatrix) then error("The number of data in the previous state vector is not equal to the number of data in the current state vector.") end
	
	local numberOfPreviousStateColumns = #previousStateMatrix[1]

	local numberOfCurrentStateColumns = #currentStateMatrix[1]
	
	if (numberOfPreviousStateColumns ~= numberOfStates) then error("The number of previous state columns is not equal to the number of states.") end
	
	if (numberOfCurrentStateColumns ~= numberOfStates) then error("The number of current state columns is not equal to the number of states.") end
	
	if (currentObservationStateMatrix) then

		if (numberOfData ~= #currentObservationStateMatrix) then error("The number of data in the previous state vector is not equal to the number of data in the current observation state vector.") end
		
		local numberOfCurrentObservationStateColumns = #currentObservationStateMatrix[1]
		
		if (numberOfCurrentObservationStateColumns ~= numberOfObservations) then error("The number of current observation state columns is not equal to the number of observations.") end
		
	end
	
	local mode = self.mode
	
	local isHidden = self.isHidden
	
	local useLogProbabilities = self.useLogProbabilities



	local ModelParameters = self.ModelParameters or {}

	local transitionCountMatrix = ModelParameters[3]
	
	local emissionCountMatrix = ModelParameters[4]

	if (mode == "Hybrid") then
		
		local emissionCountMatrixBoolean = (isHidden and emissionCountMatrix) or true
		
		mode = (transitionCountMatrix and emissionCountMatrixBoolean and "Online") or "Offline"		

	end
	
	if (mode == "Offline") then

		transitionCountMatrix = AqwamTensorLibrary:createTensor({numberOfStates, numberOfStates})
		
		if (isHidden) then emissionCountMatrix = AqwamTensorLibrary:createTensor({numberOfStates, numberOfObservations}) end

	end
	
	local unwrappedCurrentStateVector
	
	local unwrappedCurrentObservationStateVector
	
	local unwrappedTransitionCountVector
	
	local unwrappedEmissionCountVector

	for dataIndex, unwrappedPreviousStateVector in ipairs(previousStateMatrix) do
		
		unwrappedCurrentStateVector = currentStateMatrix[dataIndex]
		
		for previousStateIndex, previousStateValue in ipairs(unwrappedPreviousStateVector) do
			
			unwrappedTransitionCountVector = transitionCountMatrix[previousStateIndex]
			
			for currentStateIndex, currentStateValue in ipairs(unwrappedCurrentStateVector) do

				unwrappedTransitionCountVector[currentStateIndex] = unwrappedTransitionCountVector[currentStateIndex] + currentStateValue

			end
			
			if (isHidden) then

				unwrappedCurrentObservationStateVector = currentObservationStateMatrix[dataIndex]

				unwrappedEmissionCountVector = emissionCountMatrix[previousStateIndex]

				for observationStateIndex, observationStateValue in ipairs(unwrappedCurrentObservationStateVector) do

					unwrappedEmissionCountVector[observationStateIndex] = unwrappedEmissionCountVector[observationStateIndex] + observationStateValue

				end

			end
			
		end
		
	end
	
	local sumTransitionCountVector = AqwamTensorLibrary:sum(transitionCountMatrix, 2)
	
	local transitionProbabilityMatrix = AqwamTensorLibrary:divide(transitionCountMatrix, sumTransitionCountVector)
	
	local emissionProbabilityMatrix 
	
	if (isHidden) then
		
		local sumEmissionCountVector = AqwamTensorLibrary:sum(emissionCountMatrix, 2)

		emissionProbabilityMatrix = AqwamTensorLibrary:divide(emissionCountMatrix, sumEmissionCountVector)
		
	end

	if (useLogProbabilities) then

		transitionProbabilityMatrix = AqwamTensorLibrary:applyFunction(math.log, transitionProbabilityMatrix)

		if (isHidden) then emissionProbabilityMatrix = AqwamTensorLibrary:applyFunction(math.log, emissionProbabilityMatrix) end

	end

	self.ModelParameters = {transitionProbabilityMatrix, emissionProbabilityMatrix, transitionCountMatrix, emissionCountMatrix}
	
end

function DynamicBayesianNetworkModel:predict(stateMatrix, returnOriginalOutput)

	local isHidden = self.isHidden
	
	local useLogProbabilities = self.useLogProbabilities

	local StatesList = self.StatesList

	local ObservationsList = self.ObservationsList

	local ModelParameters = self.ModelParameters
	
	local numberOfData = #stateMatrix
	
	local numberOfStates = #StatesList
	
	local numberOfObservations = #ObservationsList
	
	local zeroValue = (useLogProbabilities and -math.huge) or 0
	
	local transitionProbabilityMatrix

	local emissionProbabilityMatrix

	if (not ModelParameters) then
		
		local transitionCountMatrix
		
		local emissionCountMatrix
		
		local transitionMatrixDimensionSizeArray = {numberOfStates, numberOfStates}

		transitionProbabilityMatrix = AqwamTensorLibrary:createTensor(transitionMatrixDimensionSizeArray, zeroValue)
		
		transitionCountMatrix = AqwamTensorLibrary:createTensor(transitionMatrixDimensionSizeArray, 0)

		if (isHidden) then
			
			local emissionMatrixDimensionSizeArray = {numberOfStates, numberOfObservations}

			emissionProbabilityMatrix = AqwamTensorLibrary:createTensor(emissionMatrixDimensionSizeArray, zeroValue)
			
			emissionCountMatrix = AqwamTensorLibrary:createTensor(emissionMatrixDimensionSizeArray, 0)

		end

		self.ModelParameters = {transitionProbabilityMatrix, emissionProbabilityMatrix, transitionCountMatrix, emissionCountMatrix}

	else

		transitionProbabilityMatrix = ModelParameters[1]

		emissionProbabilityMatrix = ModelParameters[2]

	end

	local selectedMatrix = (isHidden and emissionProbabilityMatrix) or transitionProbabilityMatrix
	
	local numberOfColumns = (isHidden and numberOfObservations) or numberOfStates
	
	local oneValue = (useLogProbabilities and 0) or 1

	local resultTensor = AqwamTensorLibrary:createTensor({numberOfData, numberOfColumns}, oneValue)
	
	local unwrappedResultVector
	
	local unwrappedProbabilityVector

	for dataIndex, unwrappedStateVector in ipairs(stateMatrix) do
		
		unwrappedResultVector = resultTensor[dataIndex]
		
		for stateIndex, stateValue in ipairs(unwrappedStateVector) do
			
			if (stateValue ~= 0) then -- To save computational resources, we skip the calculations for zero values.
				
				unwrappedProbabilityVector = selectedMatrix[stateIndex]
				
				for probabilityIndex, probability in ipairs(unwrappedProbabilityVector) do
					
					if (useLogProbabilities) then
						
						unwrappedResultVector[probabilityIndex] = unwrappedResultVector[probabilityIndex] + probability
						
					else
						
						unwrappedResultVector[probabilityIndex] = unwrappedResultVector[probabilityIndex] * probability
						
					end
					
				end
				
			end
				
		end

		resultTensor[dataIndex] = unwrappedResultVector

	end

	if (returnOriginalOutput) then return resultTensor end

	local outputVector = {}

	local maximumValueVector = {}

	local SelectedList = (isHidden and ObservationsList) or StatesList

	for dataIndex, unwrappedResultVector in ipairs(resultTensor) do

		local maximumValue = math.max(table.unpack(unwrappedResultVector))

		local outputStateIndex = table.find(unwrappedResultVector, maximumValue)

		local outputState = SelectedList[outputStateIndex] 

		if (not outputState) then error("Output state for index " .. outputStateIndex .. " does not exist in the list.") end

		outputVector[dataIndex] = {outputState}

		maximumValueVector[dataIndex] = {maximumValue}

	end

	return outputVector, maximumValueVector

end

return DynamicBayesianNetworkModel
