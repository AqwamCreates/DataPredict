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

DynamicBayesianNetworkModel = {}

DynamicBayesianNetworkModel.__index = DynamicBayesianNetworkModel

setmetatable(DynamicBayesianNetworkModel, BaseModel)

local defaultMode = "Hybrid"

local defaultIsHidden = false

local defaultUseLogProbabilities = false

function DynamicBayesianNetworkModel.new(parameterDictionary)

	parameterDictionary = parameterDictionary or {}

	local NewDynamicBayesianNetworkModel = BaseModel.new(parameterDictionary)

	setmetatable(NewDynamicBayesianNetworkModel, DynamicBayesianNetworkModel)

	NewDynamicBayesianNetworkModel:setName("DynamicBayesianNetwork")

	NewDynamicBayesianNetworkModel.mode = parameterDictionary.mode or defaultMode
	
	NewDynamicBayesianNetworkModel.isHidden = NewDynamicBayesianNetworkModel:getValueOrDefaultValue(parameterDictionary.isHidden, defaultIsHidden)

	NewDynamicBayesianNetworkModel.useLogProbabilities = NewDynamicBayesianNetworkModel:getValueOrDefaultValue(parameterDictionary.useLogProbabilities, defaultUseLogProbabilities)

	NewDynamicBayesianNetworkModel.TransitionProbabilityOptimizer = parameterDictionary.TransitionProbabilityOptimizer

	NewDynamicBayesianNetworkModel.EmissionProbabilityOptimizer = parameterDictionary.EmissionProbabilityOptimizer

	NewDynamicBayesianNetworkModel.ModelParameters = parameterDictionary.ModelParameters

	return NewDynamicBayesianNetworkModel

end

-- State matrix are basically (row) one hot encoding in which the state values are active.

function DynamicBayesianNetworkModel:train(previousStateMatrix, currentStateMatrix, currentObservationStateMatrix)
	
	local numberOfData = #previousStateMatrix
	
	if (numberOfData ~= #currentStateMatrix) then error("The number of data in the previous state vector is not equal to the number of data in the current state vector.") end

	if (currentObservationStateMatrix) then

		if (numberOfData ~= #currentObservationStateMatrix) then error("The number of data in the previous state vector is not equal to the number of data in the current observation state vector.") end

	end
	
	local mode = self.mode

	local isHidden = self.isHidden

	local useLogProbabilities = self.useLogProbabilities

	local ModelParameters = self.ModelParameters or {}

	local transitionCountMatrix = ModelParameters[3]

	local emissionCountMatrix = ModelParameters[4]
	
	local numberOfPreviousStateColumns = #previousStateMatrix[1]

	local numberOfCurrentStateColumns = #currentStateMatrix[1]
	
	local numberOfCurrentObservationStateColumns
	
	local numberOfStates

	local numberOfObservations
	
	if (currentObservationStateMatrix) then numberOfCurrentObservationStateColumns = #currentObservationStateMatrix[1] end

	if (mode == "Hybrid") then
		
		local hasTransition = transitionCountMatrix
		
		local hasEmission = (isHidden and emissionCountMatrix) or (not isHidden)

		mode = (hasTransition and hasEmission and "Online") or "Offline"		

	end
	
	if (mode == "Offline") then
		
		numberOfStates = numberOfPreviousStateColumns

		transitionCountMatrix = AqwamTensorLibrary:createTensor({numberOfStates, numberOfStates})
		
		if (isHidden) then 
			
			numberOfObservations = numberOfCurrentObservationStateColumns
			
			emissionCountMatrix = AqwamTensorLibrary:createTensor({numberOfStates, numberOfObservations}) 
			
		end
		
	else
		
		numberOfStates = #transitionCountMatrix
		
		if (isHidden) then numberOfObservations = #emissionCountMatrix[1] end
		
	end
	
	if (numberOfPreviousStateColumns ~= numberOfStates) then error("The number of previous state columns is not equal to the number of states.") end

	if (numberOfCurrentStateColumns ~= numberOfStates) then error("The number of current state columns is not equal to the number of states.") end
	
	if (isHidden) then

		if (numberOfCurrentObservationStateColumns ~= numberOfObservations) then error("The number of current observation state columns is not equal to the number of observations.") end

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

				unwrappedTransitionCountVector[currentStateIndex] = unwrappedTransitionCountVector[currentStateIndex] + (previousStateValue * currentStateValue)
				
				if (isHidden) then

					unwrappedCurrentObservationStateVector = currentObservationStateMatrix[dataIndex]

					unwrappedEmissionCountVector = emissionCountMatrix[currentStateIndex]

					for currentObservationStateIndex, currentObservationStateValue in ipairs(unwrappedCurrentObservationStateVector) do

						unwrappedEmissionCountVector[currentObservationStateIndex] = unwrappedEmissionCountVector[currentObservationStateIndex] + (currentStateValue * currentObservationStateValue)

					end

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

function DynamicBayesianNetworkModel:predict(stateMatrix)

	local isHidden = self.isHidden
	
	local useLogProbabilities = self.useLogProbabilities

	local ModelParameters = self.ModelParameters
	
	local numberOfStates

	local numberOfObservations
	
	local transitionProbabilityMatrix

	local emissionProbabilityMatrix

	if (not ModelParameters) then
		
		local zeroValue = (useLogProbabilities and -math.huge) or 0
		
		local transitionCountMatrix
		
		local emissionCountMatrix
		
		numberOfStates = #stateMatrix[1]

		local transitionMatrixDimensionSizeArray = {numberOfStates, numberOfStates}

		transitionProbabilityMatrix = AqwamTensorLibrary:createTensor(transitionMatrixDimensionSizeArray, zeroValue)
		
		transitionCountMatrix = AqwamTensorLibrary:createTensor(transitionMatrixDimensionSizeArray, 0)

		if (isHidden) then
			
			numberOfObservations = numberOfStates
			
			local emissionMatrixDimensionSizeArray = {numberOfStates, numberOfObservations}

			emissionProbabilityMatrix = AqwamTensorLibrary:createTensor(emissionMatrixDimensionSizeArray, zeroValue)
			
			emissionCountMatrix = AqwamTensorLibrary:createTensor(emissionMatrixDimensionSizeArray, 0)

		end

		self.ModelParameters = {transitionProbabilityMatrix, emissionProbabilityMatrix, transitionCountMatrix, emissionCountMatrix}

	else

		transitionProbabilityMatrix = ModelParameters[1]
		
		numberOfStates = #transitionProbabilityMatrix
		
		if (isHidden) then
			
			emissionProbabilityMatrix = ModelParameters[2]

			numberOfObservations = #emissionProbabilityMatrix[1]
			
		end

	end

	local selectedMatrix = (isHidden and emissionProbabilityMatrix) or transitionProbabilityMatrix
	
	local numberOfColumns = (isHidden and numberOfObservations) or numberOfStates
	
	local oneValue = (useLogProbabilities and 0) or 1
	
	local numberOfData = #stateMatrix

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
	
	return resultTensor

end

return DynamicBayesianNetworkModel
