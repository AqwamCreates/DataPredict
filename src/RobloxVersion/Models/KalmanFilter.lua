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

KalmanFilterModel = {}

KalmanFilterModel.__index = KalmanFilterModel

setmetatable(KalmanFilterModel, BaseModel)

function KalmanFilterModel.new(parameterDictionary)
	
	parameterDictionary = parameterDictionary or {}

	local NewKalmanFilterModel = BaseModel.new(parameterDictionary)

	setmetatable(NewKalmanFilterModel, KalmanFilterModel)

	NewKalmanFilterModel:setName("KalmanFilter")
	
	NewKalmanFilterModel.stateTransitionMatrix = parameterDictionary.stateTransitionMatrix
	
	NewKalmanFilterModel.observationMatrix = parameterDictionary.observationMatrix
	
	NewKalmanFilterModel.processNoiseCovarianceMatrix = parameterDictionary.processNoiseCovarianceMatrix
	
	NewKalmanFilterModel.observationNoiseCovarianceMatrix = parameterDictionary.observationNoiseCovarianceMatrix
	
	NewKalmanFilterModel.controlInputMatrix = parameterDictionary.controlInputMatrix
	
	NewKalmanFilterModel.controlVector = parameterDictionary.controlVector

	return NewKalmanFilterModel
	
end

function KalmanFilterModel:train(previousStateMatrix, currentStateMatrix)
	
	local numberOfData = #previousStateMatrix

	if (numberOfData ~= #currentStateMatrix) then error("The previous state matrix and the current state natrux does not contain the same number of rows.") end
	
	local numberOfStates = #previousStateMatrix[1]
	
	local dimensionSizeArray = {numberOfData, numberOfStates}
	
	local numberOfStatesDimensionSizeArray = {numberOfStates, numberOfStates}
	
	local stateTransitionMatrix = self.stateTransitionMatrix or AqwamTensorLibrary:createTensor(numberOfStatesDimensionSizeArray)

	local observationMatrix = self.observationMatrix or AqwamTensorLibrary:createTensor(numberOfStatesDimensionSizeArray)

	local processNoiseCovarianceMatrix = self.processNoiseCovarianceMatrix or AqwamTensorLibrary:createTensor(numberOfStatesDimensionSizeArray)

	local observationNoiseCovarianceMatrix = self.observationNoiseCovarianceMatrix or AqwamTensorLibrary:createTensor(numberOfStatesDimensionSizeArray)

	local controlVector = self.controlVector or AqwamTensorLibrary:createTensor(dimensionSizeArray)

	local controlInputMatrix = self.controlInputMatrix or AqwamTensorLibrary:createTensor(numberOfStatesDimensionSizeArray)
	
	--local observationNoiseMatrix = AqwamTensorLibrary:createRandomNormalTensor(dimensionSizeArray)

	--local processNoiseMatrix = AqwamTensorLibrary:createRandomNormalTensor(dimensionSizeArray)

	local ModelParameters = self.ModelParameters or {}
	
	local priorStateMatrix = ModelParameters[1]
	
	local priorCovarianceMatrix = ModelParameters[2]
	
	if (not priorStateMatrix) then
		
		local priorStateMatrixPart1 = AqwamTensorLibrary:dotProduct(previousStateMatrix, stateTransitionMatrix) -- m x n, n x n

		local priorStateMatrixPart2

		if (controlInputMatrix) then

			priorStateMatrixPart2 = AqwamTensorLibrary:dotProduct(controlVector, controlInputMatrix) -- m x n, n x n

		else

			priorStateMatrixPart2 = controlVector -- m x n

		end

		priorStateMatrix = AqwamTensorLibrary:add(priorStateMatrixPart1, priorStateMatrixPart2) -- m x n
		
	end
	
	local transposedStateTransitionMatrix = AqwamTensorLibrary:transpose(stateTransitionMatrix) -- n x n
	
	local dotProductStateTransitionMatrix = AqwamTensorLibrary:dotProduct(stateTransitionMatrix, transposedStateTransitionMatrix) -- n x n, n x n
	
	if (not priorCovarianceMatrix) then
		
		priorCovarianceMatrix = AqwamTensorLibrary:add(dotProductStateTransitionMatrix, processNoiseCovarianceMatrix)
		
	end

	local priorCovarianceMatrixPart1 = AqwamTensorLibrary:dotProduct(priorCovarianceMatrix, dotProductStateTransitionMatrix) -- n x n,  -- n x n

	priorCovarianceMatrix = AqwamTensorLibrary:add(priorCovarianceMatrixPart1, processNoiseCovarianceMatrix)  -- n x n, n x n
	
	local transposedObservationMatrix = AqwamTensorLibrary:transpose(observationMatrix)
	
	local trueStateMatrix = AqwamTensorLibrary:dotProduct(currentStateMatrix, observationMatrix)
	
	local innovationMatrixPart1 = AqwamTensorLibrary:dotProduct(observationMatrix, priorStateMatrix)
	
	local innovationMatrix = AqwamTensorLibrary:subtract(trueStateMatrix, innovationMatrixPart1)
	
	local innovationCovarianceMatrixPart1 = AqwamTensorLibrary:dotProduct(observationMatrix, priorCovarianceMatrix, transposedObservationMatrix)
	
	local innovationCovarianceMatrix = AqwamTensorLibrary:add(innovationCovarianceMatrixPart1, observationNoiseCovarianceMatrix)
	
	local inverseInovationCovarianceMatrix = AqwamTensorLibrary:inverse(innovationCovarianceMatrix)
	
	local optimalKalmanGainMatrix = AqwamTensorLibrary:dotProduct(priorCovarianceMatrix, transposedObservationMatrix, inverseInovationCovarianceMatrix)
	
	local posteriorMatrixPart1 = AqwamTensorLibrary:dotProduct(innovationMatrix, optimalKalmanGainMatrix)
	
	local posteriorMatrix = AqwamTensorLibrary:add(priorStateMatrix, posteriorMatrixPart1)
	
	local identityMatrix = AqwamTensorLibrary:createIdentityMatrix({#priorCovarianceMatrix, #priorCovarianceMatrix})

	local KHMatrix = AqwamTensorLibrary:dotProduct(optimalKalmanGainMatrix, observationMatrix)

	local identityMinusKHMatrix = AqwamTensorLibrary:subtract(identityMatrix, KHMatrix)

	local transposedIdentityMinusKHMatrix = AqwamTensorLibrary:transpose(identityMinusKHMatrix)

	local josephFormMatrixPart1 = AqwamTensorLibrary:dotProduct(identityMinusKHMatrix, priorCovarianceMatrix, transposedIdentityMinusKHMatrix)

	local transposedKalmanGainMatrix = AqwamTensorLibrary:transpose(optimalKalmanGainMatrix)

	local josephFormMatrixPart2 = AqwamTensorLibrary:dotProduct(optimalKalmanGainMatrix, observationNoiseCovarianceMatrix, transposedKalmanGainMatrix)

	local posteriorCovarianceMatrix = AqwamTensorLibrary:add(josephFormMatrixPart1, josephFormMatrixPart2)
	
	local residualMatrixPart1 = AqwamTensorLibrary:dotProduct(posteriorMatrix, observationMatrix)
	
	local residualMatrix = AqwamTensorLibrary:subtract(trueStateMatrix, residualMatrixPart1)
	
	self.ModelParameters = {posteriorMatrix, posteriorCovarianceMatrix}

end

function KalmanFilterModel:predict(stateMatrix)
	
	local stateTransitionMatrix = self.stateTransitionMatrix

	local controlVector = self.controlVector
	
	local controlInputMatrix = self.controlInputMatrix
	
	--local processNoiseMatrix = self.processNoiseMatrix
	
	local nextStateMatrixPart1 = AqwamTensorLibrary:dotProduct(stateMatrix, stateTransitionMatrix) -- m x n, n x n
	
	local nextStateMatrixPart2
	
	if (controlInputMatrix) then
		
		nextStateMatrixPart2 = AqwamTensorLibrary:dotProduct(controlVector, controlInputMatrix) -- m x n, n x n
		
	else
		
		nextStateMatrixPart2 = controlVector -- m x n
		
	end
	
	local nextStateMatrix = AqwamTensorLibrary:add(nextStateMatrixPart1, nextStateMatrixPart2) -- m x n
	
	return nextStateMatrix
	
end

return KalmanFilterModel
