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
	
	NewKalmanFilterModel.stateTransitionModelMatrix = parameterDictionary.stateTransitionModelMatrix
	
	NewKalmanFilterModel.observationModelMatrix = parameterDictionary.observationModelMatrix
	
	NewKalmanFilterModel.processNoiseCovarianceMatrix = parameterDictionary.processNoiseCovarianceMatrix
	
	NewKalmanFilterModel.observationNoiseCovarianceMatrix = parameterDictionary.observationNoiseCovarianceMatrix
	
	NewKalmanFilterModel.controlInputMatrix = parameterDictionary.controlInputMatrix
	
	NewKalmanFilterModel.controlVector = parameterDictionary.controlVector

	return NewKalmanFilterModel
	
end

function KalmanFilterModel:train(previousStateMatrix, currentStateMatrix)
	
	local numberOfData = #previousStateMatrix

	if (numberOfData ~= #currentStateMatrix) then error("The previous state matrix and the current state matrix does not contain the same number of rows.") end
	
	local numberOfStates = #previousStateMatrix[1]
	
	local dimensionSizeArray = {numberOfData, numberOfStates}
	
	local numberOfStatesDimensionSizeArray = {numberOfStates, numberOfStates}
	
	local stateTransitionModelMatrix = self.stateTransitionModelMatrix or AqwamTensorLibrary:createTensor(numberOfStatesDimensionSizeArray)

	local observationModelMatrix = self.observationModelMatrix or AqwamTensorLibrary:createTensor(numberOfStatesDimensionSizeArray)
	
	local controlInputMatrix = self.controlInputMatrix

	local controlVector = self.controlVector -- 1 x n

	--local observationNoiseMatrix = AqwamTensorLibrary:createRandomNormalTensor(dimensionSizeArray)

	--local processNoiseMatrix = AqwamTensorLibrary:createRandomNormalTensor(dimensionSizeArray)

	local ModelParameters = self.ModelParameters or {}
	
	local priorStateMatrix = ModelParameters[1]
	
	local priorCovarianceMatrix = ModelParameters[2]
	
	local observationNoiseCovarianceMatrix = ModelParameters[3] or AqwamTensorLibrary:createTensor(numberOfStatesDimensionSizeArray)
	
	local processNoiseCovarianceMatrix = ModelParameters[4] or AqwamTensorLibrary:createTensor(numberOfStatesDimensionSizeArray)
	
	if (not priorStateMatrix) then
		
		local priorStateMatrixPart1 = AqwamTensorLibrary:dotProduct(previousStateMatrix, stateTransitionModelMatrix) -- m x n, n x n

		local priorStateMatrixPart2

		if (controlInputMatrix) then

			priorStateMatrixPart2 = AqwamTensorLibrary:dotProduct(controlVector, controlInputMatrix) -- m x n, n x n

		elseif (controlVector) then

			priorStateMatrixPart2 = controlVector -- m x n

		end
		
		if (priorStateMatrixPart2) then
			
			priorStateMatrix = AqwamTensorLibrary:add(priorStateMatrixPart1, priorStateMatrixPart2) -- m x n
			
		else
			
			priorStateMatrix = priorStateMatrixPart1
			
		end
		
	end
	
	local transposedStateTransitionModelMatrix = AqwamTensorLibrary:transpose(stateTransitionModelMatrix) -- n x n
	
	local dotProductStateTransitionMatrix = AqwamTensorLibrary:dotProduct(stateTransitionModelMatrix, transposedStateTransitionModelMatrix) -- n x n, n x n
	
	if (not priorCovarianceMatrix) then
		
		priorCovarianceMatrix = AqwamTensorLibrary:add(dotProductStateTransitionMatrix, processNoiseCovarianceMatrix)
		
	end

	local priorCovarianceMatrixPart1 = AqwamTensorLibrary:dotProduct(stateTransitionModelMatrix, priorCovarianceMatrix, transposedStateTransitionModelMatrix) -- n x n, -- n x n

	priorCovarianceMatrix = AqwamTensorLibrary:add(priorCovarianceMatrixPart1, processNoiseCovarianceMatrix)  -- n x n, n x n
	
	local observationMatrix = AqwamTensorLibrary:dotProduct(currentStateMatrix, observationModelMatrix)
	
	local innovationMatrixPart1 = AqwamTensorLibrary:dotProduct(observationMatrix, priorStateMatrix)
	
	local innovationMatrix = AqwamTensorLibrary:subtract(observationMatrix, innovationMatrixPart1)
	
	local transposedObservationModelMatrix = AqwamTensorLibrary:transpose(observationModelMatrix)
	
	local innovationCovarianceMatrixPart1 = AqwamTensorLibrary:dotProduct(observationModelMatrix, priorCovarianceMatrix, transposedObservationModelMatrix)
	
	local innovationCovarianceMatrix = AqwamTensorLibrary:add(innovationCovarianceMatrixPart1, observationNoiseCovarianceMatrix)
	
	local inverseInovationCovarianceMatrix = AqwamTensorLibrary:inverse(innovationCovarianceMatrix)
	
	local optimalKalmanGainMatrix = AqwamTensorLibrary:dotProduct(priorCovarianceMatrix, transposedObservationModelMatrix, inverseInovationCovarianceMatrix)
	
	local posteriorMatrixPart1 = AqwamTensorLibrary:dotProduct(innovationMatrix, optimalKalmanGainMatrix)
	
	local posteriorMatrix = AqwamTensorLibrary:add(priorStateMatrix, posteriorMatrixPart1)
	
	local identityMatrix = AqwamTensorLibrary:createIdentityMatrix({#priorCovarianceMatrix, #priorCovarianceMatrix})

	local KHMatrix = AqwamTensorLibrary:dotProduct(optimalKalmanGainMatrix, observationModelMatrix)

	local identityMinusKHMatrix = AqwamTensorLibrary:subtract(identityMatrix, KHMatrix)

	local transposedIdentityMinusKHMatrix = AqwamTensorLibrary:transpose(identityMinusKHMatrix)

	local josephFormMatrixPart1 = AqwamTensorLibrary:dotProduct(identityMinusKHMatrix, priorCovarianceMatrix, transposedIdentityMinusKHMatrix)

	local transposedKalmanGainMatrix = AqwamTensorLibrary:transpose(optimalKalmanGainMatrix)

	local josephFormMatrixPart2 = AqwamTensorLibrary:dotProduct(optimalKalmanGainMatrix, observationNoiseCovarianceMatrix, transposedKalmanGainMatrix)

	local posteriorCovarianceMatrix = AqwamTensorLibrary:add(josephFormMatrixPart1, josephFormMatrixPart2)
	
	local residualMatrixPart1 = AqwamTensorLibrary:dotProduct(posteriorMatrix, observationModelMatrix)
	
	local residualMatrix = AqwamTensorLibrary:subtract(observationMatrix, residualMatrixPart1)
	
	-- Need to double check this part for calculating covariance matrices.
	
	local transposedResidualMatrix = AqwamTensorLibrary:transpose(residualMatrix)
	
	local dotProductResidualMatrix = AqwamTensorLibrary:dotProduct(transposedResidualMatrix, residualMatrix)
	
	observationNoiseCovarianceMatrix = AqwamTensorLibrary:divide(dotProductResidualMatrix, numberOfData)
	
	local processNoiseCovarianceMatrixPart1 = AqwamTensorLibrary:dotProduct(stateTransitionModelMatrix, priorCovarianceMatrix, transposedStateTransitionModelMatrix)
	
	processNoiseCovarianceMatrix = AqwamTensorLibrary:subtract(posteriorCovarianceMatrix, processNoiseCovarianceMatrixPart1)
	
	--
	
	self.ModelParameters = {posteriorMatrix, posteriorCovarianceMatrix, observationNoiseCovarianceMatrix, processNoiseCovarianceMatrix}

end

function KalmanFilterModel:predict(stateMatrix)
	
	local stateTransitionModelMatrix = self.stateTransitionModelMatrix
	
	local controlInputMatrix = self.controlInputMatrix

	local controlVector = self.controlVector -- 1 x n
	
	--local processNoiseMatrix = self.processNoiseMatrix
	
	local nextStateMatrixPart1 = AqwamTensorLibrary:dotProduct(stateMatrix, stateTransitionModelMatrix) -- m x n, n x n

	local nextStateMatrixPart2
	
	local nextStateMatrix

	if (controlInputMatrix) then

		nextStateMatrixPart2 = AqwamTensorLibrary:dotProduct(controlVector, controlInputMatrix) -- m x n, n x n

	elseif (controlVector) then

		nextStateMatrixPart2 = controlVector -- m x n

	end
	
	if (nextStateMatrixPart2) then

		nextStateMatrix = AqwamTensorLibrary:add(nextStateMatrixPart1, nextStateMatrixPart2) -- m x n

	else

		nextStateMatrix = nextStateMatrixPart1

	end
	
	return nextStateMatrix
	
end

return KalmanFilterModel
