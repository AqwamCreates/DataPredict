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

local defaultNoiseValue = 1 -- Do not use very small value for this. It will cause the Mahalanobis distance to have very large values.

local defaultLossFunction = "L2"

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
	
	NewKalmanFilterModel.noiseValue = parameterDictionary.noiseValue or defaultNoiseValue
	
	NewKalmanFilterModel.lossFunction = parameterDictionary.lossFunction or defaultLossFunction

	return NewKalmanFilterModel
	
end

function KalmanFilterModel:train(previousStateMatrix, currentStateMatrix)

	local numberOfData = #previousStateMatrix

	if (numberOfData ~= #currentStateMatrix) then error("The number of data in the previous state vector is not equal to the number of data in the current state vector.") end
	
	local numberOfStates = #previousStateMatrix[1]

	if (numberOfStates ~= #currentStateMatrix[1]) then error("The number of current state columns is not equal to the number of states.") end
	
	local numberOfStatesDimensionSizeArray = {numberOfStates, numberOfStates}
	
	local stateTransitionModelMatrix = self.stateTransitionModelMatrix or AqwamTensorLibrary:createIdentityTensor(numberOfStatesDimensionSizeArray)

	local observationModelMatrix = self.observationModelMatrix or AqwamTensorLibrary:createIdentityTensor(numberOfStatesDimensionSizeArray)
	
	local controlInputMatrix = self.controlInputMatrix

	local controlVector = self.controlVector -- 1 x n
	
	local noiseValue = self.noiseValue
	
	local lossFunction = self.lossFunction

	--local observationNoiseMatrix = AqwamTensorLibrary:createRandomNormalTensor(dimensionSizeArray)

	--local processNoiseMatrix = AqwamTensorLibrary:createRandomNormalTensor(dimensionSizeArray)

	local ModelParameters = self.ModelParameters or {}
	
	local priorStateMatrix = ModelParameters[1]
	
	local priorCovarianceMatrix = ModelParameters[2]
	
	local observationNoiseCovarianceMatrix = self.observationNoiseCovarianceMatrix or AqwamTensorLibrary:createIdentityTensor(numberOfStatesDimensionSizeArray, noiseValue)

	local processNoiseCovarianceMatrix = self.processNoiseCovarianceMatrix or AqwamTensorLibrary:createIdentityTensor(numberOfStatesDimensionSizeArray, noiseValue)
	
	previousStateMatrix = AqwamTensorLibrary:transpose(previousStateMatrix) -- m x n -> n x m
	
	currentStateMatrix = AqwamTensorLibrary:transpose(currentStateMatrix) -- m x n -> n x m
	
	if (not priorStateMatrix) then
		
		local priorStateMatrixPart1 = AqwamTensorLibrary:dotProduct(stateTransitionModelMatrix, previousStateMatrix)
		
		local priorStateMatrixPart2

		if (controlInputMatrix) then

			priorStateMatrixPart2 = AqwamTensorLibrary:dotProduct(controlInputMatrix, controlVector)

		elseif (controlVector) then

			priorStateMatrixPart2 = controlVector

		end
		
		if (priorStateMatrixPart2) then
			
			priorStateMatrix = AqwamTensorLibrary:add(priorStateMatrixPart1, priorStateMatrixPart2)
			
		else
			
			priorStateMatrix = priorStateMatrixPart1
			
		end
		
	end
	
	local transposedStateTransitionModelMatrix = AqwamTensorLibrary:transpose(stateTransitionModelMatrix)
	
	if (not priorCovarianceMatrix) then
		
		local priorCovarianceMatrixPart1 = AqwamTensorLibrary:dotProduct(stateTransitionModelMatrix, stateTransitionModelMatrix, transposedStateTransitionModelMatrix) -- n x n, n x n
		
		priorCovarianceMatrix = AqwamTensorLibrary:add(priorCovarianceMatrixPart1, processNoiseCovarianceMatrix)
		
	end

	local priorCovarianceMatrixPart1 = AqwamTensorLibrary:dotProduct(stateTransitionModelMatrix, priorCovarianceMatrix, transposedStateTransitionModelMatrix) -- n x n, -- n x n

	priorCovarianceMatrix = AqwamTensorLibrary:add(priorCovarianceMatrixPart1, processNoiseCovarianceMatrix)
	
	local transposedObservationModelMatrix = AqwamTensorLibrary:transpose(observationModelMatrix)
	
	local observationMatrix = currentStateMatrix
	
	local innovationMatrixPart1 = AqwamTensorLibrary:dotProduct(observationModelMatrix, priorStateMatrix)
	
	local innovationMatrix = AqwamTensorLibrary:subtract(observationMatrix, innovationMatrixPart1)
	
	local transposedObservationModelMatrix = AqwamTensorLibrary:transpose(observationModelMatrix)
	
	local innovationCovarianceMatrixPart1 = AqwamTensorLibrary:dotProduct(observationModelMatrix, priorCovarianceMatrix, transposedObservationModelMatrix)
	
	local innovationCovarianceMatrix = AqwamTensorLibrary:add(innovationCovarianceMatrixPart1, observationNoiseCovarianceMatrix)
	
	local inverseInnovationCovarianceMatrix = AqwamTensorLibrary:inverse(innovationCovarianceMatrix)
	
	if (not inverseInnovationCovarianceMatrix) then error("Could not find the inverse of innovation covariance matrix.") end
	
	local optimalKalmanGainMatrix = AqwamTensorLibrary:dotProduct(priorCovarianceMatrix, transposedObservationModelMatrix, inverseInnovationCovarianceMatrix)
	
	local posteriorMatrixPart1 = AqwamTensorLibrary:dotProduct(optimalKalmanGainMatrix, innovationMatrix)
	
	local posteriorStateMatrix = AqwamTensorLibrary:add(priorStateMatrix, posteriorMatrixPart1)
	
	local identityMatrix = AqwamTensorLibrary:createIdentityTensor(numberOfStatesDimensionSizeArray)

	local KHMatrix = AqwamTensorLibrary:dotProduct(optimalKalmanGainMatrix, observationModelMatrix)

	local identityMinusKHMatrix = AqwamTensorLibrary:subtract(identityMatrix, KHMatrix)

	local transposedIdentityMinusKHMatrix = AqwamTensorLibrary:transpose(identityMinusKHMatrix)

	local josephFormMatrixPart1 = AqwamTensorLibrary:dotProduct(identityMinusKHMatrix, priorCovarianceMatrix, transposedIdentityMinusKHMatrix)

	local transposedKalmanGainMatrix = AqwamTensorLibrary:transpose(optimalKalmanGainMatrix)

	local josephFormMatrixPart2 = AqwamTensorLibrary:dotProduct(optimalKalmanGainMatrix, observationNoiseCovarianceMatrix, transposedKalmanGainMatrix)

	local posteriorCovarianceMatrix = AqwamTensorLibrary:add(josephFormMatrixPart1, josephFormMatrixPart2)
	
	local residualMatrixPart1 = AqwamTensorLibrary:dotProduct(observationModelMatrix, observationMatrix)
	
	--[[
	
	local residualMatrix = AqwamTensorLibrary:subtract(observationMatrix, residualMatrixPart1)
	
	-- Need to double check this part for calculating covariance matrices.
	
	local transposedResidualMatrix = AqwamTensorLibrary:transpose(residualMatrix)
	
	local dotProductResidualMatrix = AqwamTensorLibrary:dotProduct(transposedResidualMatrix, residualMatrix)
	
	local processNoiseCovarianceMatrixPart1 = AqwamTensorLibrary:dotProduct(stateTransitionModelMatrix, priorCovarianceMatrix, transposedStateTransitionModelMatrix)
	
	processNoiseCovarianceMatrix = AqwamTensorLibrary:subtract(posteriorCovarianceMatrix, processNoiseCovarianceMatrixPart1)
	
	--]]
	
	self.ModelParameters = {posteriorStateMatrix, posteriorCovarianceMatrix}
	
	-- Returning this as a cost like other models.

	local lossMatrix = innovationMatrix
	
	if (lossFunction == "L1") then
		
		lossMatrix = AqwamTensorLibrary:applyFunction(math.abs, lossMatrix)
		
	elseif (lossFunction == "L2") then
		
		lossMatrix = AqwamTensorLibrary:power(lossMatrix, 2)
		
	elseif (lossFunction  == "Mahalanobis") then
		
		local transposedInnovationMatrix = AqwamTensorLibrary:transpose(innovationMatrix)
		
		lossMatrix = AqwamTensorLibrary:dotProduct(transposedInnovationMatrix, inverseInnovationCovarianceMatrix, innovationMatrix)
		
	else
		
		error("Invalid loss function.")
		
	end
	
	local cost = AqwamTensorLibrary:sum(lossMatrix)
	
	cost = cost / numberOfData

	return {cost}

end

function KalmanFilterModel:predict(stateMatrix)
	
	local stateTransitionModelMatrix = self.stateTransitionModelMatrix
	
	local controlInputMatrix = self.controlInputMatrix

	local controlVector = self.controlVector -- 1 x n
	
	--local processNoiseMatrix = self.processNoiseMatrix
	
	stateMatrix = AqwamTensorLibrary:transpose(stateMatrix)

	local nextStateMatrixPart1 = AqwamTensorLibrary:dotProduct(stateTransitionModelMatrix, stateMatrix)

	local nextStateMatrixPart2
	
	local nextStateMatrix

	if (controlInputMatrix) then

		nextStateMatrixPart2 = AqwamTensorLibrary:dotProduct(controlInputMatrix, controlVector)

	elseif (controlVector) then

		nextStateMatrixPart2 = controlVector

	end

	if (nextStateMatrixPart2) then

		nextStateMatrix = AqwamTensorLibrary:add(nextStateMatrixPart1, nextStateMatrixPart2)

	else

		nextStateMatrix = nextStateMatrixPart1

	end
	
	return nextStateMatrix
	
end

return KalmanFilterModel
