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

ExtendedKalmanFilterModel = {}

ExtendedKalmanFilterModel.__index = ExtendedKalmanFilterModel

setmetatable(ExtendedKalmanFilterModel, BaseModel)

local defaultNoiseValue = 1e-16

local function defaultStateFunction(previousStateMatrix, controlVector)
	
	return AqwamTensorLibrary:add(previousStateMatrix, controlVector)
	
end

local function defaultObservationStateFunction(stateMatrix)

	return stateMatrix

end

local function defaultStateTransitionJacobianFunction(stateMatrix, controlVector)
	
	return AqwamTensorLibrary:createIdentityTensor({#stateMatrix, #stateMatrix})
	
end

local function defaultObservationJacobianFunction(stateMatrix)
	
	return AqwamTensorLibrary:createIdentityTensor({#stateMatrix, #stateMatrix})
	
end

function ExtendedKalmanFilterModel.new(parameterDictionary)
	
	parameterDictionary = parameterDictionary or {}

	local NewExtendedKalmanFilterModel = BaseModel.new(parameterDictionary)

	setmetatable(NewExtendedKalmanFilterModel, ExtendedKalmanFilterModel)

	NewExtendedKalmanFilterModel:setName("ExtendedKalmanFilter")
	
	NewExtendedKalmanFilterModel.stateTransitionModelMatrix = parameterDictionary.stateTransitionModelMatrix
	
	NewExtendedKalmanFilterModel.observationNoiseCovarianceMatrix = parameterDictionary.observationNoiseCovarianceMatrix
	
	NewExtendedKalmanFilterModel.processNoiseCovarianceMatrix = parameterDictionary.processNoiseCovarianceMatrix
	
	NewExtendedKalmanFilterModel.controlVector = parameterDictionary.controlVector
	
	NewExtendedKalmanFilterModel.noiseValue = parameterDictionary.noiseValue or defaultNoiseValue
	
	NewExtendedKalmanFilterModel.stateFunction = parameterDictionary.stateFunction or defaultStateFunction
	
	NewExtendedKalmanFilterModel.observationStateFunction = parameterDictionary.observationStateFunction or defaultObservationStateFunction
	
	NewExtendedKalmanFilterModel.stateTransitionJacobianFunction = parameterDictionary.stateTransitionJacobianFunction or defaultStateTransitionJacobianFunction
	
	NewExtendedKalmanFilterModel.observationJacobianFunction = parameterDictionary.observationJacobianFunction or defaultObservationJacobianFunction
	
	return NewExtendedKalmanFilterModel
	
end

function ExtendedKalmanFilterModel:train(previousStateMatrix, currentStateMatrix)

	local numberOfData = #previousStateMatrix

	if (numberOfData ~= #currentStateMatrix) then error("The number of data in the previous state vector is not equal to the number of data in the current state vector.") end
	
	local numberOfStates = #previousStateMatrix[1]

	if (numberOfStates ~= #currentStateMatrix[1]) then error("The number of states in the previous state vector is not equal to the number of states in the current state vector.") end
	
	local numberOfStatesDimensionSizeArray = {numberOfStates, numberOfStates}
	
	local stateFunction = self.stateFunction
	
	local observationStateFunction = self.observationStateFunction

	local controlVector = self.controlVector
	
	local noiseValue = self.noiseValue
	
	local ModelParameters = self.ModelParameters or {}
	
	local priorStateMatrix = ModelParameters[1]

	local priorCovarianceMatrix = ModelParameters[2]
	
	local observationNoiseCovarianceMatrix = self.observationNoiseCovarianceMatrix or AqwamTensorLibrary:createIdentityTensor(numberOfStatesDimensionSizeArray, noiseValue)

	local processNoiseCovarianceMatrix = self.processNoiseCovarianceMatrix or AqwamTensorLibrary:createIdentityTensor(numberOfStatesDimensionSizeArray, noiseValue)
	
	if (not priorStateMatrix) then priorStateMatrix = previousStateMatrix end
	
	if (not priorCovarianceMatrix) then priorCovarianceMatrix = AqwamTensorLibrary:createIdentityTensor(numberOfStatesDimensionSizeArray) end

	local predictedStateMatrix = stateFunction(priorStateMatrix, controlVector)
	
	local stateTransitionJacobianMatrix = self.stateTransitionJacobianFunction(priorStateMatrix, controlVector)

	local transposedStateTransitionJacobianMatrix = AqwamTensorLibrary:transpose(stateTransitionJacobianMatrix)

	local predictedCovarianceMatrixPart1 = AqwamTensorLibrary:dotProduct(stateTransitionJacobianMatrix, priorCovarianceMatrix, transposedStateTransitionJacobianMatrix)

	local predictedCovarianceMatrix = AqwamTensorLibrary:add(predictedCovarianceMatrixPart1, processNoiseCovarianceMatrix)

	local observationMatrix = observationStateFunction(predictedStateMatrix)
	
	local observationJacobianMatrix = self.observationJacobianFunction(predictedStateMatrix)
	
	local transposedObservationJacobianMatrix = AqwamTensorLibrary:transpose(observationJacobianMatrix)

	local innovationMatrix = AqwamTensorLibrary:subtract(currentStateMatrix, observationMatrix)

	local innovationCovarianceMatrixPart1 = AqwamTensorLibrary:dotProduct(observationJacobianMatrix, predictedCovarianceMatrix, transposedObservationJacobianMatrix)

	local innovationCovarianceMatrix = AqwamTensorLibrary:add(innovationCovarianceMatrixPart1, observationNoiseCovarianceMatrix)

	local inverseInnovationCovarianceMatrix = AqwamTensorLibrary:inverse(innovationCovarianceMatrix)

	local kalmanGainMatrix = AqwamTensorLibrary:dotProduct(predictedCovarianceMatrix, transposedObservationJacobianMatrix, inverseInnovationCovarianceMatrix)

	local posteriorStateMatrixPart1 = AqwamTensorLibrary:dotProduct(kalmanGainMatrix, innovationMatrix)
	
	local posteriorStateMatrix = AqwamTensorLibrary:add(predictedStateMatrix, posteriorStateMatrixPart1)

	local identityMatrix = AqwamTensorLibrary:createIdentityTensor({#priorCovarianceMatrix, #priorCovarianceMatrix})
	
	local KHMatrix = AqwamTensorLibrary:dotProduct(kalmanGainMatrix, observationJacobianMatrix)
	
	local identityMinusKHMatrix = AqwamTensorLibrary:subtract(identityMatrix, KHMatrix)

	local posteriorCovarianceMatrix = AqwamTensorLibrary:dotProduct(identityMinusKHMatrix, predictedCovarianceMatrix)

	self.ModelParameters = {posteriorStateMatrix, posteriorCovarianceMatrix}

end

function ExtendedKalmanFilterModel:predict(stateMatrix)
	
	return self.stateFunction(self.stateFunction, self.controlVector)
	
end

return ExtendedKalmanFilterModel
