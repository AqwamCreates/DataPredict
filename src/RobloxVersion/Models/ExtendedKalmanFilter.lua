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

local defaultNoiseValue = 1 -- Do not use very small value for this. It will cause the Mahalanobis distance to have very large values.

local defaultLossFunction = "L2"

local defaultUseJosephForm = true

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
	
	NewExtendedKalmanFilterModel.lossFunction = parameterDictionary.lossFunction or defaultLossFunction
	
	NewExtendedKalmanFilterModel.useJosephForm = NewExtendedKalmanFilterModel:getValueOrDefaultValue(parameterDictionary.useJosephForm, defaultUseJosephForm)
	
	return NewExtendedKalmanFilterModel
	
end

function ExtendedKalmanFilterModel:train(previousStateMatrix, currentStateMatrix)

	local numberOfData = #previousStateMatrix

	if (numberOfData ~= #currentStateMatrix) then error("The number of data in the previous state vector is not equal to the number of data in the current state vector.") end
	
	local numberOfStates = #previousStateMatrix[1]

	if (numberOfStates ~= #currentStateMatrix[1]) then error("The number of states in the previous state vector is not equal to the number of states in the current state vector.") end
	
	local numberOfStatesDimensionSizeArray = {numberOfStates, numberOfStates}

	local controlVector = self.controlVector
	
	local noiseValue = self.noiseValue
	
	local lossFunction = self.lossFunction
	
	local useJosephForm = self.useJosephForm
	
	local ModelParameters = self.ModelParameters or {}
	
	local priorStateMatrix = ModelParameters[1] or previousStateMatrix

	local priorCovarianceMatrix = ModelParameters[2] or AqwamTensorLibrary:createIdentityTensor(numberOfStatesDimensionSizeArray)
	
	local observationNoiseCovarianceMatrix = self.observationNoiseCovarianceMatrix or AqwamTensorLibrary:createIdentityTensor(numberOfStatesDimensionSizeArray, noiseValue)

	local processNoiseCovarianceMatrix = self.processNoiseCovarianceMatrix or AqwamTensorLibrary:createIdentityTensor(numberOfStatesDimensionSizeArray, noiseValue)

	local predictedStateMatrix = self.stateFunction(priorStateMatrix, controlVector)
	
	local stateTransitionJacobianMatrix = self.stateTransitionJacobianFunction(priorStateMatrix, controlVector)

	local transposedStateTransitionJacobianMatrix = AqwamTensorLibrary:transpose(stateTransitionJacobianMatrix)

	local predictedCovarianceMatrixPart1 = AqwamTensorLibrary:dotProduct(stateTransitionJacobianMatrix, priorCovarianceMatrix, transposedStateTransitionJacobianMatrix)

	local predictedCovarianceMatrix = AqwamTensorLibrary:add(predictedCovarianceMatrixPart1, processNoiseCovarianceMatrix)

	local observationMatrix = self.observationStateFunction(predictedStateMatrix)
	
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
	
	local posteriorCovarianceMatrix = AqwamTensorLibrary:dotProduct(identityMinusKHMatrix, priorCovarianceMatrix)

	if (useJosephForm) then

		local transposedIdentityMinusKHMatrix = AqwamTensorLibrary:transpose(identityMinusKHMatrix)

		local josephFormMatrixPart1 = AqwamTensorLibrary:dotProduct(posteriorCovarianceMatrix, transposedIdentityMinusKHMatrix)

		local transposedKalmanGainMatrix = AqwamTensorLibrary:transpose(kalmanGainMatrix)

		local josephFormMatrixPart2 = AqwamTensorLibrary:dotProduct(kalmanGainMatrix, observationNoiseCovarianceMatrix, transposedKalmanGainMatrix)

		posteriorCovarianceMatrix = AqwamTensorLibrary:add(josephFormMatrixPart1, josephFormMatrixPart2)

	end
	
	local meanCorrectionMatrix = AqwamTensorLibrary:mean(posteriorStateMatrixPart1, 1)

	self.ModelParameters = {posteriorStateMatrix, posteriorCovarianceMatrix, meanCorrectionMatrix}
	
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

function ExtendedKalmanFilterModel:predict(stateMatrix)
	
	local ModelParameters = self.ModelParameters or {}
	
	local meanCorrectionMatrix = ModelParameters[3]
	
	local nextStateMatrix = self.stateFunction(stateMatrix, self.controlVector)
	
	if (meanCorrectionMatrix) then nextStateMatrix = AqwamTensorLibrary:add(nextStateMatrix, meanCorrectionMatrix) end
	
	return nextStateMatrix
	
end

return ExtendedKalmanFilterModel
