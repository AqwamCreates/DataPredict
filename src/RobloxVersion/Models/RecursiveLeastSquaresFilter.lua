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

local RecursiveLeastSquaresFilterModel = {}

RecursiveLeastSquaresFilterModel.__index = RecursiveLeastSquaresFilterModel

setmetatable(RecursiveLeastSquaresFilterModel, BaseModel)

local defaultLossFunction = "L2"

local defaultForgetFactor = 1

function RecursiveLeastSquaresFilterModel.new(parameterDictionary)
	
	parameterDictionary = parameterDictionary or {}

	local NewRecursiveLeastSquaresFilterModel = BaseModel.new(parameterDictionary)

	setmetatable(NewRecursiveLeastSquaresFilterModel, RecursiveLeastSquaresFilterModel)

	NewRecursiveLeastSquaresFilterModel:setName("RecursiveLeastSquaresFilter")
	
	NewRecursiveLeastSquaresFilterModel.lossFunction = parameterDictionary.lossFunction or defaultLossFunction
	
	NewRecursiveLeastSquaresFilterModel.forgetFactor = parameterDictionary.forgetFactor or defaultForgetFactor

	return NewRecursiveLeastSquaresFilterModel
	
end

function RecursiveLeastSquaresFilterModel:train(previousStateMatrix, currentStateMatrix)

	local numberOfData = #previousStateMatrix

	if (numberOfData ~= #currentStateMatrix) then error("The number of data in the previous state vector is not equal to the number of data in the current state vector.") end
	
	local numberOfStates = #previousStateMatrix[1]

	if (numberOfStates ~= #currentStateMatrix[1]) then error("The number of current state columns is not equal to the number of states.") end
	
	local lossFunction = self.lossFunction
	
	local forgetFactor = self.forgetFactor
	
	local ModelParameters = self.ModelParameters or {}
	
	local weightVector = ModelParameters[1] or self:initializeMatrixBasedOnMode({numberOfStates, 1})
	
	local errorCovarianceMatrix = ModelParameters[2] or AqwamTensorLibrary:createIdentityTensor({numberOfStates, numberOfStates})
	
	local predictedCurrentStateMatrix = AqwamTensorLibrary:dotProduct(previousStateMatrix, weightVector)
	
	local lossMatrix = AqwamTensorLibrary:subtract(currentStateMatrix, predictedCurrentStateMatrix) -- m x n
	
	local kalmanGainVectorNumerator = AqwamTensorLibrary:dotProduct(previousStateMatrix, errorCovarianceMatrix) -- m x n
	
	local transposedPreviousStateMatrix = AqwamTensorLibrary:transpose(previousStateMatrix)
	
	local kalmanGainVectorDenominator = AqwamTensorLibrary:dotProduct(previousStateMatrix, errorCovarianceMatrix, transposedPreviousStateMatrix) -- m x m
	
	kalmanGainVectorDenominator = AqwamTensorLibrary:add(forgetFactor, kalmanGainVectorDenominator)
	
	local kalmanGainVector = AqwamTensorLibrary:divide(kalmanGainVectorNumerator, kalmanGainVectorDenominator) -- if m = 1, then 1 x n.
	
	local transposedKalmanGainVector = AqwamTensorLibrary:transpose(kalmanGainVector)
	
	local weightChangeVector = AqwamTensorLibrary:multiply(kalmanGainVector, lossMatrix) -- 1 x n, 1 x n
	
	weightChangeVector = AqwamTensorLibrary:transpose(weightChangeVector)
	
	weightVector = AqwamTensorLibrary:add(weightVector, weightChangeVector)
	
	errorCovarianceMatrix = AqwamTensorLibrary:subtract(errorCovarianceMatrix, AqwamTensorLibrary:dotProduct(transposedKalmanGainVector, previousStateMatrix, errorCovarianceMatrix))

	if (forgetFactor ~= 1) then errorCovarianceMatrix = AqwamTensorLibrary:divide(errorCovarianceMatrix, forgetFactor) end
	
	self.ModelParameters = {weightVector, errorCovarianceMatrix}
	
	-- Returning this as a cost like other models.
	
	if (lossFunction == "L1") then
		
		lossMatrix = AqwamTensorLibrary:applyFunction(math.abs, lossMatrix)
		
	elseif (lossFunction == "L2") then
		
		lossMatrix = AqwamTensorLibrary:power(lossMatrix, 2)
		
	else
		
		error("Invalid loss function.")
		
	end
	
	local cost = AqwamTensorLibrary:sum(lossMatrix)
	
	cost = cost / numberOfData

	return {cost}

end

function RecursiveLeastSquaresFilterModel:predict(stateMatrix)

	local weightMatrix = self.ModelParameters[3]
	
	return AqwamTensorLibrary:dotProduct(stateMatrix, weightMatrix)
	
end

return RecursiveLeastSquaresFilterModel
