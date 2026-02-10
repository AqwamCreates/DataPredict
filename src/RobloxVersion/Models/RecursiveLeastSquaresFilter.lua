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

local lossFunctionList = {
	
	["L1"] = math.abs,
	
	["L2"] = function (value) return math.pow(value, 2) end
	
}

function RecursiveLeastSquaresFilterModel.new(parameterDictionary)
	
	parameterDictionary = parameterDictionary or {}

	local NewRecursiveLeastSquaresFilterModel = BaseModel.new(parameterDictionary)

	setmetatable(NewRecursiveLeastSquaresFilterModel, RecursiveLeastSquaresFilterModel)

	NewRecursiveLeastSquaresFilterModel:setName("RecursiveLeastSquaresFilter")
	
	NewRecursiveLeastSquaresFilterModel.lossFunction = parameterDictionary.lossFunction or defaultLossFunction
	
	NewRecursiveLeastSquaresFilterModel.forgetFactor = parameterDictionary.forgetFactor or defaultForgetFactor

	return NewRecursiveLeastSquaresFilterModel
	
end

function RecursiveLeastSquaresFilterModel:train(featureMatrix, labelVector)

	local numberOfData = #featureMatrix

	if (#featureMatrix ~= #labelVector) then error("The feature matrix and the label vector does not contain the same number of rows.") end
	
	local numberOfFeatures = #featureMatrix[1]
	
	local lossFunction = self.lossFunction
	
	local lossFunctionToApply = lossFunctionList[lossFunction]

	if (not lossFunctionToApply) then error("Invalid loss function.") end
	
	local forgetFactor = self.forgetFactor
	
	local ModelParameters = self.ModelParameters or {}
	
	local weightVector = ModelParameters[1] or self:initializeMatrixBasedOnMode({numberOfFeatures, 1})
	
	if (numberOfFeatures ~= #weightVector) then error("The number of features are not the same as the model parameters.") end
	
	local errorCovarianceMatrix = ModelParameters[2] or AqwamTensorLibrary:createIdentityTensor({numberOfFeatures, numberOfFeatures})
	
	local featureVector
	
	local predictedValue
	
	local lossValue
	
	local kalmanGainVectorNumerator
	
	local transposedFeatureVector
	
	local kalmanGainVectorDenominator
	
	local kalmanGainVector
	
	local transposedKalmanGainVector
	
	local weightChangeVector
	
	local cost = 0
	
	for dataIndex, unwrappedFeatureVector in ipairs(featureMatrix) do
		
		featureVector = {unwrappedFeatureVector}
		
		predictedValue = AqwamTensorLibrary:dotProduct(featureVector, weightVector)[1][1]
		
		lossValue = predictedValue - labelVector[dataIndex][1]
		
		kalmanGainVectorNumerator = AqwamTensorLibrary:dotProduct(featureVector, errorCovarianceMatrix) -- 1 x n
		
		transposedFeatureVector = AqwamTensorLibrary:transpose(featureVector) -- n x 1
		
		kalmanGainVectorDenominator = AqwamTensorLibrary:dotProduct(featureVector, errorCovarianceMatrix, transposedFeatureVector) -- 1 x 1
		
		kalmanGainVectorDenominator = AqwamTensorLibrary:add(forgetFactor, kalmanGainVectorDenominator)
		
		kalmanGainVector = AqwamTensorLibrary:divide(kalmanGainVectorNumerator, kalmanGainVectorDenominator) -- 1 x n
		
		transposedKalmanGainVector = AqwamTensorLibrary:transpose(kalmanGainVector)
		
		weightChangeVector = AqwamTensorLibrary:multiply(kalmanGainVector, lossValue) -- 1 x n
		
		weightChangeVector = AqwamTensorLibrary:transpose(weightChangeVector)

		weightVector = AqwamTensorLibrary:add(weightVector, weightChangeVector)
		
		errorCovarianceMatrix = AqwamTensorLibrary:subtract(errorCovarianceMatrix, AqwamTensorLibrary:dotProduct(transposedKalmanGainVector, featureVector, errorCovarianceMatrix))

		if (forgetFactor ~= 1) then errorCovarianceMatrix = AqwamTensorLibrary:divide(errorCovarianceMatrix, forgetFactor) end
		
		cost = cost + lossFunctionToApply(lossValue)
		
	end
	
	self.ModelParameters = {weightVector, errorCovarianceMatrix}
	
	cost = cost / numberOfData

	return {cost}

end

function RecursiveLeastSquaresFilterModel:predict(stateMatrix)

	local weightMatrix = self.ModelParameters[1]
	
	return AqwamTensorLibrary:dotProduct(stateMatrix, weightMatrix)
	
end

return RecursiveLeastSquaresFilterModel
