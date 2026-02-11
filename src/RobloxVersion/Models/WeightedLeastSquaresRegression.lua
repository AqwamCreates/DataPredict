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

local WeightedLeastSquaresRegressionModel = {}

WeightedLeastSquaresRegressionModel.__index = WeightedLeastSquaresRegressionModel

setmetatable(WeightedLeastSquaresRegressionModel, BaseModel)

function WeightedLeastSquaresRegressionModel.new(parameterDictionary)
	
	parameterDictionary = parameterDictionary or {}

	local NewWeightedLeastSquaresRegressionModel = BaseModel.new(parameterDictionary)

	setmetatable(NewWeightedLeastSquaresRegressionModel, WeightedLeastSquaresRegressionModel)
	
	NewWeightedLeastSquaresRegressionModel:setName("WeightedLeastSquaresRegression")
	
	NewWeightedLeastSquaresRegressionModel.weightMatrix = parameterDictionary.weightMatrix

	return NewWeightedLeastSquaresRegressionModel

end

function WeightedLeastSquaresRegressionModel:train(featureMatrix, labelVector)
	
	local numberOfdata = #featureMatrix

	if (numberOfdata ~= #labelVector) then error("The feature matrix and the label vector does not contain the same number of rows.") end
	
	local weightMatrix = self.weightMatrix

	if (weightMatrix) then

		if (#weightMatrix ~= numberOfdata) then error("The number of data does not match the number of rows in the weight matrix.") end

		if (#weightMatrix[1] ~= numberOfdata) then error("The number of data does not match the number of columns in the weight matrix.") end

	else

		weightMatrix = AqwamTensorLibrary:createIdentityTensor({numberOfdata, numberOfdata}, 1)

	end
	
	local numberOfFeatures = #featureMatrix[1]
	
	local betaVector = self.ModelParameters

	if (betaVector) then

		if (numberOfFeatures ~= #betaVector) then error("The number of features are not the same as the model parameters.") end

	else

		betaVector = self:initializeMatrixBasedOnMode({numberOfFeatures, 1})

	end
	
	local tansposedFeatureMatrix = AqwamTensorLibrary:transpose(featureMatrix)
	
	local responseVector = AqwamTensorLibrary:dotProduct(featureMatrix, betaVector)
	
	local errorVector = AqwamTensorLibrary:subtract(labelVector, responseVector)
	
	local betaChangeVector = AqwamTensorLibrary:dotProduct(tansposedFeatureMatrix, weightMatrix, featureMatrix)
	
	betaChangeVector = AqwamTensorLibrary:inverse(betaChangeVector)
	
	betaChangeVector = AqwamTensorLibrary:dotProduct(betaChangeVector, tansposedFeatureMatrix, weightMatrix, errorVector)
	
	betaVector = AqwamTensorLibrary:add(betaVector, betaChangeVector)
		
	self.ModelParameters = betaVector

end

function WeightedLeastSquaresRegressionModel:predict(featureMatrix)
	
	local betaVector = self.ModelParameters
	
	if (not betaVector) then
		
		local numberOfFeatures = #featureMatrix[1]
		
		betaVector = self:initializeMatrixBasedOnMode({numberOfFeatures, 1})
		
		self.ModelParameters = betaVector
		
	end

	local predictedVector = AqwamTensorLibrary:dotProduct(featureMatrix, betaVector)

	return predictedVector

end

return WeightedLeastSquaresRegressionModel
