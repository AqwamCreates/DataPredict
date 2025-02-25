--[[

	--------------------------------------------------------------------

	Aqwam's Machine And Deep Learning Library (DataPredict)

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

local BaseModel = require("Model_BaseModel")

NormalLinearRegressionModel = {}

NormalLinearRegressionModel.__index = NormalLinearRegressionModel

setmetatable(NormalLinearRegressionModel, BaseModel)

local AqwamTensorLibrary = require("AqwamTensorLibrary")

function NormalLinearRegressionModel.new(parameterDictionary)
	
	local NewNormalLinearRegressionModel = BaseModel.new(parameterDictionary)

	setmetatable(NewNormalLinearRegressionModel, NormalLinearRegressionModel)
	
	NewNormalLinearRegressionModel:setName("NormalLinearRegression")
	
	return NewNormalLinearRegressionModel
	
end

function NormalLinearRegressionModel:train(featureMatrix, labelVector)
	
	local transposedFeatureMatrix = AqwamTensorLibrary:transpose(featureMatrix)
	
	local dotProductFeatureMatrix = AqwamTensorLibrary:dotProduct(featureMatrix, transposedFeatureMatrix)
	
	local inverseDotProduct = AqwamTensorLibrary:inverse(dotProductFeatureMatrix)
	
	if (inverseDotProduct == nil) then error("Could not find the model parameters!") end
	
	local dotProductFeatureMatrixAndLabelVector = AqwamTensorLibrary:dotProduct(transposedFeatureMatrix, labelVector)
	
	local ModelParameters = AqwamTensorLibrary:multiply(inverseDotProduct, dotProductFeatureMatrixAndLabelVector)
	
	self.ModelParameters = ModelParameters
	
end

function NormalLinearRegressionModel:predict(featureMatrix)
	
	return AqwamTensorLibrary:dotProduct(featureMatrix, self.ModelParameters)
	
end

return NormalLinearRegressionModel
