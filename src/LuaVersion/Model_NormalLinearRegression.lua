--[[

	--------------------------------------------------------------------

	Author: Aqwam Harish Aiman
	
	YouTube: https://www.youtube.com/channel/UCUrwoxv5dufEmbGsxyEUPZw
	
	LinkedIn: https://www.linkedin.com/in/aqwam-harish-aiman/
	
	--------------------------------------------------------------------
	
	DO NOT SELL, RENT, DISTRIBUTE THIS LIBRARY
	
	DO NOT SELL, RENT, DISTRIBUTE MODIFIED VERSION OF THIS LIBRARY
	
	DO NOT CLAIM OWNERSHIP OF THIS LIBRARY
	
	GIVE CREDIT AND SOURCE WHEN USING THIS LIBRARY IF YOUR USAGE FALLS UNDER ONE OF THESE CATEGORIES:
	
		- USED AS A VIDEO OR ARTICLE CONTENT
		- USED AS RESEARCH AND EDUCATION CONTENT
	
	--------------------------------------------------------------------
		
	By using this library, you agree to comply with our Terms and Conditions in the link below:
	
	https://github.com/AqwamCreates/DataPredict/blob/main/docs/TermsAndConditions.md
	
	--------------------------------------------------------------------

--]]

local BaseModel = require("Model_BaseModel")

NormalLinearRegressionModel = {}

NormalLinearRegressionModel.__index = NormalLinearRegressionModel

setmetatable(NormalLinearRegressionModel, BaseModel)

local AqwamMatrixLibrary = require("AqwamMatrixLibrary")

function NormalLinearRegressionModel.new()
	
	local NewNormalLinearRegressionModel = BaseModel.new()

	setmetatable(NewNormalLinearRegressionModel, NormalLinearRegressionModel)
	
	return NewNormalLinearRegressionModel
	
end

function NormalLinearRegressionModel:train(featureMatrix, labelVector)
	
	local transposedFeatureMatrix = AqwamMatrixLibrary:transpose(featureMatrix)
	
	local dotProductFeatureMatrix = AqwamMatrixLibrary:dotProduct(featureMatrix, transposedFeatureMatrix)
	
	local inverseDotProduct = AqwamMatrixLibrary:inverse(dotProductFeatureMatrix)
	
	if (inverseDotProduct == nil) then error("Could not find the model parameters!") end
	
	local dotProductFeatureMatrixAndLabelVector = AqwamMatrixLibrary:dotProduct(transposedFeatureMatrix, labelVector)
	
	local ModelParameters = AqwamMatrixLibrary:multiply(inverseDotProduct, dotProductFeatureMatrixAndLabelVector)
	
	self.ModelParameters = ModelParameters
	
end

function NormalLinearRegressionModel:predict(featureMatrix)
	
	return AqwamMatrixLibrary:dotProduct(featureMatrix, self.ModelParameters)
	
end

return NormalLinearRegressionModel
