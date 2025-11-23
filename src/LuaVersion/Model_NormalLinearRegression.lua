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

local AqwamTensorLibrary = require("AqwamTensorLibrary")

local BaseModel = require("Model_BaseModel")

local NormalLinearRegressionModel = {}

NormalLinearRegressionModel.__index = NormalLinearRegressionModel

setmetatable(NormalLinearRegressionModel, BaseModel)

local defaultLambda = 0

function NormalLinearRegressionModel.new(parameterDictionary)

	local NewNormalLinearRegressionModel = BaseModel.new(parameterDictionary)

	setmetatable(NewNormalLinearRegressionModel, NormalLinearRegressionModel)

	NewNormalLinearRegressionModel:setName("NormalLinearRegression")

	NewNormalLinearRegressionModel.lambda = parameterDictionary.lambda or defaultLambda

	return NewNormalLinearRegressionModel

end

function NormalLinearRegressionModel:train(featureMatrix, labelVector)

	if (#featureMatrix ~= #labelVector) then error("The feature matrix and the label vector does not contain the same number of rows.") end

	local lambda = self.lambda

	local transposedFeatureMatrix = AqwamTensorLibrary:transpose(featureMatrix)

	local dotProductFeatureMatrix = AqwamTensorLibrary:dotProduct(featureMatrix, transposedFeatureMatrix)

	if (lambda ~= 0) then

		local numberOfFeatures = #featureMatrix[1]

		local lambdaIdentityMatrix = AqwamTensorLibrary:createIdentityTensor({numberOfFeatures, numberOfFeatures})

		lambdaIdentityMatrix = AqwamTensorLibrary:multiply(lambdaIdentityMatrix, lambda)

		dotProductFeatureMatrix = AqwamTensorLibrary:add(dotProductFeatureMatrix, lambdaIdentityMatrix)

	end

	local inverseDotProduct = AqwamTensorLibrary:inverse(dotProductFeatureMatrix)

	if (not inverseDotProduct) then error("Could not find the model parameters.") end

	local dotProductFeatureMatrixAndLabelVector = AqwamTensorLibrary:dotProduct(transposedFeatureMatrix, labelVector)

	local ModelParameters = AqwamTensorLibrary:multiply(inverseDotProduct, dotProductFeatureMatrixAndLabelVector)

	self.ModelParameters = ModelParameters

end

function NormalLinearRegressionModel:predict(featureMatrix)

	local ModelParameters = self.ModelParameters

	if (not ModelParameters) then

		ModelParameters = self:initializeMatrixBasedOnMode({#featureMatrix[1], 1})

		self.ModelParameters = ModelParameters

	end

	return AqwamTensorLibrary:dotProduct(featureMatrix, ModelParameters)

end

return NormalLinearRegressionModel
