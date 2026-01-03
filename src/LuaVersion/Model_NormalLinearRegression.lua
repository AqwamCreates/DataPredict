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

local BaseModel = require(script.Parent.BaseModel)

local NormalLinearRegressionModel = {}

NormalLinearRegressionModel.__index = NormalLinearRegressionModel

setmetatable(NormalLinearRegressionModel, BaseModel)

local defaultLambda = 0

local defaultWeightDecay = 1

function NormalLinearRegressionModel.new(parameterDictionary)

	local NewNormalLinearRegressionModel = BaseModel.new(parameterDictionary)

	setmetatable(NewNormalLinearRegressionModel, NormalLinearRegressionModel)

	NewNormalLinearRegressionModel:setName("NormalLinearRegression")

	NewNormalLinearRegressionModel.lambda = parameterDictionary.lambda or defaultLambda
	
	NewNormalLinearRegressionModel.weightDecay = parameterDictionary.weightDecay or defaultWeightDecay

	return NewNormalLinearRegressionModel

end

function NormalLinearRegressionModel:train(featureMatrix, labelVector)

	if (#featureMatrix ~= #labelVector) then error("The feature matrix and the label vector does not contain the same number of rows.") end

	local lambda = self.lambda
	
	local weightDecay = self.weightDecay
	
	local ModelParameters = self.ModelParameters or {}
	
	local oldDotProductFeatureMatrix = ModelParameters[2]
	
	local oldDotProductFeatureMatrixAndLabelVector = ModelParameters[3]

	local transposedFeatureMatrix = AqwamTensorLibrary:transpose(featureMatrix)

	local dotProductFeatureMatrix = AqwamTensorLibrary:dotProduct(transposedFeatureMatrix, featureMatrix)

	if (lambda ~= 0) then

		local numberOfFeatures = #featureMatrix[1]

		local lambdaIdentityMatrix = AqwamTensorLibrary:createIdentityTensor({numberOfFeatures, numberOfFeatures}, lambda)

		dotProductFeatureMatrix = AqwamTensorLibrary:add(dotProductFeatureMatrix, lambdaIdentityMatrix)

	end
	
	if (oldDotProductFeatureMatrix) then
		
		oldDotProductFeatureMatrix = AqwamTensorLibrary:multiply(weightDecay, oldDotProductFeatureMatrix)

		dotProductFeatureMatrix = AqwamTensorLibrary:add(dotProductFeatureMatrix, oldDotProductFeatureMatrix)

	end

	local inverseDotProduct = AqwamTensorLibrary:inverse(dotProductFeatureMatrix)

	if (not inverseDotProduct) then error("Could not find the model parameters.") end
	
	local dotProductFeatureMatrixAndLabelVector = AqwamTensorLibrary:dotProduct(transposedFeatureMatrix, labelVector)
	
	if (oldDotProductFeatureMatrixAndLabelVector) then
		
		oldDotProductFeatureMatrixAndLabelVector = AqwamTensorLibrary:multiply(weightDecay, oldDotProductFeatureMatrixAndLabelVector)
		
		dotProductFeatureMatrixAndLabelVector = AqwamTensorLibrary:add(dotProductFeatureMatrixAndLabelVector, oldDotProductFeatureMatrixAndLabelVector)
		
	end

	local weightMatrix = AqwamTensorLibrary:multiply(inverseDotProduct, dotProductFeatureMatrixAndLabelVector)

	self.ModelParameters = {weightMatrix, dotProductFeatureMatrix, dotProductFeatureMatrixAndLabelVector}

end

function NormalLinearRegressionModel:predict(featureMatrix)

	local ModelParameters = self.ModelParameters or {}
	
	local weightMatrix = ModelParameters[1]

	if (not weightMatrix) then

		weightMatrix = self:initializeMatrixBasedOnMode({#featureMatrix[1], 1})

		self.ModelParameters = {weightMatrix}

	end

	return AqwamTensorLibrary:dotProduct(featureMatrix, weightMatrix)

end

return NormalLinearRegressionModel
