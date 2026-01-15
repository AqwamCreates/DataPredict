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

local NormalEquationLinearRegressionModel = {}

NormalEquationLinearRegressionModel.__index = NormalEquationLinearRegressionModel

setmetatable(NormalEquationLinearRegressionModel, BaseModel)

local defaultLambda = 0

local defaultWeightDecay = 1

function NormalEquationLinearRegressionModel.new(parameterDictionary)

	local NewNormalEquationLinearRegressionModel = BaseModel.new(parameterDictionary)

	setmetatable(NewNormalEquationLinearRegressionModel, NormalEquationLinearRegressionModel)

	NewNormalEquationLinearRegressionModel:setName("NormalEquationLinearRegression")

	NewNormalEquationLinearRegressionModel.lambda = parameterDictionary.lambda or defaultLambda
	
	NewNormalEquationLinearRegressionModel.weightDecay = parameterDictionary.weightDecay or defaultWeightDecay

	return NewNormalEquationLinearRegressionModel

end

function NormalEquationLinearRegressionModel:train(featureMatrix, labelVector)

	if (#featureMatrix ~= #labelVector) then error("The feature matrix and the label vector does not contain the same number of rows.") end

	local lambda = self.lambda
	
	local weightDecay = self.weightDecay
	
	local ModelParameters = self.ModelParameters or {}
	
	local oldDotProductFeatureMatrix = ModelParameters[2]
	
	local oldDotProductFeatureMatrixAndLabelVector = ModelParameters[3]

	local transposedFeatureMatrix = AqwamTensorLibrary:transpose(featureMatrix)

	local newDotProductFeatureMatrix = AqwamTensorLibrary:dotProduct(transposedFeatureMatrix, featureMatrix)

	if (lambda ~= 0) then

		local numberOfFeatures = #featureMatrix[1]

		local lambdaIdentityMatrix = AqwamTensorLibrary:createIdentityTensor({numberOfFeatures, numberOfFeatures}, lambda)

		newDotProductFeatureMatrix = AqwamTensorLibrary:add(newDotProductFeatureMatrix, lambdaIdentityMatrix)

	end
	
	if (oldDotProductFeatureMatrix) then
		
		oldDotProductFeatureMatrix = AqwamTensorLibrary:multiply(weightDecay, oldDotProductFeatureMatrix)

		newDotProductFeatureMatrix = AqwamTensorLibrary:add(newDotProductFeatureMatrix, oldDotProductFeatureMatrix)

	end

	local newInverseDotProduct = AqwamTensorLibrary:inverse(newDotProductFeatureMatrix)

	if (not newInverseDotProduct) then error("Could not find the model parameters.") end
	
	local newDotProductFeatureMatrixAndLabelVector = AqwamTensorLibrary:dotProduct(transposedFeatureMatrix, labelVector)
	
	if (oldDotProductFeatureMatrixAndLabelVector) then
		
		oldDotProductFeatureMatrixAndLabelVector = AqwamTensorLibrary:multiply(weightDecay, oldDotProductFeatureMatrixAndLabelVector)
		
		newDotProductFeatureMatrixAndLabelVector = AqwamTensorLibrary:add(newDotProductFeatureMatrixAndLabelVector, oldDotProductFeatureMatrixAndLabelVector)
		
	end

	local newWeightVector = AqwamTensorLibrary:dotProduct(newInverseDotProduct, newDotProductFeatureMatrixAndLabelVector)

	self.ModelParameters = {newWeightVector, newDotProductFeatureMatrix, newDotProductFeatureMatrixAndLabelVector}

end

function NormalEquationLinearRegressionModel:predict(featureMatrix)

	local ModelParameters = self.ModelParameters or {}
	
	local weightVector = ModelParameters[1]

	if (not weightVector) then

		weightVector = self:initializeMatrixBasedOnMode({#featureMatrix[1], 1})

		self.ModelParameters = {weightVector}

	end

	return AqwamTensorLibrary:dotProduct(featureMatrix, weightVector)

end

return NormalEquationLinearRegressionModel
