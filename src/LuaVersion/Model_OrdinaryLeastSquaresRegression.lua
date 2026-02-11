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

local OrdinaryLeastSquaresRegressionModel = {}

OrdinaryLeastSquaresRegressionModel.__index = OrdinaryLeastSquaresRegressionModel

setmetatable(OrdinaryLeastSquaresRegressionModel, BaseModel)

local defaultModelParametersInitializationMode = "Zero"

function OrdinaryLeastSquaresRegressionModel.new(parameterDictionary)
	
	parameterDictionary = parameterDictionary or {}
	
	parameterDictionary.modelParametersInitializationMode = parameterDictionary.modelParametersInitializationMode or defaultModelParametersInitializationMode

	local NewOrdinaryLeastSquaresRegressionModel = BaseModel.new(parameterDictionary)

	setmetatable(NewOrdinaryLeastSquaresRegressionModel, OrdinaryLeastSquaresRegressionModel)

	NewOrdinaryLeastSquaresRegressionModel:setName("OrdinaryLeastSquaresRegression")

	return NewOrdinaryLeastSquaresRegressionModel

end

function OrdinaryLeastSquaresRegressionModel:train(featureMatrix, labelVector)

	if (#featureMatrix ~= #labelVector) then error("The feature matrix and the label vector does not contain the same number of rows.") end
	
	local numberOfFeatures = #featureMatrix[1]
	
	local betaVector = self.ModelParameters

	if (betaVector) then

		if (numberOfFeatures ~= #betaVector) then error("The number of features are not the same as the model parameters.") end

	else

		betaVector = self:initializeMatrixBasedOnMode({numberOfFeatures, 1})

	end

	local transposedFeatureMatrix = AqwamTensorLibrary:transpose(featureMatrix)

	local dotProductFeatureMatrix = AqwamTensorLibrary:dotProduct(transposedFeatureMatrix, featureMatrix)

	local inverseDotProductMatrix = AqwamTensorLibrary:inverse(dotProductFeatureMatrix)

	if (not inverseDotProductMatrix) then error("Could not find the model parameters.") end
	
	local responseVector = AqwamTensorLibrary:dotProduct(featureMatrix, betaVector)
	
	local errorVector = AqwamTensorLibrary:subtract(labelVector, responseVector)
	
	local betaChangeVector = AqwamTensorLibrary:dotProduct(inverseDotProductMatrix, transposedFeatureMatrix, errorVector)

	betaVector = AqwamTensorLibrary:add(betaVector, betaChangeVector)

	self.ModelParameters = betaVector

end

function OrdinaryLeastSquaresRegressionModel:predict(featureMatrix)

	local betaVector = self.ModelParameters

	if (not betaVector) then

		betaVector = self:initializeMatrixBasedOnMode({#featureMatrix[1], 1})

		self.ModelParameters = betaVector

	end

	return AqwamTensorLibrary:dotProduct(featureMatrix, betaVector)

end

return OrdinaryLeastSquaresRegressionModel
