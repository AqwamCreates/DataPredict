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
	
	local oldWeightVector = self.ModelParameters or self:initializeMatrixBasedOnMode({#featureMatrix[1], 1})

	local transposedFeatureMatrix = AqwamTensorLibrary:transpose(featureMatrix)

	local dotProductFeatureMatrix = AqwamTensorLibrary:dotProduct(transposedFeatureMatrix, featureMatrix)

	local inverseDotProductMatrix = AqwamTensorLibrary:inverse(dotProductFeatureMatrix)

	if (not inverseDotProductMatrix) then error("Could not find the model parameters.") end
	
	local responseVector = AqwamTensorLibrary:dotProduct(featureMatrix, oldWeightVector)
	
	local errorVector = AqwamTensorLibrary:subtract(labelVector, responseVector)
	
	local weightChangeVector = AqwamTensorLibrary:dotProduct(inverseDotProductMatrix, transposedFeatureMatrix, errorVector)

	local newWeightVector = AqwamTensorLibrary:add(oldWeightVector, weightChangeVector)

	self.ModelParameters = newWeightVector

end

function OrdinaryLeastSquaresRegressionModel:predict(featureMatrix)

	local weightVector = self.ModelParameters

	if (not weightVector) then

		weightVector = self:initializeMatrixBasedOnMode({#featureMatrix[1], 1})

		self.ModelParameters = weightVector

	end

	return AqwamTensorLibrary:dotProduct(featureMatrix, weightVector)

end

return OrdinaryLeastSquaresRegressionModel
