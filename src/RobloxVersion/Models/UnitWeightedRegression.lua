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

local UnitWeightedRegressionModel = {}

UnitWeightedRegressionModel.__index = UnitWeightedRegressionModel

setmetatable(UnitWeightedRegressionModel, BaseModel)

function UnitWeightedRegressionModel.new(parameterDictionary)

	local NewUnitWeightedRegressionModel = BaseModel.new(parameterDictionary)

	setmetatable(NewUnitWeightedRegressionModel, UnitWeightedRegressionModel)

	NewUnitWeightedRegressionModel:setName("UnitWeightedRegression")

	return NewUnitWeightedRegressionModel

end

function UnitWeightedRegressionModel:train(featureMatrix, labelVector)

	if (#featureMatrix ~= #labelVector) then error("The feature matrix and the label vector does not contain the same number of rows.") end

	local sumFeatureVector = AqwamTensorLibrary:sum(featureMatrix, 2)

	local biasVector = AqwamTensorLibrary:subtract(labelVector, sumFeatureVector)

	local biasValue = AqwamTensorLibrary:mean(biasVector)

	self.ModelParameters = biasValue

end

function UnitWeightedRegressionModel:predict(featureMatrix)

	local biasValue = self.ModelParameters or math.huge

	local sumFeatureVector = AqwamTensorLibrary:sum(featureMatrix, 2)

	return AqwamTensorLibrary:add(sumFeatureVector, biasValue)

end

return UnitWeightedRegressionModel
