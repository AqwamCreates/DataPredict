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
	
	local currentNumberOfData = #featureMatrix

	if (#featureMatrix ~= #labelVector) then error("The feature matrix and the label vector does not contain the same number of rows.") end
	
	local ModelParameters = self.ModelParameters or {}
	
	local oldMeanBiasValue = ModelParameters[1]
	
	local oldNumberOfData = ModelParameters[2]

	local sumFeatureVector = AqwamTensorLibrary:sum(featureMatrix, 2)

	local biasVector = AqwamTensorLibrary:subtract(labelVector, sumFeatureVector)

	local sumCurrentBiasValue = AqwamTensorLibrary:sum(biasVector)
	
	if (oldMeanBiasValue) and (oldNumberOfData) then
		
		local sumOldBiasValue = oldMeanBiasValue * oldNumberOfData
		
		sumCurrentBiasValue = sumCurrentBiasValue + sumOldBiasValue
		
		currentNumberOfData = currentNumberOfData + oldNumberOfData
		
	end
	
	local meanBiasValue = sumCurrentBiasValue / currentNumberOfData

	self.ModelParameters = {meanBiasValue, currentNumberOfData}

end

function UnitWeightedRegressionModel:predict(featureMatrix)
	
	local ModelParameters = self.ModelParameters or {}

	local meanBiasValue = ModelParameters[1] or math.huge

	local sumFeatureVector = AqwamTensorLibrary:sum(featureMatrix, 2)

	return AqwamTensorLibrary:add(sumFeatureVector, meanBiasValue)

end

return UnitWeightedRegressionModel
