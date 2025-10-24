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

local BaseIntstance = require("Core_BaseInstance")

local NormalModelModifier = {}

NormalModelModifier.__index = NormalModelModifier

setmetatable(NormalModelModifier, BaseIntstance)

local defaultNoiseValue = 1e-16

function NormalModelModifier.new(parameterDictionary)
	
	parameterDictionary = parameterDictionary or {}
	
	local NewNormalModelModifier = BaseIntstance.new(parameterDictionary)
	
	setmetatable(NewNormalModelModifier, NormalModelModifier)
	
	NewNormalModelModifier:setName("NormalModelModifier")
	
	NewNormalModelModifier:setClassName("NormalModelModifier")
	
	NewNormalModelModifier.Model = parameterDictionary.Model
	
	NewNormalModelModifier.noiseValue = parameterDictionary.noiseValue or defaultNoiseValue
	
	return NewNormalModelModifier
	
end

function NormalModelModifier:train(featureMatrix, labelMatrix)
	
	local noiseValue = self.noiseValue
	
	local numberOfData = #featureMatrix
	
	local numberOfFeatures = #featureMatrix[1]
	
	local smallestDimensionSize = math.min(numberOfData, numberOfFeatures)
	
	local newFeatureMatrix = {}
	
	local newLabelMatrix
	
	if (labelMatrix) then
		
		newLabelMatrix = {}
		
		for dataIndex, unwrappedFeatureVector in ipairs(featureMatrix) do
			
			if (dataIndex > smallestDimensionSize) then break end
			
			newFeatureMatrix[dataIndex] = unwrappedFeatureVector
			
			newLabelMatrix[dataIndex] = labelMatrix[dataIndex]
			
		end
		
	else
		
		for dataIndex, unwrappedFeatureVector in ipairs(featureMatrix) do

			if (dataIndex > smallestDimensionSize) then break end

			newFeatureMatrix[dataIndex] = unwrappedFeatureVector

		end
		
	end
	
	-- This step helps to break collinearity in our feature matrix.
	
	if (noiseValue ~= 0) then
		
		local noiseMatrix = AqwamTensorLibrary:createRandomUniformTensor({smallestDimensionSize, smallestDimensionSize}, -noiseValue, noiseValue)

		newFeatureMatrix = AqwamTensorLibrary:add(newFeatureMatrix, noiseMatrix)

	end
	
	return self.Model:train(newFeatureMatrix, newLabelMatrix)
	
end

function NormalModelModifier:update(...)

	return self.Model:update(...)

end

function NormalModelModifier:predict(...)
	
	return self.Model:predict(...)
	
end

function NormalModelModifier:setModel(Model)
	
	self.Model = Model
	
end

function NormalModelModifier:getModel()

	return self.Model

end

function NormalModelModifier:getModelParameters(...)

	return self.Model:getModelParameters(...)

end

function NormalModelModifier:setModelParameters(...)

	self.Model:setModelParameters(...)

end

return NormalModelModifier
