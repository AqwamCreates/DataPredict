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

local BaseRegularizer = require("Regularizer_BaseRegularizer")

ElasticNet = {}

ElasticNet.__index = ElasticNet

setmetatable(ElasticNet, BaseRegularizer)

function ElasticNet.new(parameterDictionary)
	
	local NewElasticNet = BaseRegularizer.new(parameterDictionary)
	
	setmetatable(NewElasticNet, ElasticNet)
	
	NewElasticNet:setName("ElasticNet")
	
	NewElasticNet:setCalculateCostFunction(function(ModelParameters)
		
		local lambda = NewElasticNet.lambda
		
		ModelParameters = NewElasticNet:adjustModelParameters(ModelParameters)
		
		local SquaredModelParameters = AqwamTensorLibrary:power(ModelParameters, 2)

		local sumSquaredModelParameters = AqwamTensorLibrary:sum(SquaredModelParameters)

		local absoluteModelParameters = AqwamTensorLibrary:applyFunction(math.abs, ModelParameters)

		local sumAbsoluteModelParameters = AqwamTensorLibrary:sum(absoluteModelParameters)

		local regularizationValuePart1 = lambda * sumSquaredModelParameters

		local regularizationValuePart2 = lambda * sumAbsoluteModelParameters

		return regularizationValuePart1 + regularizationValuePart2
		
	end)
	
	NewElasticNet:setCalculateFunction(function(ModelParameters)
		
		ModelParameters = NewElasticNet:adjustModelParameters(ModelParameters)
		
		local lambda = NewElasticNet.lambda
		
		local signMatrix = AqwamTensorLibrary:applyFunction(math.sign, ModelParameters)

		local regularizationMatrixPart1 = AqwamTensorLibrary:multiply(lambda, signMatrix)

		local regularizationMatrixPart2 = AqwamTensorLibrary:multiply(2, lambda, ModelParameters)

		return AqwamTensorLibrary:add(regularizationMatrixPart1, regularizationMatrixPart2)
		
	end)
	
	return NewElasticNet
	
end

return ElasticNet
