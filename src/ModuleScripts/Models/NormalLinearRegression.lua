local BaseModel = require(script.Parent.BaseModel)

NormalLinearRegressionModel = {}

NormalLinearRegressionModel.__index = NormalLinearRegressionModel

setmetatable(NormalLinearRegressionModel, BaseModel)

local AqwamMatrixLibrary = require(script.Parent.Parent.AqwamRobloxMatrixLibraryLinker.Value)

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
