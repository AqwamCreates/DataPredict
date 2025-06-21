--[[

	--------------------------------------------------------------------

	Aqwam's Machine And Deep Learning Library (DataPredict)

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

local BaseInstance = require(script.Parent.Parent.Cores.BaseInstance)

local RandomNetworkDistillation = {}

RandomNetworkDistillation.__index = RandomNetworkDistillation

setmetatable(RandomNetworkDistillation, BaseInstance)

function RandomNetworkDistillation.new(parameterDictionary)
	
	parameterDictionary = parameterDictionary or {}
	
	local NewRandomNetworkDistillation = BaseInstance.new(parameterDictionary)
	
	setmetatable(NewRandomNetworkDistillation, RandomNetworkDistillation)
	
	NewRandomNetworkDistillation:setName("RandomNetworkDistillation")
	
	NewRandomNetworkDistillation:setClassName("RandomNetworkDistillation")
	
	NewRandomNetworkDistillation.Model = parameterDictionary.Model
	
	NewRandomNetworkDistillation.TargetModelParameters = parameterDictionary.TargetModelParameters
	
	NewRandomNetworkDistillation.PredictorModelParameters = parameterDictionary.PredictorModelParameters
	
	return NewRandomNetworkDistillation
	
end

function RandomNetworkDistillation:setModel(Model)
	
	self.Model = Model
	
end

function RandomNetworkDistillation:getModel(Model)
	
	return self.Model
	
end

function RandomNetworkDistillation:generate(featureMatrix)
	
	local Model = self.Model
	
	if (not Model) then error("No model!") end
	
	local TargetModelParameters = self.TargetModelParameters
	
	local PredictorModelParameters = self.PredictorModelParameters
	
	if (not TargetModelParameters) then
		
		Model:generateLayers()
		
		TargetModelParameters = Model:getModelParameters(true)
		
	end
	
	Model:setModelParameters(TargetModelParameters, true)
	
	local targetMatrix = Model:predict(featureMatrix, true)
	
	if (not PredictorModelParameters) then

		Model:generateLayers()

		PredictorModelParameters = Model:getModelParameters(true)

	end
	
	Model:setModelParameters(PredictorModelParameters, true)

	local predictorMatrix = Model:predict(featureMatrix, true)
	
	local errorMatrix = AqwamTensorLibrary:subtract(predictorMatrix, targetMatrix)
	
	local squaredErrorMatrix = AqwamTensorLibrary:power(errorMatrix, 2)
	
	local sumSquaredErrorMatrix = AqwamTensorLibrary:sum(squaredErrorMatrix, 2)
	
	local generatedMatrix = AqwamTensorLibrary:power(sumSquaredErrorMatrix, 0.5)

	Model:forwardPropagate(featureMatrix, true)

	Model:update(errorMatrix, true)
	
	self.TargetModelParameters = TargetModelParameters
	
	self.PredictorModelParameters = Model:getModelParameters(true)

	return generatedMatrix
	
end

function RandomNetworkDistillation:getTargetModelParameters(doNotDeepCopy)
	
	if (doNotDeepCopy) then
		
		return self.TargetModelParameters 
		
	else
		
		return self:deepCopyTable(self.TargetModelParameters)
		
	end
	
end

function RandomNetworkDistillation:getPredictorModelParameters(doNotDeepCopy)
	
	if (doNotDeepCopy) then

		return self.PredictorModelParameters 

	else

		return self:deepCopyTable(self.PredictorModelParameters)

	end
	
end

function RandomNetworkDistillation:setTargetModelParameters(TargetModelParameters, doNotDeepCopy)
	
	if (doNotDeepCopy) then

		self.TargetModelParameters = TargetModelParameters

	else

		self.TargetModelParameters = self:deepCopyTable(TargetModelParameters)

	end
	
end

function RandomNetworkDistillation:setPredictorModelParameters(PredictorModelParameters, doNotDeepCopy)
	
	if (doNotDeepCopy) then

		self.PredictorModelParameters = PredictorModelParameters

	else

		self.PredictorModelParameters = self:deepCopyTable(PredictorModelParameters)

	end

end

return RandomNetworkDistillation