--[[

	--------------------------------------------------------------------

	Aqwam's Machine And Deep Learning Library (DataPredict)

	Author: Aqwam Harish Aiman
	
	YouTube: https://www.youtube.com/channel/UCUrwoxv5dufEmbGsxyEUPZw
	
	LinkedIn: https://www.linkedin.com/in/aqwam-harish-aiman/
	
	--------------------------------------------------------------------
		
	By using this library, you agree to comply with our Terms and Conditions in the link below:
	
	https://github.com/AqwamCreates/DataPredict/blob/main/docs/TermsAndConditions.md
	
	--------------------------------------------------------------------

--]]

local AqwamMatrixLibrary = require("AqwamMatrixLibrary")

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

local AqwamMatrixLibrary = require(script.Parent.Parent.AqwamMatrixLibraryLinker.Value)

local RandomNetworkDistillation = {}

RandomNetworkDistillation.__index = RandomNetworkDistillation

local function deepCopyTable(original, copies)

	copies = copies or {}

	local originalType = type(original)

	local copy

	if (originalType == 'table') then

		if copies[original] then

			copy = copies[original]

		else

			copy = {}

			copies[original] = copy

			for originalKey, originalValue in next, original, nil do

				copy[deepCopyTable(originalKey, copies)] = deepCopyTable(originalValue, copies)

			end

			setmetatable(copy, deepCopyTable(getmetatable(original), copies))

		end

	else

		copy = original

	end

	return copy

end

function RandomNetworkDistillation.new()
	
	local NewRandomNetworkDistillation = {}
	
	setmetatable(NewRandomNetworkDistillation, RandomNetworkDistillation)
	
	NewRandomNetworkDistillation.Model = nil
	
	NewRandomNetworkDistillation.TargetModelParameters = nil
	
	NewRandomNetworkDistillation.PredictorModelParameters = nil
	
	return NewRandomNetworkDistillation
	
end

function RandomNetworkDistillation:setModel(Model)
	
	self.Model = Model
	
end

function RandomNetworkDistillation:getModel(Model)
	
	return self.Model
	
end

function RandomNetworkDistillation:generateModelParameters()
	
	local Model = self.Model
	
	if (not self.TargetModelParameters) then
		
		Model:generateLayers()
		
		self.TargetModelParameters = Model:getModelParameters(true)
		
	end
	
	if (not self.PredictorModelParameters) then

		Model:generateLayers()

		self.PredictorModelParameters = Model:getModelParameters(true)

	end
	
end

function RandomNetworkDistillation:generate(featureVector)
	
	local Model = self.Model
	
	if (not Model) then error("No model!") end
	
	local PredictorModelParameters = self.PredictorModelParameters
	
	local TargetModelParameters = self.TargetModelParameters
	
	if (not TargetModelParameters) then
		
		Model:generateLayers()
		
		TargetModelParameters = Model:getModelParameters(true)
		
	end
	
	Model:setModelParameters(TargetModelParameters, true)
	
	local targetVector = Model:predict(featureVector, true)
	
	if (not PredictorModelParameters) then

		Model:generateLayers()

		PredictorModelParameters = Model:getModelParameters(true)

	end
	
	Model:setModelParameters(PredictorModelParameters, true)

	local predictorVector = Model:predict(featureVector, true)
	
	local errorVector = AqwamMatrixLibrary:subtract(predictorVector, targetVector)
	
	local squaredErrorVector = AqwamMatrixLibrary:power(errorVector, 2)
	
	local sumErrorVector = AqwamMatrixLibrary:horizontalSum(squaredErrorVector)
	
	local rewardVector = AqwamMatrixLibrary:power(sumErrorVector, 0.5)

	Model:forwardPropagate(featureVector, true)
	Model:backPropagate(errorVector, true)
	
	self.PredictorModelParameters = PredictorModelParameters

	self.TargetModelParameters = TargetModelParameters
	
	return rewardVector
	
end

function RandomNetworkDistillation:getTargetModelParameters(doNotDeepCopy)
	
	if (doNotDeepCopy) then
		
		return self.TargetModelParameters 
		
	else
		
		return deepCopyTable(self.TargetModelParameters)
		
	end
	
end

function RandomNetworkDistillation:getPredictorModelParameters(doNotDeepCopy)
	
	if (doNotDeepCopy) then

		return self.PredictorModelParameters 

	else

		return deepCopyTable(self.PredictorModelParameters)

	end
	
end

function RandomNetworkDistillation:setTargetModelParameters(TargetModelParameters, doNotDeepCopy)
	
	if (doNotDeepCopy) then

		self.TargetModelParameters = TargetModelParameters

	else

		self.TargetModelParameters = deepCopyTable(TargetModelParameters)

	end
	
end

function RandomNetworkDistillation:setPredictorModelParameters(PredictorModelParameters, doNotDeepCopy)
	
	if (doNotDeepCopy) then

		self.PredictorModelParameters = PredictorModelParameters

	else

		self.PredictorModelParameters = deepCopyTable(PredictorModelParameters)

	end

end

return RandomNetworkDistillation