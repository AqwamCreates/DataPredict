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

local AqwamTensorLibrary = require("AqwamTensorLibrary")

local BaseInstance = require("Cores_BaseInstance")

DistributedGradientsCoordinator = {}

DistributedGradientsCoordinator.__index = DistributedGradientsCoordinator

setmetatable(DistributedGradientsCoordinator, BaseInstance)

local defaultGradientChangeMode = "Descent"

local defaultAveragingRate = 0

local function checkDepth(array, depth)

	depth = depth or 0

	local valueType = typeof(array)

	if (valueType == "table") then

		return checkDepth(array[1], depth + 1)

	else

		return depth

	end

end

local function checkIfIsTableOfMatrices(array)

	local depth = checkDepth(array)

	local isTableOfMatrices = (depth == 3)

	return isTableOfMatrices

end

local function gradientDescent(ModelParameters, Gradients, isTableOfMatrices, averagingRate, averagingRateComplement)

	if (isTableOfMatrices) then

		local averagedSubModelParameters 

		local averagedSubGradient

		for i = 1, #ModelParameters, 1 do

			averagedSubModelParameters = AqwamTensorLibrary:multiply(ModelParameters[i], averagingRateComplement)

			averagedSubGradient = AqwamTensorLibrary:multiply(Gradients[i], averagingRate)

			ModelParameters[i] = AqwamTensorLibrary:subtract(averagedSubModelParameters, averagedSubGradient)

		end

	else

		local averagedModelParameters = AqwamTensorLibrary:multiply(ModelParameters, averagingRateComplement)

		local averagedGradient = AqwamTensorLibrary:multiply(Gradients, averagingRate)

		ModelParameters = AqwamTensorLibrary:subtract(ModelParameters, Gradients)

	end

end

local function gradientAscent(ModelParameters, Gradients, isTableOfMatrices, averagingRate, averagingRateComplement)

	if (isTableOfMatrices) then
		
		local averagedSubModelParameters 

		local averagedSubGradient

		for i = 1, #ModelParameters, 1 do
			
			averagedSubModelParameters = AqwamTensorLibrary:multiply(ModelParameters[i], averagingRateComplement)
			
			averagedSubGradient = AqwamTensorLibrary:multiply(Gradients[i], averagingRate)

			ModelParameters[i] = AqwamTensorLibrary:add(averagedSubModelParameters, averagedSubGradient)

		end

	else
		
		local averagedModelParameters = AqwamTensorLibrary:multiply(ModelParameters, averagingRateComplement)

		local averagedGradient = AqwamTensorLibrary:multiply(Gradients, averagingRate)

		ModelParameters = AqwamTensorLibrary:add(ModelParameters, Gradients)

	end

	return ModelParameters

end

local functionToApplyList = {
	
	["Descent"] = gradientDescent,
	
	["Ascent"] = gradientAscent
	
}

function DistributedGradientsCoordinator.new(parameterDictionary)

	local NewDistributedGradientsCoordinator = BaseInstance.new(parameterDictionary)

	setmetatable(NewDistributedGradientsCoordinator, DistributedGradientsCoordinator)
	
	NewDistributedGradientsCoordinator:setName("DistributedGradientsCoordinator")
	
	NewDistributedGradientsCoordinator:setClassName("DistributedGradientsCoordinator")
	
	NewDistributedGradientsCoordinator.gradientChangeMode = parameterDictionary.gradientChangeMode or defaultGradientChangeMode
	
	NewDistributedGradientsCoordinator.averagingRate = parameterDictionary.averagingRate or defaultAveragingRate
	
	NewDistributedGradientsCoordinator.ModelParameters = parameterDictionary.ModelParameters
	
	NewDistributedGradientsCoordinator.GradientsArray = parameterDictionary.GradientsArray or {}

	NewDistributedGradientsCoordinator.isDistributedGradientsRunning = false

	return NewDistributedGradientsCoordinator

end

function DistributedGradientsCoordinator:addGradients(Gradients)
	
	table.insert(self.GradientsArray, Gradients)
	
end

function DistributedGradientsCoordinator:clearGradients()
	
	table.clear(self.GradientsArray)
	
end

function DistributedGradientsCoordinator:setModelParameters(ModelParameters, doNotDeepCopy)
	
	if (doNotDeepCopy) then
		
		self.ModelParameters = ModelParameters
		
	else
		
		self.ModelParameters = self:deepCopyTable(ModelParameters)
		
	end

end

function DistributedGradientsCoordinator:getModelParameters(doNotDeepCopy)
	
	if (doNotDeepCopy) then

		return self.ModelParameters

	else

		return self:deepCopyTable(self.ModelParameters)

	end

end

function DistributedGradientsCoordinator:start()
	
	if (not self.ModelParameters) then error("No model parameters loaded.") end

	if (self.isDistributedGradientsRunning) then error("It is already running.") end
	
	self.isDistributedGradientsRunning = true
	
	local isTableOfMatrices = checkIfIsTableOfMatrices(self.ModelParameters)
	
	local functionToApply = functionToApplyList[self.gradientChangeMode]
	
	local averagingRate = self.averagingRate

	local averagingRateComplement = 1 - averagingRate
	
	local GradientsArray = self.GradientsArray
	
	local gradientChangeCoroutine = coroutine.create(function()

		repeat

			task.wait()

			while (#GradientsArray > 0) do

				self.ModelParameters = functionToApply(self.ModelParameters, GradientsArray[1], isTableOfMatrices, averagingRate, averagingRateComplement)

				table.remove(GradientsArray, 1)

			end

		until (not self.isDistributedGradientsRunning)

	end)

	coroutine.resume(gradientChangeCoroutine)

	return gradientChangeCoroutine

end

function DistributedGradientsCoordinator:stop()

	self.isDistributedGradientsRunning = false

end

return DistributedGradientsCoordinator
