local AqwamMatrixLibrary = require(script.Parent.Parent.AqwamMatrixLibraryLinker.Value)

DistributedGradient = {}

DistributedGradient.__index = DistributedGradient

local defaultGradientChangeMode = "Descent"

local functionToApplyList = {
	
	["Descent"] = function (x, y) return (x - y) end,
	["Ascent"] = function (x, y) return (x + y) end,
	
}

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

function DistributedGradient.new(gradientChangeMode)

	local NewDistributedGradient = {}

	setmetatable(NewDistributedGradient, DistributedGradient)
	
	NewDistributedGradient.gradientChangeMode = gradientChangeMode or defaultGradientChangeMode
	
	NewDistributedGradient.ModelParameters = nil

	NewDistributedGradient.isDistributedGradientRunning = false
	
	NewDistributedGradient.GradientArray = {}

	return NewDistributedGradient

end

function DistributedGradient:setParameters(gradientChangeMode)
	
	self.gradientChangeMode = gradientChangeMode or self.gradientChangeMode
	
end

function DistributedGradient:addGradient(Gradient)
	
	table.insert(self.GradientArray, Gradient)
	
end

function DistributedGradient:setModelParameters(ModelParameters, doNotDeepCopy)
	
	if (doNotDeepCopy) then
		
		self.ModelParameters = ModelParameters
		
	else
		
		self.ModelParameters = deepCopyTable(ModelParameters)
		
	end

end

function DistributedGradient:getModelParameters(doNotDeepCopy)
	
	if (doNotDeepCopy) then

		return self.ModelParameters

	else

		return deepCopyTable(self.ModelParameters)

	end

end

function DistributedGradient:gradientDescent(Gradient, isTableOfMatrices, functionToApply)
	
	local ModelParameters = self.ModelParameters
	
	if isTableOfMatrices then
		
		for i = 1, #ModelParameters, 1 do
			
			ModelParameters[i] = AqwamMatrixLibrary:applyFunction(functionToApply, ModelParameters[i], Gradient)
			
		end
		
	else
		
		ModelParameters = AqwamMatrixLibrary:applyFunction(functionToApply, ModelParameters, Gradient)
		
	end
	
	self.ModelParameters = ModelParameters
	
end

function DistributedGradient:start()
	
	if (self.ModelParameters == nil) then error("No model parameters loaded!") end

	if (self.isDistributedGradientRunning == true) then error("The model is already running!") end
	
	local isTableOfMatrices = checkIfIsTableOfMatrices(self.ModelParameters)
	
	local GradientArray = self.GradientArray
	
	local functionToApply = functionToApplyList[self.gradientChangeMode]

	self.isDistributedGradientRunning = true

	local gradientChangeCoroutine = coroutine.create(function()

		repeat

			task.wait()

			if (#GradientArray == 0) then continue end
			
			while (#GradientArray > 0) do
				
				self:gradientDescent(GradientArray[1], isTableOfMatrices, functionToApply)

				table.remove(GradientArray, 1)
				
			end

		until (self.isDistributedGradientRunning == false)

	end)

	coroutine.resume(gradientChangeCoroutine)

	return gradientChangeCoroutine

end

function DistributedGradient:stop()

	self.isDistributedGradientRunning = false

end

function DistributedGradient:destroy()

	setmetatable(self, nil)

	table.clear(self)

	self = nil

end

return DistributedGradient
