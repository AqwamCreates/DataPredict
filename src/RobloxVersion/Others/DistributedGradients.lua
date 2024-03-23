local AqwamMatrixLibrary = require(script.Parent.Parent.AqwamMatrixLibraryLinker.Value)

DistributedGradients = {}

DistributedGradients.__index = DistributedGradients

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

function DistributedGradients.new(gradientChangeMode)

	local NewDistributedGradients = {}

	setmetatable(NewDistributedGradients, DistributedGradients)
	
	NewDistributedGradients.gradientChangeMode = gradientChangeMode or defaultGradientChangeMode
	
	NewDistributedGradients.ModelParameters = nil

	NewDistributedGradients.isDistributedGradientsRunning = false
	
	NewDistributedGradients.GradientsArray = {}

	return NewDistributedGradients

end

function DistributedGradients:setParameters(gradientChangeMode)
	
	self.gradientChangeMode = gradientChangeMode or self.gradientChangeMode
	
end

function DistributedGradients:addGradients(Gradients)
	
	table.insert(self.GradientsArray, Gradients)
	
end

function DistributedGradients:clearGradients()
	
	table.clear(self.GradientsArray)
	
end

function DistributedGradients:setModelParameters(ModelParameters, doNotDeepCopy)
	
	if (doNotDeepCopy) then
		
		self.ModelParameters = ModelParameters
		
	else
		
		self.ModelParameters = deepCopyTable(ModelParameters)
		
	end

end

function DistributedGradients:getModelParameters(doNotDeepCopy)
	
	if (doNotDeepCopy) then

		return self.ModelParameters

	else

		return deepCopyTable(self.ModelParameters)

	end

end

function DistributedGradients:gradientDescent(Gradient, isTableOfMatrices, functionToApply)
	
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

function DistributedGradients:start()
	
	if (self.ModelParameters == nil) then error("No model parameters loaded!") end

	if (self.isDistributedGradientsRunning == true) then error("The model is already running!") end
	
	local isTableOfMatrices = checkIfIsTableOfMatrices(self.ModelParameters)
	
	local GradientsArray = self.GradientsArray
	
	local functionToApply = functionToApplyList[self.gradientChangeMode]

	self.isDistributedGradientsRunning = true

	local gradientChangeCoroutine = coroutine.create(function()

		repeat

			task.wait()

			if (#GradientsArray == 0) then continue end
			
			while (#GradientsArray > 0) do
				
				self:gradientDescent(GradientsArray[1], isTableOfMatrices, functionToApply)

				table.remove(GradientsArray, 1)
				
			end

		until (self.isDistributedGradientsRunning == false)

	end)

	coroutine.resume(gradientChangeCoroutine)

	return gradientChangeCoroutine

end

function DistributedGradients:stop()

	self.isDistributedGradientsRunning = false

end

function DistributedGradients:destroy()

	setmetatable(self, nil)

	table.clear(self)

	self = nil

end

return DistributedGradients
