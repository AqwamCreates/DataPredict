local AqwamMatrixLibrary = require(script.Parent.Parent.AqwamRobloxMatrixLibraryLinker.Value)

BaseModel = {}

BaseModel.__index = BaseModel

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
		
	else -- number, string, boolean, etc
		
		copy = original
		
	end
	
	return copy
	
end

function BaseModel.new()
	
	local NewBaseModel = {}
	
	setmetatable(NewBaseModel, BaseModel)
	
	NewBaseModel.IsOutputPrinted = true

	NewBaseModel.ModelParameters = nil
	
	NewBaseModel.ModelParametersInitializationMode = "RandomNormalNegativeAndPositive"
	
	NewBaseModel.MinimumModelParametersInitializationValue = nil

	NewBaseModel.MaximumModelParametersInitializationValue = nil
	
	NewBaseModel.IterationWaitDuration = nil
	
	NewBaseModel.DataWaitDuration = nil
	
	NewBaseModel.SequenceWaitDuration = nil
	
	NewBaseModel.AutoResetOptimizers = true

	return NewBaseModel
	
end

function BaseModel:setAutoResetOptimizers(option)
	
	self.AutoResetOptimizers = self:getBooleanOrDefaultOption(option, self.AutoResetOptimizers)
	
end

function BaseModel:setWaitDurations(iterationWaitDuration, dataWaitDuration, sequenceWaitDuration)
	
	self.IterationWaitDuration = iterationWaitDuration

	self.DataWaitDuration = dataWaitDuration

	self.SequenceWaitDuration = sequenceWaitDuration
	
end

function BaseModel:baseModelWait(waitDuration)
	
	if (type(waitDuration) == "nil") or (waitDuration == false) then return nil end
	
	if (type(waitDuration) == "number") then
		
		task.wait(waitDuration)
		
	else
		
		task.wait()
		
	end
	
end

function BaseModel:iterationWait()
	
	self:baseModelWait(self.IterationWaitDuration)
	
end

function BaseModel:dataWait()

	self:baseModelWait(self.DataWaitDuration)

end

function BaseModel:sequenceWait()

	self:baseModelWait(self.SequenceWaitDuration)

end

function BaseModel:getModelParameters()
	
	return deepCopyTable(self.ModelParameters)
	
end

function BaseModel:setModelParameters(ModelParameters)
	
	if ModelParameters then
		
		self.ModelParameters = deepCopyTable(ModelParameters) 
		
	end
	
end

function BaseModel:clearModelParameters()
	
	self.ModelParameters = nil
	
end

function BaseModel:printCostAndNumberOfIterations(cost, numberOfIteration)
	
	if self.IsOutputPrinted then print("Iteration: " .. numberOfIteration .. "\t\tCost: " .. cost) end

end

function BaseModel:setPrintOutput(option) 
	
	if (option == false) then
		
		self.IsOutputPrinted = false
		
	else
		
		self.IsOutputPrinted = true
		
	end
	
end

function BaseModel:getBooleanOrDefaultOption(boolean, defaultBoolean)
	
	if (type(boolean) == "nil") then return defaultBoolean end
		
	return boolean
	
end

function BaseModel:setModelParametersInitializationMode(initializationMode, minimumModelParametersInitializationValue, maximumModelParametersInitializationValue)
	
	self.ModelParametersInitializationMode = initializationMode
	
	self.MinimumModelParametersInitializationValue = minimumModelParametersInitializationValue
	
	self.MaximumModelParametersInitializationValue = maximumModelParametersInitializationValue
	
end

function BaseModel:initializeMatrixBasedOnMode(numberOfRows, numberOfColumns)
	
	local initializationMode = self.ModelParametersInitializationMode
	
	if (initializationMode == "Zero") then
		
		return AqwamMatrixLibrary:createMatrix(numberOfRows, numberOfColumns, 0)
	
	elseif (initializationMode == "Random") then
		
		return AqwamMatrixLibrary:createRandomMatrix(numberOfRows, numberOfColumns, self.MinimumModelParametersInitializationValue, self.MaximumModelParametersInitializationValue)
		
	elseif (initializationMode == "RandomNormalPositive") then
		
		return AqwamMatrixLibrary:createRandomNormalMatrix(numberOfRows, numberOfColumns)
		
	elseif (initializationMode == "RandomNormalNegative") then
		
		local RandomNormal = AqwamMatrixLibrary:createRandomNormalMatrix(numberOfRows, numberOfColumns)

		return AqwamMatrixLibrary:multiply(RandomNormal, -1)
		
	elseif (initializationMode == "RandomNormalNegativeAndPositive") then
		
		local RandomNormal1 = AqwamMatrixLibrary:createRandomNormalMatrix(numberOfRows, numberOfColumns)
		
		local RandomNormal2 = AqwamMatrixLibrary:createRandomNormalMatrix(numberOfRows, numberOfColumns)

		return AqwamMatrixLibrary:subtract(RandomNormal1, RandomNormal2)
		
	elseif (initializationMode == "HeNormal") then
		
		local variancePart1 = 2 / numberOfRows
		
		local variancePart = math.sqrt(variancePart1)
		
		local RandomNormal = AqwamMatrixLibrary:createRandomNormalMatrix(numberOfRows, numberOfColumns)
		
		return  AqwamMatrixLibrary:multiply(variancePart, RandomNormal) 
		
	elseif (initializationMode == "XavierNormal") then

		local variancePart1 = 2 / (numberOfRows + numberOfColumns)

		local variancePart = math.sqrt(variancePart1)

		local RandomNormal = AqwamMatrixLibrary:createRandomNormalMatrix(numberOfRows, numberOfColumns)

		return AqwamMatrixLibrary:multiply(variancePart, RandomNormal) 
		
	elseif (initializationMode == "HeUniform") then

		local variancePart1 = 6 / numberOfRows

		local variancePart = math.sqrt(variancePart1)

		local RandomNormalPart1 = AqwamMatrixLibrary:createRandomNormalMatrix(numberOfRows, numberOfColumns)
		
		local RandomNormalPart2 = AqwamMatrixLibrary:createRandomNormalMatrix(numberOfRows, numberOfColumns)
		
		local RandomNormal = AqwamMatrixLibrary:subtract(RandomNormalPart1, RandomNormalPart2)

		return  AqwamMatrixLibrary:multiply(variancePart, RandomNormal) 

	elseif (initializationMode == "XavierUniform") then

		local variancePart1 = 6 / (numberOfRows + numberOfColumns)

		local variancePart = math.sqrt(variancePart1)

		local RandomNormalPart1 = AqwamMatrixLibrary:createRandomNormalMatrix(numberOfRows, numberOfColumns)

		local RandomNormalPart2 = AqwamMatrixLibrary:createRandomNormalMatrix(numberOfRows, numberOfColumns)

		local RandomNormal = AqwamMatrixLibrary:subtract(RandomNormalPart1, RandomNormalPart2)

		return AqwamMatrixLibrary:multiply(variancePart, RandomNormal) 
		
	elseif (initializationMode == "LeCunNormal") then

		local variancePart1 = 1 / numberOfRows

		local variancePart = math.sqrt(variancePart1)

		local RandomNormal = AqwamMatrixLibrary:createRandomNormalMatrix(numberOfRows, numberOfColumns)

		return  AqwamMatrixLibrary:multiply(variancePart, RandomNormal) 
		
	elseif (initializationMode == "LeCunUniform") then

		local variancePart1 = 1 / numberOfRows

		local variancePart = math.sqrt(variancePart1)

		local RandomNormalPart1 = AqwamMatrixLibrary:createRandomNormalMatrix(numberOfRows, numberOfColumns)

		local RandomNormalPart2 = AqwamMatrixLibrary:createRandomNormalMatrix(numberOfRows, numberOfColumns)

		local RandomNormal = AqwamMatrixLibrary:subtract(RandomNormalPart1, RandomNormalPart2)

		return AqwamMatrixLibrary:multiply(variancePart, RandomNormal) 
		
	end
	
end

function BaseModel:destroy()
	
	setmetatable(self, nil)
	
	table.clear(self)
	
	table.freeze(self)
	
end

return BaseModel
