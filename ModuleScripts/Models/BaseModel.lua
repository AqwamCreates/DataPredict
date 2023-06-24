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
	
	NewBaseModel.LastPredictedOutput = nil
	
	NewBaseModel.LastCalculations = nil
	
	NewBaseModel.ModelParametersInitializationMode = "RandomNormalPositive"
	
	NewBaseModel.MinimumModelParametersInitializationValue = nil

	NewBaseModel.MaximumModelParametersInitializationValue = nil

	return NewBaseModel
	
end

function BaseModel:getModelParameters()
	
	return deepCopyTable(self.ModelParameters)
	
end

function BaseModel:setModelParameters(ModelParameters)
	
	self.ModelParameters = ModelParameters or self.ModelParameters
	
end

function BaseModel:clearModelParameters()
	
	self.ModelParameters = nil
	
end

function BaseModel:clearLastPredictedOutput()
	
	self.LastPredictedOutput = nil
	
end

function BaseModel:clearLastCalculations()
	
	self.LastCalculations = nil
	
end

function BaseModel:clearLastPredictedOutputAndCalculations()
	
	BaseModel:clearLastCalculations()
	
	BaseModel:clearLastPredictedOutput()
	
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
		
	elseif (initializationMode == "He") then

		local variance = 2 / numberOfColumns
		
		return AqwamMatrixLibrary:createRandomNormalMatrix(numberOfRows, numberOfColumns) * math.sqrt(variance)

	elseif (initializationMode == "Xavier") then

		local variance = 1 / numberOfColumns
		
		return AqwamMatrixLibrary:createRandomNormalMatrix(numberOfRows, numberOfColumns) * math.sqrt(variance)
		
	elseif (initializationMode == "Uniform") then
		
		local range = math.sqrt(3 / numberOfColumns)  -- Uniform initialization range
		
		return AqwamMatrixLibrary:createRandomUniformMatrix(numberOfRows, numberOfColumns, -range, range)

	end
	
end


function BaseModel:destroy()
	
	setmetatable(self, nil)
	
	table.clear(self)
	
	table.freeze(self)
	
end

return BaseModel

