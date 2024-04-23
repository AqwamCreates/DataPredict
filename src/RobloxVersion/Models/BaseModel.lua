local AqwamMatrixLibrary = require(script.Parent.Parent.AqwamMatrixLibraryLinker.Value)

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
		
	else
		
		copy = original
		
	end
	
	return copy
	
end

function BaseModel.new()
	
	local NewBaseModel = {}
	
	setmetatable(NewBaseModel, BaseModel)
	
	NewBaseModel.isOutputPrinted = true

	NewBaseModel.ModelParameters = nil
	
	NewBaseModel.modelParametersInitializationMode = "RandomUniformNegativeAndPositive"
	
	NewBaseModel.numberOfIterationsPerCostCalculation = 1
	
	NewBaseModel.minimumModelParametersInitializationValue = nil

	NewBaseModel.maximumModelParametersInitializationValue = nil
	
	NewBaseModel.iterationWaitDuration = nil
	
	NewBaseModel.dataWaitDuration = nil
	
	NewBaseModel.sequenceWaitDuration = nil
	
	NewBaseModel.targetCostUpperBound = 0
	
	NewBaseModel.targetCostLowerBound = 0
	
	NewBaseModel.currentCostToCheckForConvergence = nil
	
	NewBaseModel.currentNumberOfIterationsToCheckIfConverged = 1
	
	NewBaseModel.numberOfIterationsToCheckIfConverged = math.huge
	
	NewBaseModel.autoResetOptimizers = true

	return NewBaseModel
	
end

function BaseModel:deepCopyTable(original)
	
	return deepCopyTable(original)
	
end

function BaseModel:setNumberOfIterationsToCheckIfConverged(numberOfIterations)
	
	self.numberOfIterationsToCheckIfConverged = numberOfIterations or self.numberOfIterationsToCheckIfConverged
	
end

function BaseModel:checkIfConverged(cost)
	
	if (not cost) then return false end
	
	if (not self.currentCostToCheckForConvergence) then
		
		self.currentCostToCheckForConvergence = cost
		
		return false
		
	end
	
	if (self.currentCostToCheckForConvergence ~= cost) then
		
		self.currentNumberOfIterationsToCheckIfConverged = 1
		
		self.currentCostToCheckForConvergence = cost

		return false
		
	end
	
	if (self.currentNumberOfIterationsToCheckIfConverged < self.numberOfIterationsToCheckIfConverged) then
		
		self.currentNumberOfIterationsToCheckIfConverged += 1
		
		return false
		
	end
	
	self.currentNumberOfIterationsToCheckIfConverged = 1
	
	self.currentCostToCheckForConvergence = nil
	
	return true
	
end

function BaseModel:setTargetCost(upperBound, lowerBound)

	self.targetCostUpperBound = upperBound or self.targetCostUpperBound
	
	self.targetCostLowerBound = lowerBound or self.targetCostLowerBound

end

function BaseModel:checkIfTargetCostReached(cost)
	
	if (not cost) then return false end
	
	return (cost >= self.targetCostLowerBound) and (cost <= self.targetCostUpperBound)
	
end

function BaseModel:calculateCostWhenRequired(currentNumberOfIteration, costFunction)
	
	if ((currentNumberOfIteration % self.numberOfIterationsPerCostCalculation) == 0) then 
		
		return costFunction()
		
	else
		
		return nil
		
	end
	
end

function BaseModel:setNumberOfIterationsPerCostCalculation(numberOfIterationsPerCostCalculation)
	
	self.numberOfIterationsPerCostCalculation = self:getBooleanOrDefaultOption(numberOfIterationsPerCostCalculation, self.numberOfIterationsPerCostCalculation)
	
end

function BaseModel:setAutoResetOptimizers(option)
	
	self.autoResetOptimizers = self:getBooleanOrDefaultOption(option, self.autoResetOptimizers)
	
end

function BaseModel:setWaitDurations(iterationWaitDuration, dataWaitDuration, sequenceWaitDuration)
	
	self.iterationWaitDuration = iterationWaitDuration

	self.dataWaitDuration = dataWaitDuration

	self.sequenceWaitDuration = sequenceWaitDuration
	
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
	
	self:baseModelWait(self.iterationWaitDuration)
	
end

function BaseModel:dataWait()

	self:baseModelWait(self.dataWaitDuration)

end

function BaseModel:sequenceWait()

	self:baseModelWait(self.sequenceWaitDuration)

end

function BaseModel:getModelParameters(doNotDeepCopy)
	
	if doNotDeepCopy then
		
		return self.ModelParameters
		
	else
		
		return deepCopyTable(self.ModelParameters)
		
	end
	
end

function BaseModel:setModelParameters(ModelParameters, doNotDeepCopy)
	
	if ModelParameters and doNotDeepCopy then
		
		self.ModelParameters = ModelParameters
		
	elseif ModelParameters and not doNotDeepCopy then
		
		self.ModelParameters = deepCopyTable(ModelParameters) 
		
	end
	
end

function BaseModel:clearModelParameters()
	
	self.ModelParameters = nil
	
end

function BaseModel:printCostAndNumberOfIterations(cost, numberOfIteration)
	
	if self.isOutputPrinted then print("Iteration: " .. numberOfIteration .. "\t\tCost: " .. cost) end

end

function BaseModel:setPrintOutput(option) 
	
	if (option == false) then
		
		self.isOutputPrinted = false
		
	else
		
		self.isOutputPrinted = true
		
	end
	
end

function BaseModel:getBooleanOrDefaultOption(boolean, defaultBoolean)
	
	if (type(boolean) == "nil") then return defaultBoolean end
		
	return boolean
	
end

function BaseModel:setModelParametersInitializationMode(initializationMode, minimumModelParametersInitializationValue, maximumModelParametersInitializationValue)
	
	self.modelParametersInitializationMode = initializationMode
	
	self.minimumModelParametersInitializationValue = minimumModelParametersInitializationValue
	
	self.maximumModelParametersInitializationValue = maximumModelParametersInitializationValue
	
end

function BaseModel:initializeMatrixBasedOnMode(numberOfRows, numberOfColumns)
	
	local initializationMode = self.modelParametersInitializationMode
	
	if (initializationMode == "Zero") then
		
		return AqwamMatrixLibrary:createMatrix(numberOfRows, numberOfColumns, 0)
	
	elseif (initializationMode == "Random") then
		
		return AqwamMatrixLibrary:createRandomMatrix(numberOfRows, numberOfColumns, self.minimumModelParametersInitializationValue, self.maximumModelParametersInitializationValue)
		
	elseif (initializationMode == "RandomNormalPositive") then
		
		return AqwamMatrixLibrary:createRandomNormalMatrix(numberOfRows, numberOfColumns)
		
	elseif (initializationMode == "RandomNormalNegative") then
		
		local randomNormal = AqwamMatrixLibrary:createRandomNormalMatrix(numberOfRows, numberOfColumns)

		return AqwamMatrixLibrary:multiply(randomNormal, -1)
		
	elseif (initializationMode == "RandomNormalNegativeAndPositive") then
		
		local randomNormal1 = AqwamMatrixLibrary:createRandomNormalMatrix(numberOfRows, numberOfColumns)
		
		local randomNormal2 = AqwamMatrixLibrary:createRandomNormalMatrix(numberOfRows, numberOfColumns)

		return AqwamMatrixLibrary:subtract(randomNormal1, randomNormal2)
		
	elseif (initializationMode == "RandomUniformPositive") then

		return AqwamMatrixLibrary:createRandomUniformMatrix(numberOfRows, numberOfColumns)

	elseif (initializationMode == "RandomUniformNegative") then

		local randomUniform = AqwamMatrixLibrary:createRandomUniformMatrix(numberOfRows, numberOfColumns)

		return AqwamMatrixLibrary:multiply(randomUniform, -1)

	elseif (initializationMode == "RandomUniformNegativeAndPositive") then

		local randomUniform1 = AqwamMatrixLibrary:createRandomUniformMatrix(numberOfRows, numberOfColumns)

		local randomUniform2 = AqwamMatrixLibrary:createRandomUniformMatrix(numberOfRows, numberOfColumns)

		return AqwamMatrixLibrary:subtract(randomUniform1, randomUniform2)
		
	elseif (initializationMode == "HeNormal") then
		
		local variancePart1 = 2 / numberOfRows
		
		local variancePart = math.sqrt(variancePart1)
		
		local randomNormal = AqwamMatrixLibrary:createRandomNormalMatrix(numberOfRows, numberOfColumns)
		
		return  AqwamMatrixLibrary:multiply(variancePart, randomNormal)
		
	elseif (initializationMode == "HeUniform") then

		local variancePart1 = 6 / numberOfRows

		local variancePart = math.sqrt(variancePart1)

		local randomUniform = AqwamMatrixLibrary:createRandomUniformMatrix(numberOfRows, numberOfColumns)

		return  AqwamMatrixLibrary:multiply(variancePart, randomUniform) 
		
	elseif (initializationMode == "XavierNormal") then

		local variancePart1 = 2 / (numberOfRows + numberOfColumns)

		local variancePart = math.sqrt(variancePart1)

		local randomNormal = AqwamMatrixLibrary:createRandomNormalMatrix(numberOfRows, numberOfColumns)

		return AqwamMatrixLibrary:multiply(variancePart, randomNormal) 

	elseif (initializationMode == "XavierUniform") then

		local variancePart1 = 6 / (numberOfRows + numberOfColumns)

		local variancePart = math.sqrt(variancePart1)

		local randomUniform = AqwamMatrixLibrary:createRandomUniformMatrix(numberOfRows, numberOfColumns)

		return AqwamMatrixLibrary:multiply(variancePart, randomUniform)
		
	elseif (initializationMode == "LeCunNormal") then

		local variancePart1 = 1 / numberOfRows

		local variancePart = math.sqrt(variancePart1)

		local randomNormal = AqwamMatrixLibrary:createRandomNormalMatrix(numberOfRows, numberOfColumns)

		return AqwamMatrixLibrary:multiply(variancePart, randomNormal) 
		
	elseif (initializationMode == "LeCunUniform") then

		local variancePart1 = 3 / numberOfRows

		local variancePart = math.sqrt(variancePart1)

		local randomUniform = AqwamMatrixLibrary:createRandomUniformMatrix(numberOfRows, numberOfColumns)

		return AqwamMatrixLibrary:multiply(variancePart, randomUniform) 

	end
	
end

function BaseModel:destroy()
	
	setmetatable(self, nil)
	
	table.clear(self)
	
	self = nil
	
end

return BaseModel
