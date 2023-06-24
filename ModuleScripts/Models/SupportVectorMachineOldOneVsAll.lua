local SupportVectorMachine = require(script.Parent.SupportVectorMachine)

local AqwamMatrixLibrary = require(script.Parent.Parent.AqwamRobloxMatrixLibraryLinker.Value)

local defaultMaxNumberOfIterations = 500

local defaultLearningRate = 0.3

local defaultCvalue = 0.3

local defaultDistanceFunction = "euclidean"

local defaultTargetCost = 0

local distanceFunctionList = {

	["manhattan"] = function (y, h) return math.abs(y - h) end,

	["euclidean"] = function (y, h) return (y - h)^2 end,

}

local SupportVectorMachineOneVsAllModel = {}

SupportVectorMachineOneVsAllModel.__index = SupportVectorMachineOneVsAllModel

local function getClassesList(labelVector)

	local classesList = {}

	local value

	for i = 1, #labelVector, 1 do

		value = labelVector[i][1]

		if not table.find(classesList, value) then

			table.insert(classesList, value)

		end

	end

	return classesList

end

local function convertToBinaryLabelVector(labelVector, selectedClass)
	
	local numberOfRows = #labelVector
	
	local newLabelVector = AqwamMatrixLibrary:createMatrix(numberOfRows, 1)
	
	for row = 1, numberOfRows, 1 do
		
		if (labelVector[row][1] == selectedClass) then
			
			newLabelVector[row][1] = 1
			
		else
			
			newLabelVector[row][1] = -1
			
		end
		
	end
	
	return newLabelVector
	
end

function SupportVectorMachineOneVsAllModel.new(maxNumberOfIterations, learningRate, cValue, distanceFunction, targetCost)

	local NewSupportVectorMachineOneVsAll = {}
	
	setmetatable(NewSupportVectorMachineOneVsAll, SupportVectorMachineOneVsAllModel)

	NewSupportVectorMachineOneVsAll.maxNumberOfIterations = maxNumberOfIterations or defaultMaxNumberOfIterations

	NewSupportVectorMachineOneVsAll.learningRate = learningRate or defaultLearningRate

	NewSupportVectorMachineOneVsAll.cValue = cValue or defaultCvalue

	NewSupportVectorMachineOneVsAll.distanceFunction = distanceFunction or defaultDistanceFunction

	NewSupportVectorMachineOneVsAll.targetCost = targetCost or defaultTargetCost

	NewSupportVectorMachineOneVsAll.validationFeatureMatrix = nil

	NewSupportVectorMachineOneVsAll.validationLabelVector = nil

	NewSupportVectorMachineOneVsAll.Optimizer = nil
	
	NewSupportVectorMachineOneVsAll.IsOutputPrinted = true

	return NewSupportVectorMachineOneVsAll

end

function SupportVectorMachineOneVsAllModel:setParameters(maxNumberOfIterations, learningRate, cValue, distanceFunction, targetCost)

	self.maxNumberOfIterations = maxNumberOfIterations or self.maxNumberOfIterations

	self.learningRate = learningRate or self.learningRate

	self.cValue = cValue or self.cValue

	self.distanceFunction = distanceFunction or self.distanceFunction

	self.targetCost = targetCost or self.targetCost

end

function SupportVectorMachineOneVsAllModel:setOptimizer(Optimizer)

	self.Optimizer = Optimizer

end

function SupportVectorMachineOneVsAllModel:train(featureMatrix, labelVector)
	
	local classesList = getClassesList(labelVector)
	
	table.sort(classesList, function(a,b) return a < b end)
	
	local total
	
	local totalCost
	
	local cost
	
	local costArray = {}
	
	local ModelParameters = {}
	
	local SupportVectorMachineModel
	
	local SupportVectorMachineModelsArray = {}
	
	local binaryLabelVector
	
	local binaryLabelVectorTable = {}
	
	local ModelParametersVectorColumn
	
	local ModelParametersVectorRow
	
	local numberOfIterations = 0
	
	self.ClassesList = classesList
	
	for i, class in ipairs(classesList) do
		
		SupportVectorMachineModel = SupportVectorMachine.new(1, self.learningRate, self.cValue, self.distanceFunction, self.targetCost)
		
		SupportVectorMachineModel:setOptimizer(self.Optimizer)
		
		SupportVectorMachineModel:setCValue(self.Regularization)
		
		SupportVectorMachineModel:setPrintOutput(false) 
		
		binaryLabelVector = convertToBinaryLabelVector(labelVector, class)
		
		table.insert(SupportVectorMachineModelsArray, SupportVectorMachineModel)
		
		table.insert(binaryLabelVectorTable, binaryLabelVector)
		
	end
	
	repeat

		numberOfIterations += 1
		
		totalCost = 0
		
		for i, class in ipairs(classesList) do
			
			binaryLabelVector = binaryLabelVectorTable[i]
			
			SupportVectorMachineModel = SupportVectorMachineModelsArray[i]

			cost = SupportVectorMachineModel:train(featureMatrix, binaryLabelVector)
			
			cost = cost[1]
			
			totalCost += cost
			
		end
		
		if self.IsOutputPrinted then print("Iteration: " .. numberOfIterations .. "\t\tCost: " .. cost) end
		
		table.insert(costArray, totalCost)

	until (numberOfIterations == self.maxNumberOfIterations) or (math.abs(totalCost) <= self.targetCost)
	
	if (cost == math.huge) then warn("The model diverged! Please repeat the experiment again or change the argument values.") end
	
	for i, class in ipairs(classesList) do
		
		SupportVectorMachineModel = SupportVectorMachineModelsArray[i]
		
		ModelParametersVectorColumn = SupportVectorMachineModel:getModelParameters()

		ModelParametersVectorRow = AqwamMatrixLibrary:transpose(ModelParametersVectorColumn)

		table.insert(ModelParameters, ModelParametersVectorRow[1])
		
	end
	
	self.ModelParameters = AqwamMatrixLibrary:transpose(ModelParameters)
	
	return costArray
	
end

function SupportVectorMachineOneVsAllModel:predict(featureMatrix)
	
	local hypothesis
	
	local highestClass
	
	local longestDistance = -math.huge
	
	local hypothesisVector = AqwamMatrixLibrary:dotProduct(featureMatrix, self.ModelParameters)
	
	for column = 1, #hypothesisVector[1], 1 do
		
		hypothesis = hypothesisVector[1][column]
		
		if (hypothesis > 0) and (hypothesis > longestDistance) then
			
			highestClass = self.ClassesList[column]
			
			longestDistance = hypothesis
			
		end
		
	end
	
	return highestClass, longestDistance
	
end

function SupportVectorMachineOneVsAllModel:getModelParameters()
	
	return self.ModelParameters
	
end

function SupportVectorMachineOneVsAllModel:getClassesList()
	
	return self.ClassesList
	
end

function SupportVectorMachineOneVsAllModel:setModelParameters(ModelParameters)

	self.ModelParameters = ModelParameters

end

function SupportVectorMachineOneVsAllModel:setClassesList(ClassesList)

	self.ClassesList = ClassesList

end

function SupportVectorMachineOneVsAllModel:setPrintOutput(option) 

	if (option == false) then

		self.IsOutputPrinted = false

	else

		self.IsOutputPrinted = true

	end

end

return SupportVectorMachineOneVsAllModel
