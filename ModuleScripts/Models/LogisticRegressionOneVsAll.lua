local LogisticRegression = require(script.Parent.LogisticRegression)

local AqwamMatrixLibrary = require(script.Parent.Parent.AqwamRobloxMatrixLibraryLinker.Value)

local defaultMaxNumberOfIterations = 500

local defaultLearningRate = 0.1

local defaultSigmoidFunction = "sigmoid"

local defaultTargetCost = 0

local defaultLambda = 0

local sigmoidFunctionList = {

	["sigmoid"] = function (z) return 1/(1+math.exp(-1 * z)) end,

}

local LogisticRegressionOneVsAllModel = {}

LogisticRegressionOneVsAllModel.__index = LogisticRegressionOneVsAllModel

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
			
			newLabelVector[row][1] = 0
			
		end
		
	end
	
	return newLabelVector
	
end

local function softMax(matrix)
	
	local e = AqwamMatrixLibrary:applyFunction(math.exp, matrix)

	local eSum = AqwamMatrixLibrary:sum(e)

	local result = AqwamMatrixLibrary:divide(e, eSum)
	
	return result
	
end

function LogisticRegressionOneVsAllModel.new(maxNumberOfIterations, learningRate, sigmoidFunction, targetCost)

	local NewLogisticRegressionOneVsAllModel = {}

	setmetatable(NewLogisticRegressionOneVsAllModel, LogisticRegressionOneVsAllModel)

	NewLogisticRegressionOneVsAllModel.maxNumberOfIterations = maxNumberOfIterations or defaultMaxNumberOfIterations

	NewLogisticRegressionOneVsAllModel.learningRate = learningRate or defaultLearningRate

	NewLogisticRegressionOneVsAllModel.sigmoidFunction = sigmoidFunction or defaultSigmoidFunction

	NewLogisticRegressionOneVsAllModel.targetCost = targetCost or defaultTargetCost

	NewLogisticRegressionOneVsAllModel.validationFeatureMatrix = nil

	NewLogisticRegressionOneVsAllModel.validationLabelVector = nil

	NewLogisticRegressionOneVsAllModel.Optimizer = nil

	NewLogisticRegressionOneVsAllModel.Regularization = nil
	
	NewLogisticRegressionOneVsAllModel.ModelParameters = nil
	
	NewLogisticRegressionOneVsAllModel.ClassesList = {}
	
	NewLogisticRegressionOneVsAllModel.IsOutputPrinted = true

	return NewLogisticRegressionOneVsAllModel

end

function LogisticRegressionOneVsAllModel:setParameters(maxNumberOfIterations, learningRate, sigmoidFunction, targetCost)

	self.maxNumberOfIterations = maxNumberOfIterations or self.maxNumberOfIterations

	self.learningRate = learningRate or self.learningRate

	self.sigmoidFunction = sigmoidFunction or self.sigmoidFunction

	self.targetCost = targetCost or self.targetCost

end

function LogisticRegressionOneVsAllModel:setRegularization(Regularization)

	self.Regularization = Regularization

end

function LogisticRegressionOneVsAllModel:train(featureMatrix, labelVector)
	
	local classesList = getClassesList(labelVector)
	
	table.sort(classesList, function(a,b) return a < b end)
	
	local total
	
	local totalCost
	
	local cost
	
	local costArray = {}
	
	local internalCostArray = {}
	
	local ModelParameters = {}
	
	local LogisticRegressionModel
	
	local LogisticRegressionModelsArray = {}
	
	local binaryLabelVector
	
	local binaryLabelVectorTable = {}
	
	local ModelParametersVectorColumn
	
	local ModelParametersVectorRow
	
	local numberOfIterations = 0
	
	self.ClassesList = classesList
	
	for i, class in ipairs(classesList) do
		
		LogisticRegressionModel = LogisticRegression.new(1, self.learningRate, self.sigmoidFunction, self.targetCost)
		
		LogisticRegressionModel:setRegularization(self.Regularization)
		
		LogisticRegressionModel:setPrintOutput(false) 
		
		binaryLabelVector = convertToBinaryLabelVector(labelVector, class)
		
		table.insert(LogisticRegressionModelsArray, LogisticRegressionModel)
		
		table.insert(binaryLabelVectorTable, binaryLabelVector)
		
	end
	
	repeat

		numberOfIterations += 1
		
		totalCost = 0
		
		for i, class in ipairs(classesList) do
			
			binaryLabelVector = binaryLabelVectorTable[i]
			
			LogisticRegressionModel = LogisticRegressionModelsArray[i]

			internalCostArray = LogisticRegressionModel:train(featureMatrix, binaryLabelVector)
			
			cost = internalCostArray[1]
			
			totalCost += cost
			
		end
		
		if self.IsOutputPrinted then print("Iteration: " .. numberOfIterations .. "\t\tCost: " .. cost) end
		
		table.insert(costArray, totalCost)

	until (numberOfIterations == self.maxNumberOfIterations) or (math.abs(totalCost) <= self.targetCost)
	
	if (cost == math.huge) then warn("The model diverged! Please repeat the experiment again or change the argument values.") end
	
	for i, class in ipairs(classesList) do
		
		LogisticRegressionModel = LogisticRegressionModelsArray[i]
		
		ModelParametersVectorColumn = LogisticRegressionModel:getModelParameters()

		ModelParametersVectorRow = AqwamMatrixLibrary:transpose(ModelParametersVectorColumn)

		table.insert(ModelParameters, ModelParametersVectorRow[1])
		
	end
	
	self.ModelParameters = AqwamMatrixLibrary:transpose(ModelParameters)
	
	return costArray
	
end

function LogisticRegressionOneVsAllModel:predict(featureMatrix)
	
	local highestClass
	
	local probability

	local highestProbability = -math.huge
	
	local zVector = AqwamMatrixLibrary:dotProduct(featureMatrix, self.ModelParameters)
	
	local zNormalVector = AqwamMatrixLibrary:normalize(zVector)
	
	local softMaxVector = softMax(zNormalVector)
	
	for column = 1, #softMaxVector[1], 1 do
		
		probability = softMaxVector[1][column]
		
		if (probability > highestProbability) then
			
			highestClass = self.ClassesList[column]
			
			highestProbability = probability
			
		end
		
	end
	
	return highestClass, highestProbability
	
end

function LogisticRegressionOneVsAllModel:getModelParameters()
	
	return self.ModelParameters
	
end

function LogisticRegressionOneVsAllModel:getClassesList()
	
	return self.ClassesList
	
end

function LogisticRegressionOneVsAllModel:setModelParameters(ModelParameters)

	self.ModelParameters = ModelParameters

end

function LogisticRegressionOneVsAllModel:setClassesList(ClassesList)

	self.ClassesList = ClassesList

end

function LogisticRegressionOneVsAllModel:setPrintOutput(option) 

	if (option == false) then

		self.IsOutputPrinted = false

	else

		self.IsOutputPrinted = true

	end

end

return LogisticRegressionOneVsAllModel
