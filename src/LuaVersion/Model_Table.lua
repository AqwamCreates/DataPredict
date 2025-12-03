--[[

	--------------------------------------------------------------------

	Aqwam's Machine, Deep And Reinforcement Learning Library (DataPredict)

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

local BaseModel = require("Model_BaseModel")

local TableModel = {}

TableModel.__index = TableModel

setmetatable(TableModel, BaseModel)

local defaultLearningRate = 0.1

local function areNumbersOnlyInList(list)

	for i, value in ipairs(list) do

		if (typeof(value) ~= "number") then return false end

	end

	return true

end

local function createClassesList(labelVector)

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

local function checkIfAnyLabelVectorIsNotRecognized(labelVector, ClassesList)

	for i = 1, #labelVector, 1 do

		if (not table.find(ClassesList, labelVector[i][1])) then return true end

	end

	return false

end


function TableModel.new(parameterDictionary)
	
	parameterDictionary = parameterDictionary or {}
	
	local NewTableModel = BaseModel.new(parameterDictionary)
	
	setmetatable(NewTableModel, TableModel)
	
	NewTableModel:setName("TableModel")

	NewTableModel:setClassName("TableModel")
	
	NewTableModel.learningRate = parameterDictionary.learningRate or defaultLearningRate
	
	NewTableModel.Optimizer = parameterDictionary.Optimizer
	
	NewTableModel.FeaturesList = parameterDictionary.FeaturesList or {}
	
	NewTableModel.ClassesList = parameterDictionary.ClassesList or {}
	
	NewTableModel.ModelParameters = parameterDictionary.ModelParameters
	
	return NewTableModel
	
end

function TableModel:setLearningRate(learningRate)

	self.learningRate = learningRate

end

function TableModel:getLearningRate()

	return self.learningRate

end

function TableModel:setOptimizer(Optimizer)

	self.Optimizer = Optimizer

end

function TableModel:getOptimizer()

	return self.Optimizer

end

function TableModel:getOutputMatrix(featureVector, saveFeatureIndexArray)
	
	if (type(featureVector) ~= "table") then featureVector = {{featureVector}} end
	
	local FeaturesList = self.FeaturesList

	local ClassesList = self.ClassesList

	local ModelParameters = self.ModelParameters
	
	local featureIndexArray = {}

	if (not ModelParameters) then

		ModelParameters = self:initializeMatrixBasedOnMode({#FeaturesList, #ClassesList})

		self.ModelParameters = ModelParameters

	end

	local outputMatrix = {}
	
	local feature
	
	local featureIndex

	for i, wrappedFeature in ipairs(featureVector) do

		feature = wrappedFeature[1]

		featureIndex = table.find(FeaturesList, feature)

		if (not featureIndex) then error("Feature \"" .. feature ..  "\" does not exist in the features list.") end

		outputMatrix[i] = ModelParameters[featureIndex]
		
		featureIndexArray[i] = featureIndex

	end
	
	if (saveFeatureIndexArray) then self.featureIndexArray = featureIndexArray end
	
	return outputMatrix, featureIndexArray
	
end

function TableModel:calculateLossFunctionDerivativeMatrix(featureIndexArray, lossGradientMatrix)
	
	local costFunctionDerivativeMatrix = AqwamTensorLibrary:createTensor({#self.FeaturesList, #self.ActionsList})

	for i, index in ipairs(featureIndexArray) do
		
		costFunctionDerivativeMatrix[index] = AqwamTensorLibrary:add({costFunctionDerivativeMatrix[index]}, {lossGradientMatrix[i]})[1]

	end
	
	return costFunctionDerivativeMatrix
	
end

function TableModel:gradientDescent(costFunctionDerivativeMatrix)
	
	local learningRate = self.learningRate
	
	local Optimizer = self.Optimizer
	
	if (Optimizer) then

		costFunctionDerivativeMatrix = Optimizer:calculate(learningRate, costFunctionDerivativeMatrix)

	else

		costFunctionDerivativeMatrix = AqwamTensorLibrary:multiply(learningRate, costFunctionDerivativeMatrix)

	end

	self.ModelParameters = AqwamTensorLibrary:subtract(self.ModelParameters, costFunctionDerivativeMatrix)
	
end

function TableModel:update(lossGradientMatrix, clearFeatureIndexArray)
	
	if (type(lossGradientMatrix) ~= "table") then lossGradientMatrix = {{lossGradientMatrix}} end
	
	local costFunctionDerivativeMatrix = self:calculateLossFunctionDerivativeMatrix(self.featureIndexArray, lossGradientMatrix)
	
	self:gradientDescent(costFunctionDerivativeMatrix)
	
	if (clearFeatureIndexArray) then self.featureIndexArray = nil end
	
end

function NeuralNetworkModel:processLabelVector(labelVector)

	local ClassesList = self.ClassesList

	if (#ClassesList == 0) then

		ClassesList = createClassesList(labelVector)

		local areNumbersOnly = areNumbersOnlyInList(ClassesList)

		if (areNumbersOnly) then table.sort(ClassesList, function(a,b) return a < b end) end

		self.ClassesList = ClassesList

	else

		if checkIfAnyLabelVectorIsNotRecognized(labelVector, ClassesList) then error("A value does not exist in the neural network\'s classes list is present in the label vector.") end

	end

	local logisticMatrix = self:convertLabelVectorToLogisticMatrix(labelVector)

	return logisticMatrix

end

function TableModel:train(featureVector, labelVector)

	if (#featureVector ~= #labelVector) then error("Number of rows of feature vector and the label vector is not the same.") end
	
	local numberOfClasses = #self.ClassesList
	
	local logisticMatrix

	if (#labelVector[1] == 1) and (numberOfClasses ~= 1) then

		logisticMatrix = self:processLabelVector(labelVector)

	else

		if (#labelVector[1] ~= numberOfClasses) then error("The number of columns for the label matrix is not equal to number of neurons at final layer.") end

		logisticMatrix = labelVector

	end
	
	local costArray = {}

	local numberOfIterations = 0
	
	local outputMatrix
	
	local lossGradientMatrix
	
	local cost

	repeat

		numberOfIterations = numberOfIterations + 1

		self:iterationWait()

		outputMatrix = self:getOutputMatrix(featureVector, true)
		
		local lossGradientMatrix = AqwamTensorLibrary:subtract(labelVector, logisticMatrix)

		cost = self:calculateCostWhenRequired(numberOfIterations, function()

			return AqwamTensorLibrary:sum(AqwamTensorLibrary:power(lossGradientMatrix, 2))

		end)

		if (cost) then 

			table.insert(costArray, cost)

			self:printNumberOfIterationsAndCost(numberOfIterations, cost)

		end

		self:update(lossGradientMatrix, true)

	until (numberOfIterations == self.maximumNumberOfIterations) or self:checkIfTargetCostReached(cost) or self:checkIfConverged(cost)

	if (cost == math.huge) then warn("The model diverged. Please repeat the experiment again or change the argument values.") end

	if (self.autoResetOptimizers) then

		for i, Optimizer in ipairs(self.OptimizerArray) do

			if (Optimizer ~= 0) then Optimizer:reset() end

		end

	end

	return costArray

end

function TableModel:predict(featureVector, returnOriginalOutput)
	
	local outputMatrix = self:getOutputMatrix(featureVector, false)
	
	if (returnOriginalOutput) then return outputMatrix end
	
	local ClassesList = self.ClassesList
	
	local outputVector = {}
	
	local maximumValueVector = {}
	
	local maximumValue
	
	local highestClassIndex
	
	local class
	
	for i, unwrappedOutputVector in ipairs(outputMatrix) do
		
		maximumValue = -math.huge
		
		highestClassIndex = nil
		
		for classIndex, value in ipairs(unwrappedOutputVector) do
			
			if (value > maximumValue) then
				
				maximumValue = value
				
				highestClassIndex = classIndex
				
			end
			
		end
		
		class = ClassesList[highestClassIndex] 
		
		if (not class) then error("Class for class index " .. highestClassIndex ..  "  does not exist in the classes list.") end
		
		outputVector[i] = {class}
		
		maximumValueVector[i] = {maximumValue}
		
	end

	return outputVector, maximumValueVector

end

function TableModel:setFeaturesList(FeaturesList)
	
	self.FeaturesList = FeaturesList
	
end

function TableModel:getFeaturesList()
	
	return self.FeaturesList
	
end

function TableModel:setClassesList(ClassesList)
	
	self.ClassesList = ClassesList
	
end

function TableModel:getClassesList()
	
	return self.ClassesList
	
end

return TableModel
