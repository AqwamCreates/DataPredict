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

local DataPredictLibrary = script.Parent.Parent

local Models = DataPredictLibrary.Models

local Optimizers = DataPredictLibrary.Optimizers

local Regularizers = DataPredictLibrary.Regularizers

local ValueSchedulers = DataPredictLibrary.ValueSchedulers

local AqwamTensorLibrary = require("AqwamTensorLibrary")

local IterativeMethodBaseModel = require("Model_IterativeMethodBaseModel")

OneVsOne = {}

OneVsOne.__index = OneVsOne

setmetatable(OneVsOne, IterativeMethodBaseModel)

local defaultModelName = "LogisticRegression"

local defaultMaximumNumberOfIterations = 500

local defaultUseNegativeOneBinaryLabel = false

local defaultMode = "Value"

local function isTableEmpty(t)
	
	for _ in pairs(t) do return false end
	
	return true
end

function OneVsOne.new(parameterDictionary)
	
	parameterDictionary = parameterDictionary or {}
	
	parameterDictionary.maximumNumberOfIterations = parameterDictionary.maximumNumberOfIterations or defaultMaximumNumberOfIterations
	
	local NewOneVsOne = IterativeMethodBaseModel.new(parameterDictionary)
	
	setmetatable(NewOneVsOne, OneVsOne)
	
	NewOneVsOne:setName("OneVsOne")
	
	NewOneVsOne:setClassName("OneVsOne")
	
	NewOneVsOne.modelName = parameterDictionary.modelName or defaultModelName
	
	NewOneVsOne.useNegativeOneBinaryLabel = NewOneVsOne:getValueOrDefaultValue(parameterDictionary.useNegativeOneBinaryLabel, defaultUseNegativeOneBinaryLabel)
	
	NewOneVsOne.mode = NewOneVsOne:getValueOrDefaultValue(parameterDictionary.mode, defaultMode)
	
	NewOneVsOne.ClassesList = parameterDictionary.ClassesList or {}
	
	NewOneVsOne.ModelArray = parameterDictionary.ModelArray or {}
	
	return NewOneVsOne
	
end

function OneVsOne:generateModel(parameterDictionary)
	
	parameterDictionary = parameterDictionary or {}
	
	parameterDictionary.maximumNumberOfIterations = parameterDictionary.maximumNumberOfIterations or 1

	parameterDictionary.isOutputPrinted = self:getValueOrDefaultValue(parameterDictionary.isOutputPrinted, false)
	
	local SelectedModel = require(Models[self.modelName])
	
	local ClassesList = self.ClassesList
	
	local numberOfClasses = #ClassesList
	
	local ModelArray = {}

	for i = 1, numberOfClasses, 1 do
		
		for j = i + 1, numberOfClasses, 1 do
			
			local classArray = {ClassesList[i], ClassesList[j]}
			
			local ModelObject = SelectedModel.new(parameterDictionary)
			
			ModelArray[classArray] = ModelObject
			
		end

	end

	self.ModelArray = ModelArray
	
	return ModelArray
	
end

function OneVsOne:setModel(parameterDictionary)
	
	local ModelArray = self.ModelArray
	
	if (isTableEmpty(ModelArray)) then 
		
		self:generateModel(parameterDictionary) 
		
	else
		
		for parameterKey, parameterValue in parameterDictionary do

			for _, Model in pairs(ModelArray) do Model[parameterKey] = parameterValue end

		end
		
	end
	
end

function OneVsOne:setOptimizer(parameterDictionary)
	
	if (not parameterDictionary) then return end
	
	local ModelArray = self.ModelArray
	
	if (isTableEmpty(ModelArray)) then ModelArray = self:generateModel() end
	
	local optimizerName = parameterDictionary.optimizerName
	
	if (not optimizerName) then error("No optimizer name.") end
	
	local SelectedOptimizer = require(Optimizers[optimizerName] or ValueSchedulers[optimizerName])
	
	local valueSchedulerName = parameterDictionary.valueSchedulerName
	
	local SelectedValueScheduler
	
	if (valueSchedulerName) then SelectedValueScheduler = require(ValueSchedulers[valueSchedulerName]) end
		
	for classArray, Model in pairs(ModelArray) do 

		local success = pcall(function()
			
			if (SelectedValueScheduler) then parameterDictionary.LearningRateValueScheduler = SelectedValueScheduler.new(parameterDictionary) end
				
			local OptimizerObject = SelectedOptimizer.new(parameterDictionary)

			Model:setOptimizer(OptimizerObject)
				
		end)

		if (not success) then 
			
			warn("The model for \"" .. classArray[1] .. " - " ..  classArray[2] .. "\" class pair does not have setOptimizer() function. No optimizer have been set to the model.") 
			
		end

	end
		
end

function OneVsOne:setRegularizer(parameterDictionary)
	
	if (not parameterDictionary) then return end
	
	local ModelArray = self.ModelArray
	
	if (isTableEmpty(ModelArray)) then ModelArray = self:generateModel() end
	
	local regularizerName = parameterDictionary.regularizerName
	
	if (not regularizerName) then error("No regularizer name.") end
	
	local SelectedRegularizer = require(Regularizers[regularizerName])
	
	local RegularizerObject = SelectedRegularizer.new(parameterDictionary)
	
	for classArray, Model in pairs(ModelArray) do 
		
		local success = pcall(function() Model:setRegularizer(RegularizerObject) end)
		
		if (not success) then 

			warn("The model for \"" .. classArray[1] .. " - " ..  classArray[2] .. "\" class pair does not have setRegularizer() function. No optimizer have been set to the model.") 

		end
	
	end
	
end

function OneVsOne:setClassesList(ClassesList)

	self.ClassesList = ClassesList

end

function OneVsOne:getClassesList()

	return self.ClassesList

end

local function checkIfAnyLabelVectorIsNotRecognized(labelVector, ClassesList)

	for i = 1, #labelVector, 1 do

		if table.find(ClassesList, labelVector[i][1]) then continue end

		return true

	end

	return false

end

local function createClassesList(labelVector)

	local ClassesList = {}

	local value

	for i = 1, #labelVector, 1 do

		value = labelVector[i][1]

		if not table.find(ClassesList, value) then

			table.insert(ClassesList, value)

		end

	end

	return ClassesList

end

function OneVsOne:processLabelVector(labelVector)
	
	local ClassesList = self.ClassesList

	if (#ClassesList == 0) then

		ClassesList = createClassesList(labelVector)

		table.sort(ClassesList, function(a,b) return a < b end)

	else

		if checkIfAnyLabelVectorIsNotRecognized(labelVector, ClassesList) then error("A value does not exist in the classes list is present in the label vector") end

	end
	
	self.ClassesList = ClassesList
	
	return ClassesList

end

local function convertToBinaryLabelVector(featureMatrix, labelVector, positiveClass, negativeClass, useNegativeOneBinaryLabel)
	
	local newFeatureMatrix = {}
	
	local newLabelVector = {}
	
	local labelValueDictionary = {}
	
	labelValueDictionary[positiveClass] = 1
	
	labelValueDictionary[negativeClass] = useNegativeOneBinaryLabel and -1 or 0
	
	local featureIndex = 1
	
	local labelValue

	for i, unwrappedLabelTable in ipairs(labelVector) do
		
		labelValue = unwrappedLabelTable[1]
		
		if (labelValue == positiveClass) or (labelValue == negativeClass) then
			
			newFeatureMatrix[featureIndex] = featureMatrix[i]
			
			newLabelVector[featureIndex] = {labelValueDictionary[labelValue]}
			
			featureIndex = featureIndex + 1
			
		end

	end

	return newFeatureMatrix, newLabelVector

end

function OneVsOne:train(featureMatrix, labelVector)
	
	local ClassesList = self.ClassesList

	local ModelArray = self.ModelArray

	if (#ClassesList == 0) then ClassesList = self:processLabelVector(labelVector) end

	if (isTableEmpty(ModelArray)) then ModelArray = self:generateModel() end
	
	self:processLabelVector(labelVector)
	
	local useNegativeOneBinaryLabel = self.useNegativeOneBinaryLabel
	
	local binaryLabelVectorTable = {}
	
	for classArray, _ in pairs(ModelArray) do

		local binaryFeatureMatrix, binaryLabelVector = convertToBinaryLabelVector(featureMatrix, labelVector,  classArray[1], classArray[2], useNegativeOneBinaryLabel)
		
		binaryLabelVectorTable[classArray] = {binaryFeatureMatrix, binaryLabelVector}

	end
	
	local maximumNumberOfIterations = self.maximumNumberOfIterations
	
	local isOutputPrinted = self.isOutputPrinted
	
	local numberOfIterations = 0
	
	local costArray = {}
	
	local modelCostArray
	
	local totalCost
	
	local cost
	
	repeat
		
		numberOfIterations = numberOfIterations + 1
		
		totalCost = 0
		
		for classArray, Model in pairs(ModelArray) do
			
			local datasetArray = binaryLabelVectorTable[classArray]
			
			local binaryFeatureMatrix = datasetArray[1]
			
			local binaryLabelVector = datasetArray[2]

			modelCostArray = Model:train(binaryFeatureMatrix, binaryLabelVector)

			totalCost = totalCost + modelCostArray[#modelCostArray]

		end
		
		cost = self:calculateCostWhenRequired(numberOfIterations, function() return totalCost end)
		
		if (cost) then

			table.insert(costArray, cost)

			self:printNumberOfIterationsAndCost(numberOfIterations, cost)

		end
				
	until (numberOfIterations >= maximumNumberOfIterations) or self:checkIfTargetCostReached(totalCost) or self:checkIfConverged(totalCost)
	
	return costArray
	
end

local function getClassWithHighestValue(featureVector, ModelArray, thresholdValue)
	
	local selectedClass
	
	local highestValue = -math.huge
	
	local predictedClass

	for classArray, Model in pairs(ModelArray) do 

		local value = Model:predict(featureVector, true)[1][1]
		
		predictedClass = nil
		
		if (value > thresholdValue) then
			
			predictedClass = classArray[1]
			
		elseif (value < thresholdValue) then
			
			predictedClass = classArray[2]
			
		end
		
		if (predictedClass) then
			
			local absoluteValue = math.abs(value)

			if (absoluteValue > highestValue) then

				selectedClass = predictedClass

				highestValue = absoluteValue

			end
			
		end

	end

	return selectedClass, highestValue
	
end

local function getClassWithHighestVote(featureVector, ModelArray, thresholdValue)
	
	local highestVote = -math.huge
	
	local classVoteDictionary = {}
	
	local classToVote
	
	local selectedClass
	
	for classArray, Model in pairs(ModelArray) do 

		local value = Model:predict(featureVector, true)[1][1]
		
		classToVote = nil

		if (value > thresholdValue) then
			
			classToVote = classArray[1]
			
		elseif (value < thresholdValue) then
			
			classToVote = classArray[2]

		end
		
		if (classToVote) then

			classVoteDictionary[classToVote] = (classVoteDictionary[classToVote] or 0) + 1
			
		end

	end
	
	for class, classVote in pairs(classVoteDictionary) do
		
		if (classVote > highestVote) then
			
			selectedClass = class
			
			highestVote = classVote
			
		end
		
	end
	
	return selectedClass, highestVote
	
end

local function getClassWithHighestSoftVote(featureVector, ModelArray, thresholdValue)
	
	local highestValue = -math.huge

	local classValueDictionary = {}
	
	local selectedClass

	for classArray, Model in pairs(ModelArray) do 

		local value = Model:predict(featureVector, true)[1][1]

		local positiveClassValue
		local negativeClassValue

		if (thresholdValue == 0) then
			
			-- Handles useNegativeOneBinaryLabel = true.
			
			-- Convert raw decision value to [0, 1] using sigmoid.
			
			positiveClassValue = 1 / (1 + math.exp(-value))
			
			negativeClassValue = 1 - positiveClassValue

		elseif (thresholdValue == 0.5) then
			
			-- Handles normal probabilistic models with [0, 1] output.
			
			-- Center around threshold instead of hard voting.
			
			if (value > thresholdValue) then
				
				positiveClassValue = value
				
				negativeClassValue = 1 - value
				
			elseif (value < thresholdValue) then
				
				-- small stabilization when below threshold.
				
				positiveClassValue = 1 - value
				
				negativeClassValue = value
				
			else
				
				positiveClassValue = 0
				
				negativeClassValue = 0
				
			end
			
		else
			
			error("Unknown threshold value.")
			
		end
		
		local positiveClass = classArray[1]
		local negativeClass = classArray[2]

		classValueDictionary[positiveClass] = (classValueDictionary[positiveClass] or 0) + positiveClassValue
		classValueDictionary[negativeClass] = (classValueDictionary[negativeClass] or 0) + negativeClassValue

	end

	for class, value in pairs(classValueDictionary) do
		
		if (value > highestValue) then
			
			selectedClass = class
			
			highestValue = value
			
		end
		
	end

	return selectedClass, highestValue
end

local predictionFunctionList = {
	
	["Value"] = getClassWithHighestValue,
	
	["Vote"] = getClassWithHighestVote,
	
	["SoftVote"] = getClassWithHighestSoftVote
	
}

function OneVsOne:predict(featureMatrix)
	
	local ModelArray = self.ModelArray
	
	if (isTableEmpty(ModelArray)) then ModelArray = self:generateModel() end
	
	local predictionFunction = predictionFunctionList[self.mode]
	
	if (not predictionFunction) then error("Unknown mode.") end
	
	local thresholdValue = (self.useNegativeOneBinaryLabel and 0) or 0.5
	
	local numberOfData = #featureMatrix
	
	local selectedClassVector = AqwamTensorLibrary:createTensor({numberOfData, 1})
	
	local classScoreVector = AqwamTensorLibrary:createTensor({numberOfData, 1}, thresholdValue)
	
	for i = 1, #featureMatrix, 1 do
		
		local featureVector = {featureMatrix[i]}
		
		local selectedClass, classScore = predictionFunction(featureVector, ModelArray, thresholdValue)
		
		selectedClassVector[i][1] = selectedClass
		
		classScoreVector[i][1] = classScore
		
	end
	
	return selectedClassVector, classScoreVector
	
end

function OneVsOne:getModelParametersArray(doNotDeepCopy)
	
	local ModelArray = self.ModelArray
	
	if (isTableEmpty(ModelArray)) then ModelArray = self:generateModel() end
	
	local ModelParametersArray = {}
	
	for _, Model in pairs(ModelArray) do 
		
		local ModelParameters = Model:getModelParameters(doNotDeepCopy)
		
		table.insert(ModelParametersArray, ModelParameters) 
		
	end
	
	return ModelParametersArray
	
end

function OneVsOne:setModelParametersArray(ModelParametersArray, doNotDeepCopy)
	
	local ModelArray = self.ModelArray
	
	if (isTableEmpty(ModelArray)) then ModelArray = self:generateModel() end
	
	if (ModelParametersArray == nil) then return nil end
	
	if (#ModelParametersArray ~= #ModelArray) then error("The number of model parameters does not match with the number of models!") end
	
	for classArray, Model in pairs(ModelArray) do 
		
		local ModelParameters = ModelParametersArray[classArray]

		Model:setModelParameters(ModelParameters, doNotDeepCopy)

	end
	
end

function OneVsOne:clearModelParameters()
	
	local ModelArray = self.ModelArray
	
	if (isTableEmpty(ModelArray)) then ModelArray = self:generateModel() end
	
	for _, Model in pairs(ModelArray) do Model:clearModelParameters() end

end

return OneVsOne
