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

local AqwamTensorLibrary = require(script.Parent.Parent.AqwamTensorLibraryLinker.Value)

local BaseModel = require(script.Parent.BaseModel)

NaiveBayesBaseModel = {}

NaiveBayesBaseModel.__index = NaiveBayesBaseModel

setmetatable(NaiveBayesBaseModel, BaseModel)

local defaultUseLogProbabilities = false

local function areNumbersOnlyInList(list)

	for i, value in ipairs(list) do

		if (typeof(value) ~= "number") then return false end

	end

	return true

end

local function extractFeatureMatrixFromPosition(featureMatrix, positionList)

	local extractedFeatureMatrix = {}

	for i = 1, #featureMatrix, 1 do

		if table.find(positionList, i) then

			table.insert(extractedFeatureMatrix, featureMatrix[i])

		end	

	end

	return extractedFeatureMatrix

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

local function checkIfAnyLabelVectorIsNotRecognized(labelVector, ClassesList)

	for i = 1, #labelVector, 1 do

		if (not table.find(ClassesList, labelVector[i][1])) then return true end

	end

	return false

end

function NaiveBayesBaseModel.new(parameterDictionary)
	
	parameterDictionary = parameterDictionary or {}
	
	local NewBaseModel = BaseModel.new(parameterDictionary)

	setmetatable(NewBaseModel, NaiveBayesBaseModel)

	NewBaseModel:setName("NaiveBayesBaseModel")

	NewBaseModel:setClassName("NaiveBayesModel")
	
	NewBaseModel.ClassesList = parameterDictionary.ClassesList or {}
	
	NewBaseModel.useLogProbabilities = BaseModel:getValueOrDefaultValue(parameterDictionary.useLogProbabilities, defaultUseLogProbabilities)

	return NewBaseModel
	
end

function NaiveBayesBaseModel:separateFeatureMatrixByClass(featureMatrix, labelMatrix)
	
	local ClassesList = self.ClassesList
	
	local extractedFeatureMatrixTable = {}
	
	for i = 1, #ClassesList, 1 do extractedFeatureMatrixTable[i] = {} end
	
	for i, labelTable in ipairs(labelMatrix) do
		
		for j, labelValue in ipairs(labelTable) do
			
			if (labelValue > 0) then table.insert(extractedFeatureMatrixTable[j], featureMatrix[i]) end
			
		end
		
	end

	return extractedFeatureMatrixTable

end

function NaiveBayesBaseModel:processLabelVector(labelVector)
	
	local ClassesList = self.ClassesList

	if (#ClassesList == 0) then

		ClassesList = createClassesList(labelVector)

		local areNumbersOnly = areNumbersOnlyInList(ClassesList)

		if (areNumbersOnly) then table.sort(ClassesList, function(a,b) return a < b end) end
		
		self.ClassesList = ClassesList

	else

		if checkIfAnyLabelVectorIsNotRecognized(labelVector, ClassesList) then error("A value does not exist in the model\'s classes list is present in the label vector.") end

	end

end

function NaiveBayesBaseModel:convertLabelVectorToLogisticMatrix(labelVector)

	if (typeof(labelVector) == "number") then

		labelVector = {{labelVector}}

	end

	local incorrectLabelValue

	local numberOfData = #labelVector
	
	local ClassesList = self.ClassesList

	local logisticMatrix = AqwamTensorLibrary:createTensor({numberOfData, #ClassesList}, 0)

	local label

	local labelPosition

	for data = 1, numberOfData, 1 do

		label = labelVector[data][1]
		
		labelPosition = table.find(ClassesList, label)
		
		if (labelPosition) then
			
			logisticMatrix[data][labelPosition] = 1
			
		end

	end

	return logisticMatrix

end

function NaiveBayesBaseModel:getLabelFromOutputMatrix(outputMatrix)
	
	local ClassesList = self.ClassesList

	local numberOfData = #outputMatrix

	local predictedLabelVector = AqwamTensorLibrary:createTensor({numberOfData, 1}, 0)

	local highestProbabilityVector = AqwamTensorLibrary:createTensor({numberOfData, 1}, 0)

	local highestProbability

	local outputVector

	local classIndexArray

	local predictedLabel

	for i = 1, #outputMatrix, 1 do

		outputVector = {outputMatrix[i]}

		classIndexArray, highestProbability = AqwamTensorLibrary:findMaximumValueDimensionIndexArray(outputMatrix)

		if (classIndexArray) then
			
			predictedLabel = ClassesList[classIndexArray[2]]

			predictedLabelVector[i][1] = predictedLabel

			highestProbabilityVector[i][1] = highestProbability
			
		end

	end

	return predictedLabelVector, highestProbabilityVector

end

function NaiveBayesBaseModel:categoricalCrossEntropy(labelMatrix, generatedLabelMatrix)
	
	local functionToApply = function (labelValue, generatedLabelValue) return -(labelValue * math.log(generatedLabelValue)) end

	local categoricalCrossEntropyTensor = AqwamTensorLibrary:applyFunction(functionToApply, labelMatrix, generatedLabelMatrix)

	local sumCategoricalCrossEntropyValue = AqwamTensorLibrary:sum(categoricalCrossEntropyTensor)

	return sumCategoricalCrossEntropyValue
	
end

function NaiveBayesBaseModel:train(featureMatrix, labelVector)
	
	if (#featureMatrix ~= #labelVector) then error("The feature matrix and the label vector does not contain the same number of rows.") end
	
	self:processLabelVector(labelVector)
	
	return self.trainFunction(featureMatrix, labelVector)
	
end


function NaiveBayesBaseModel:setTrainFunction(trainFunction)
	
	self.trainFunction = trainFunction
	
end

function NaiveBayesBaseModel:predict(featureMatrix, returnOriginalOutput)

	return self.predictFunction(featureMatrix, returnOriginalOutput)

end

function NaiveBayesBaseModel:setPredictFunction(predictFunction)

	self.predictFunction = predictFunction

end

function NaiveBayesBaseModel:generate(labelVector, ...)
	
	return self.generateFunction(labelVector, ...)
	
end

function NaiveBayesBaseModel:setGenerateFunction(generateFunction)
	
	self.generateFunction = generateFunction
	
end

function NaiveBayesBaseModel:getClassesList()

	return self.ClassesList

end

function NaiveBayesBaseModel:setClassesList(ClassesList)

	self.ClassesList = ClassesList

end

return NaiveBayesBaseModel
