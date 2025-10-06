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

local NaiveBayesBaseModel = require(script.Parent.NaiveBayesBaseModel)

CategoricalNaiveBayesModel = {}

CategoricalNaiveBayesModel.__index = CategoricalNaiveBayesModel

setmetatable(CategoricalNaiveBayesModel, NaiveBayesBaseModel)

local defaultMode = "Hybrid"

local function applyFunctionToDictionaryArrayArray(functionToApply, dictionaryArrayArray)
	
	local newDictionaryArrayArray = {}
	
	for classIndex, featureDictionaryArray in ipairs(dictionaryArrayArray) do
		
		local newFeatureDictionaryArray = {}
		
		for featureIndex, featureDictionary in ipairs(featureDictionaryArray) do
			
			local newFeatureDictionary = {}
			
			for key, value in pairs(featureDictionary) do
				
				newFeatureDictionary[key] = functionToApply(value)
				
			end
			newFeatureDictionaryArray[featureIndex] = newFeatureDictionary
			
		end
		
		newDictionaryArrayArray[classIndex] = newFeatureDictionaryArray
		
	end
	
	return newDictionaryArrayArray
end

local function calculateCategoricalProbability(useLogProbabilities, featureTable, featureProbabilityDictionaryArray)
	
	local probabilityInitialization = (useLogProbabilities and 0) or 1
	
	local categoricalProbability = probabilityInitialization
	
	for f, value in ipairs(featureTable) do
		
		local featureProbability = featureProbabilityDictionaryArray[f][value] or probabilityInitialization
		
		if useLogProbabilities then

			categoricalProbability = categoricalProbability + math.log(featureProbability)

		else

			categoricalProbability = categoricalProbability * featureProbability

		end
		
	end
	
	return categoricalProbability
	
end

local function calculatePosteriorProbability(useLogProbabilities, featureTable, featureProbabilityDictionaryArray, priorProbabilityValue)

	local posteriorProbability

	local likelihoodProbability = calculateCategoricalProbability(useLogProbabilities, featureTable, featureProbabilityDictionaryArray)

	if (useLogProbabilities) then

		posteriorProbability = likelihoodProbability + priorProbabilityValue

	else

		posteriorProbability = likelihoodProbability * priorProbabilityValue

	end

	return posteriorProbability

end

function CategoricalNaiveBayesModel:calculateCost(featureMatrix, labelMatrix)

	local useLogProbabilities = self.useLogProbabilities

	local ClassesList = self.ClassesList

	local ModelParameters = self.ModelParameters

	local featureProbabilityDictionaryArrayArray = ModelParameters[1]

	local priorProbabilityVector = ModelParameters[2]

	local numberOfData = #featureMatrix

	local numberOfClasses = #ClassesList

	local posteriorProbabilityMatrix = AqwamTensorLibrary:createTensor({numberOfData, numberOfClasses}, 0)

	local featureProbabilityDictionaryArray

	local priorProbabilityValue

	local posteriorProbabilityValue

	local classIndex

	local label
	
	for data, featureTable in ipairs(featureMatrix) do

		for class = 1, numberOfClasses, 1 do

			featureProbabilityDictionaryArray = featureProbabilityDictionaryArrayArray[class]

			priorProbabilityValue = priorProbabilityVector[class][1]

			posteriorProbabilityMatrix[data][class] = calculatePosteriorProbability(useLogProbabilities, featureTable, featureProbabilityDictionaryArray, priorProbabilityValue)

		end

	end

	if (useLogProbabilities) then

		posteriorProbabilityMatrix = AqwamTensorLibrary:applyFunction(math.exp, posteriorProbabilityMatrix)

	end

	local cost = self:categoricalCrossEntropy(labelMatrix, posteriorProbabilityMatrix)

	return cost

end

local function offlineCategoricalNaiveBayes(extractedFeatureMatrixTable, numberOfData, numberOfFeatures)

	local featureProbabilityDictionaryArrayArray = {}
	
	local priorProbabilityVector = {}
	
	local numberOfDataPointVector = {}

	for classIndex, extractedFeatureMatrix in ipairs(extractedFeatureMatrixTable) do

		local numberOfSubData
		
		local featureDictionaryArray = {}
		
		for featureIndex = 1, numberOfFeatures, 1 do featureDictionaryArray[featureIndex] = {} end

		if (type(extractedFeatureMatrix) == "table") then
			
			numberOfSubData = #extractedFeatureMatrix

			for _, unwrappedFeatureVector in ipairs(extractedFeatureMatrix) do
				
				for featureIndex, featureValue in ipairs(unwrappedFeatureVector) do
					
					local featureDictionary = featureDictionaryArray[featureIndex]
					
					featureDictionary[featureValue] = (featureDictionary[featureValue] or 0) + 1
					
				end
				
			end

		else
			
			numberOfSubData = 0
			
		end
		
		priorProbabilityVector[classIndex] = {numberOfSubData / numberOfData}
		
		numberOfDataPointVector[classIndex] = {numberOfSubData}
		
		local featureProbabilityDictionaryArray = {}
		
		for featureProbabilityIndex = 1, numberOfFeatures, 1 do featureProbabilityDictionaryArray[featureProbabilityIndex] = {} end
		
		for featureColumn, featureDictionary in ipairs(featureDictionaryArray) do

			local featureProbabiltyDictionary = {} 

			for featureKey, featureValue in pairs(featureDictionary) do

				featureProbabiltyDictionary[featureKey] = featureValue / numberOfSubData

			end

			featureProbabilityDictionaryArray[featureColumn] = featureProbabiltyDictionary

		end
		
		featureProbabilityDictionaryArrayArray[classIndex] = featureProbabilityDictionaryArray
		
	end

	return featureProbabilityDictionaryArrayArray, priorProbabilityVector, numberOfDataPointVector
	
end

local function calculateMatrices(extractedFeatureMatrixTable, numberOfData, numberOfFeatures, featureProbabilityDictionaryArrayArray, priorProbabilityVector, numberOfDataPointVector)
	
	local newTotalNumberOfDataPoint = numberOfData + AqwamTensorLibrary:sum(numberOfDataPointVector)
	
	local newFeatureProbabilityDictionaryArrayArray = {}

	local newNumberOfDataPointVector = {}
	
	local featureProbabilityDictionaryArray
	
	local numberOfOldSubData

	local numberOfSubData
	
	local newFeatureDictionaryArray
	
	local newFeatureDictionary

	for classIndex, extractedFeatureMatrix in ipairs(extractedFeatureMatrixTable) do
		
		numberOfOldSubData = numberOfDataPointVector[classIndex][1]
		
		featureProbabilityDictionaryArray = featureProbabilityDictionaryArrayArray[classIndex]

		newFeatureDictionaryArray = {}

		if (type(extractedFeatureMatrix) == "table") then

			numberOfSubData = (#extractedFeatureMatrix + numberOfOldSubData)
			
			for featureColumn, featureProbabilityDictionary in ipairs(featureProbabilityDictionaryArray) do

				newFeatureDictionary = {}

				for featureKey, featureProbability in pairs(featureProbabilityDictionary) do newFeatureDictionary[featureKey] = featureProbability * numberOfOldSubData end

				newFeatureDictionaryArray[featureColumn] = newFeatureDictionary

			end

			for _, unwrappedFeatureVector in ipairs(extractedFeatureMatrix) do

				for featureIndex, featureValue in ipairs(unwrappedFeatureVector) do

					newFeatureDictionary = newFeatureDictionaryArray[featureIndex] or {}

					newFeatureDictionary[featureValue] = (newFeatureDictionary[featureValue] or 0) + 1
					
					newFeatureDictionaryArray[featureIndex] = newFeatureDictionary
					
				end

			end
			
			local newFeatureProbabilityDictionaryArray = {}

			for featureColumn, newFeatureDictionary in ipairs(newFeatureDictionaryArray) do

				local newFeatureProbabilityDictionary = {} 

				for featureKey, featureValue in pairs(newFeatureDictionary) do

					newFeatureProbabilityDictionary[featureKey] = featureValue / numberOfSubData

				end

				newFeatureProbabilityDictionaryArray[featureColumn] = newFeatureProbabilityDictionary

			end
			
			newFeatureProbabilityDictionaryArrayArray[classIndex] = newFeatureProbabilityDictionaryArray

		else

			numberOfSubData = numberOfOldSubData
			
			newFeatureProbabilityDictionaryArrayArray[classIndex] = featureProbabilityDictionaryArray

		end

		newNumberOfDataPointVector[classIndex] = {numberOfSubData}

	end
	
	local newPriorProbabilityVector = AqwamTensorLibrary:divide(newNumberOfDataPointVector, newTotalNumberOfDataPoint)

	return newFeatureProbabilityDictionaryArrayArray, newPriorProbabilityVector, newNumberOfDataPointVector

end

function CategoricalNaiveBayesModel.new(parameterDictionary)

	parameterDictionary = parameterDictionary or {}

	local NewCategoricalNaiveBayesModel = NaiveBayesBaseModel.new(parameterDictionary)

	setmetatable(NewCategoricalNaiveBayesModel, CategoricalNaiveBayesModel)

	NewCategoricalNaiveBayesModel:setName("CategoricalNaiveBayes")

	NewCategoricalNaiveBayesModel.mode = parameterDictionary.mode or defaultMode

	NewCategoricalNaiveBayesModel:setTrainFunction(function(featureMatrix, labelVector)

		local mode = NewCategoricalNaiveBayesModel.mode

		local useLogProbabilities = NewCategoricalNaiveBayesModel.useLogProbabilities

		local ModelParameters = NewCategoricalNaiveBayesModel.ModelParameters or {}

		local featureProbabilityDictionaryArrayArray = ModelParameters[1]

		local priorProbabilityVector = ModelParameters[2]

		local numberOfDataPointVector = ModelParameters[3]

		if (mode == "Hybrid") then

			mode = (featureProbabilityDictionaryArrayArray and priorProbabilityVector and numberOfDataPointVector and "Online") or "Offline"		

		end

		if (mode == "Offline") then
			
			featureProbabilityDictionaryArrayArray = nil

			priorProbabilityVector = nil
			
			numberOfDataPointVector = nil

		end
		
		local numberOfData = #featureMatrix

		local numberOfFeatures = #featureMatrix[1]
		
		local numberOfClasses = #NewCategoricalNaiveBayesModel.ClassesList

		local zeroValue = (useLogProbabilities and math.huge) or 0

		local oneValue = (useLogProbabilities and 0) or 1
		
		local logisticMatrix = NewCategoricalNaiveBayesModel:convertLabelVectorToLogisticMatrix(labelVector)

		local extractedFeatureMatrixTable = NewCategoricalNaiveBayesModel:separateFeatureMatrixByClass(featureMatrix, logisticMatrix)
		
		if (not featureProbabilityDictionaryArrayArray) then

			featureProbabilityDictionaryArrayArray = {}

			for class = 1, numberOfClasses, 1 do

				local featureDictionaryArray = {}

				for feature = 1, numberOfFeatures, 1 do

					featureDictionaryArray[numberOfFeatures] = {}

				end

				featureProbabilityDictionaryArrayArray[class] = featureDictionaryArray

			end

		end

		priorProbabilityVector = priorProbabilityVector or AqwamTensorLibrary:createTensor({numberOfClasses, 1}, oneValue)

		numberOfDataPointVector = numberOfDataPointVector or AqwamTensorLibrary:createTensor({numberOfClasses, 1}, 0)

		if (useLogProbabilities) then

			if (featureProbabilityDictionaryArrayArray) then featureProbabilityDictionaryArrayArray = applyFunctionToDictionaryArrayArray(math.exp, featureProbabilityDictionaryArrayArray) end

			if (priorProbabilityVector) then priorProbabilityVector = AqwamTensorLibrary:applyFunction(math.exp, priorProbabilityVector) end

		end

		featureProbabilityDictionaryArrayArray, priorProbabilityVector, numberOfDataPointVector = calculateMatrices(extractedFeatureMatrixTable, numberOfData, numberOfFeatures, featureProbabilityDictionaryArrayArray, priorProbabilityVector, numberOfDataPointVector)

		if (useLogProbabilities) then

			featureProbabilityDictionaryArrayArray = applyFunctionToDictionaryArrayArray(math.log, featureProbabilityDictionaryArrayArray)

			priorProbabilityVector = AqwamTensorLibrary:applyFunction(math.log, priorProbabilityVector)

		end

		NewCategoricalNaiveBayesModel.ModelParameters = {featureProbabilityDictionaryArrayArray, priorProbabilityVector, numberOfDataPointVector}

		local cost = NewCategoricalNaiveBayesModel:calculateCost(featureMatrix, logisticMatrix)

		return {cost}

	end)

	NewCategoricalNaiveBayesModel:setPredictFunction(function(featureMatrix, returnOriginalOutput)

		local ClassesList = NewCategoricalNaiveBayesModel.ClassesList

		local useLogProbabilities = NewCategoricalNaiveBayesModel.useLogProbabilities

		local ModelParameters = NewCategoricalNaiveBayesModel.ModelParameters

		local featureProbabilityDictionaryArrayArray = ModelParameters[1]

		local priorProbabilityVector = ModelParameters[2]

		local numberOfData = #featureMatrix

		local numberOfClasses = #ClassesList

		local posteriorProbabilityMatrix = AqwamTensorLibrary:createTensor({numberOfData, numberOfClasses}, 0)
		
		local featureProbabilityDictionaryArray
		
		local priorProbabilityValue

		for data, featureTable in ipairs(featureMatrix) do

			for class = 1, numberOfClasses, 1 do

				featureProbabilityDictionaryArray = featureProbabilityDictionaryArrayArray[class]

				priorProbabilityValue = priorProbabilityVector[class][1]

				posteriorProbabilityMatrix[data][class] = calculatePosteriorProbability(useLogProbabilities, featureTable, featureProbabilityDictionaryArray, priorProbabilityValue)

			end

		end

		if (returnOriginalOutput) then return posteriorProbabilityMatrix end

		return NewCategoricalNaiveBayesModel:getLabelFromOutputMatrix(posteriorProbabilityMatrix)

	end)

	NewCategoricalNaiveBayesModel:setGenerateFunction(function(labelVector, noiseMatrix)

		local numberOfData = #labelVector
		
		if (noiseMatrix) then

			if (numberOfData ~= #noiseMatrix) then error("The label vector and the noise matrix does not contain the same number of rows.") end

		end

		local ClassesList = NewCategoricalNaiveBayesModel.ClassesList

		local useLogProbabilities = NewCategoricalNaiveBayesModel.useLogProbabilities

		local ModelParameters = NewCategoricalNaiveBayesModel.ModelParameters

		local featureProbabilityDictionaryArrayArray = ModelParameters[1]

		local numberOfFeatures = #featureProbabilityDictionaryArrayArray[1]

		local generatedFeatureMatrix = {}
		
		noiseMatrix = noiseMatrix or AqwamTensorLibrary:createRandomUniformTensor({numberOfData, numberOfFeatures})

		if (useLogProbabilities) then

			featureProbabilityDictionaryArrayArray = applyFunctionToDictionaryArrayArray(math.exp, featureProbabilityDictionaryArrayArray)

		end

		for data, unwrappedLabelVector in ipairs(labelVector) do

			local label = unwrappedLabelVector[1]
			
			local classIndex = table.find(ClassesList, label)
			
			local generatedFeatureVector

			if (classIndex) then
				
				local featureProbabilityDictionaryArray = featureProbabilityDictionaryArrayArray[classIndex]

				generatedFeatureVector = {}

				for featureIndex, featureProbabilityDictionary in ipairs(featureProbabilityDictionaryArray) do

					-- Sample from categorical distribution.
					
					local randomProbability = noiseMatrix[data][featureIndex]
					
					local cumulativeProbability = 0
					
					local chosenValue

					for value, featureProbability in pairs(featureProbabilityDictionary) do
						
						cumulativeProbability = cumulativeProbability + featureProbability
						
						if (randomProbability <= cumulativeProbability) then
							
							chosenValue = value
							
							break
							
						end
						
					end
					
					generatedFeatureVector[featureIndex] = chosenValue
					
				end
				
			else
				
				generatedFeatureVector = table.create(numberOfFeatures, 0)
				
			end
			
			generatedFeatureMatrix[data] = generatedFeatureVector

		end

		return generatedFeatureMatrix

	end)

	return NewCategoricalNaiveBayesModel

end

return CategoricalNaiveBayesModel
