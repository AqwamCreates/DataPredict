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

DynamicBayesianNetworkModel = {}

DynamicBayesianNetworkModel.__index = DynamicBayesianNetworkModel

setmetatable(DynamicBayesianNetworkModel, BaseModel)

local defaultMode = "Hybrid"

local defaultUseLogProbabilities = false

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
		
		if (useLogProbabilities) then

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

function DynamicBayesianNetworkModel:calculateCost(featureMatrix, labelMatrix)

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

function DynamicBayesianNetworkModel.new(parameterDictionary)

	parameterDictionary = parameterDictionary or {}

	local NewDynamicBayesianNetworkModel = BaseModel.new(parameterDictionary)

	setmetatable(NewDynamicBayesianNetworkModel, DynamicBayesianNetworkModel)

	NewDynamicBayesianNetworkModel:setName("DynamicBayesianNetwork")

	local isHidden = parameterDictionary.isHidden

	local StatesList = parameterDictionary.StatesList or {}

	local ObservationsList = parameterDictionary.ObservationsList or {}

	if (type(isHidden) ~= "boolean") then isHidden = (#ObservationsList > 0) and (ObservationsList ~= StatesList) end
	
	NewDynamicBayesianNetworkModel.mode = parameterDictionary.mode or defaultMode

	NewDynamicBayesianNetworkModel.useLogProbabilities = NewDynamicBayesianNetworkModel:getValueOrDefaultValue(parameterDictionary.useLogProbabilities, defaultUseLogProbabilities)

	NewDynamicBayesianNetworkModel.isHidden = isHidden

	NewDynamicBayesianNetworkModel.StatesList = StatesList

	NewDynamicBayesianNetworkModel.ObservationsList = ObservationsList

	NewDynamicBayesianNetworkModel.TransitionProbabilityOptimizer = parameterDictionary.TransitionProbabilityOptimizer

	NewDynamicBayesianNetworkModel.EmissionProbabilityOptimizer = parameterDictionary.EmissionProbabilityOptimizer

	NewDynamicBayesianNetworkModel.ModelParameters = parameterDictionary.ModelParameters

	return NewDynamicBayesianNetworkModel

end

function DynamicBayesianNetworkModel:train(previousStateVector, currentStateVector, observationStateVector)
	
	local mode = self.mode
	
	local useLogProbabilities = self.useLogProbabilities

	local ModelParameters = self.ModelParameters or {}

	local transitionProbabilityDictionaryArrayArray = ModelParameters[1]

	local emissionProbabilityDictionaryArrayArray = ModelParameters[2]

	local transitionCountMatrix = ModelParameters[3]

	if (mode == "Hybrid") then

		mode = (transitionProbabilityDictionaryArrayArray and emissionProbabilityDictionaryArrayArray and transitionCountMatrix and "Online") or "Offline"		

	end

	if (mode == "Offline") then

		transitionProbabilityDictionaryArrayArray = nil

		emissionProbabilityDictionaryArrayArray = nil

		transitionCountMatrix = nil

	end
	
	local StatesList = self.StatesList
	
	local ObservationsList = self.ObservationsList

	local numberOfData = #previousStateVector
	
	local numberOfStates = #StatesList
	
	local numberOfObservations = #ObservationsList

	local zeroValue = (useLogProbabilities and math.huge) or 0

	local oneValue = (useLogProbabilities and 0) or 1

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

	transitionCountMatrix = transitionCountMatrix or AqwamTensorLibrary:createTensor({numberOfStates, numberOfStates}, 0)

	if (useLogProbabilities) then

		if (featureProbabilityDictionaryArrayArray) then featureProbabilityDictionaryArrayArray = applyFunctionToDictionaryArrayArray(math.exp, featureProbabilityDictionaryArrayArray) end

		if (priorProbabilityVector) then priorProbabilityVector = AqwamTensorLibrary:applyFunction(math.exp, priorProbabilityVector) end

	end

	featureProbabilityDictionaryArrayArray, priorProbabilityVector, transitionCountMatrix = calculateMatrices(extractedFeatureMatrixTable, numberOfData, numberOfFeatures, featureProbabilityDictionaryArrayArray, priorProbabilityVector, numberOfDataPointVector)

	if (useLogProbabilities) then

		featureProbabilityDictionaryArrayArray = applyFunctionToDictionaryArrayArray(math.log, featureProbabilityDictionaryArrayArray)

		priorProbabilityVector = AqwamTensorLibrary:applyFunction(math.log, priorProbabilityVector)

	end

	self.ModelParameters = {featureProbabilityDictionaryArrayArray, priorProbabilityVector, transitionCountMatrix}
	
end

function DynamicBayesianNetworkModel:setPredictFunction(stateMatrix, returnOriginalOutput)

	local ClassesList = self.ClassesList

	local useLogProbabilities = self.useLogProbabilities

	local ModelParameters = self.ModelParameters

	local numberOfClasses = #ClassesList

	local numberOfData = #stateMatrix

	local posteriorProbabilityMatrixDimensionSizeArray = {numberOfData, numberOfClasses}

	local initialValue = (useLogProbabilities and -math.huge) or 0

	if (not ModelParameters) then

		if (returnOriginalOutput) then return AqwamTensorLibrary:createTensor(posteriorProbabilityMatrixDimensionSizeArray, initialValue) end

		local dimensionSizeArray = {numberOfData, 1}

		local placeHolderLabelVector = AqwamTensorLibrary:createTensor(dimensionSizeArray, nil)

		local placeHolderLabelProbabilityVector = AqwamTensorLibrary:createTensor(dimensionSizeArray, initialValue)

		return placeHolderLabelVector, placeHolderLabelProbabilityVector

	end

	local featureProbabilityDictionaryArrayArray = ModelParameters[1]

	local priorProbabilityVector = ModelParameters[2]

	local posteriorProbabilityMatrix = AqwamTensorLibrary:createTensor(posteriorProbabilityMatrixDimensionSizeArray, 0)

	local featureProbabilityDictionaryArray

	local priorProbabilityValue

	for data, featureTable in ipairs(stateMatrix) do

		for class = 1, numberOfClasses, 1 do

			featureProbabilityDictionaryArray = featureProbabilityDictionaryArrayArray[class]

			priorProbabilityValue = priorProbabilityVector[class][1]

			posteriorProbabilityMatrix[data][class] = calculatePosteriorProbability(useLogProbabilities, featureTable, featureProbabilityDictionaryArray, priorProbabilityValue)

		end

	end

	if (returnOriginalOutput) then return posteriorProbabilityMatrix end

	return 

end

return DynamicBayesianNetworkModel
