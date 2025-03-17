--[[

	--------------------------------------------------------------------

	Aqwam's Machine And Deep Learning Library (DataPredict)

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

local BaseModel = require(script.Parent.BaseModel)

GaussianNaiveBayes = {}

GaussianNaiveBayes.__index = GaussianNaiveBayes

setmetatable(GaussianNaiveBayes, BaseModel)

local AqwamTensorLibrary = require(script.Parent.Parent.AqwamTensorLibraryLinker.Value)

local function calculateGaussianDensity(useLogProbabilities, featureVector, meanVector, standardDeviationVector)
	
	local logGaussianDensity
	
	local exponentStep1 = AqwamTensorLibrary:subtract(featureVector, meanVector)
	
	local exponentStep2 = AqwamTensorLibrary:power(exponentStep1, 2)
	
	local exponentPart3 = AqwamTensorLibrary:power(standardDeviationVector, 2)
	
	local exponentStep4 = AqwamTensorLibrary:divide(exponentStep2, exponentPart3)
	
	local exponentStep5 = AqwamTensorLibrary:multiply(-0.5, exponentStep4)
	
	local exponentWithTerms = AqwamTensorLibrary:applyFunction(math.exp, exponentStep5)
	
	local divisor = AqwamTensorLibrary:multiply(standardDeviationVector, math.sqrt(2 * math.pi))
	
	local gaussianDensity = AqwamTensorLibrary:divide(exponentWithTerms, divisor)
	
	if (useLogProbabilities) then
		
		logGaussianDensity = AqwamTensorLibrary:applyFunction(math.log, gaussianDensity)
		
		return logGaussianDensity	
		
	else
		
		return gaussianDensity
		
	end
	
end

local function calculateLogLoss(labelVector, predictedProbabilitiesVector)
	
	local loglossFunction = function (y, p) return (y * math.log(p)) + ((1 - y) * math.log(1 - p)) end
	
	local logLossVector = AqwamTensorLibrary:applyFunction(loglossFunction, labelVector, predictedProbabilitiesVector)
	
	local logLossSum = AqwamTensorLibrary:sum(logLossVector)

	local logLoss = -logLossSum / #labelVector
	
	return logLoss
	
end

function GaussianNaiveBayes.new(parameterDictionary)
	
	parameterDictionary = parameterDictionary or {}
	
	local NewGaussianNaiveBayes = BaseModel.new(parameterDictionary)
	
	setmetatable(NewGaussianNaiveBayes, GaussianNaiveBayes)
	
	NewGaussianNaiveBayes.useLogProbabilities = BaseModel:getValueOrDefaultValue(parameterDictionary.useLogProbabilities, false)
	
	return NewGaussianNaiveBayes
	
end

function GaussianNaiveBayes:calculateCost(featureMatrix, labelVector)
	
	local cost
	
	local useLogProbabilities = self.useLogProbabilities

	local meanVector = self.ModelParameters[1]

	local standardDeviationVector = self.ModelParameters[2]
	
	local probabilityVector = self.ModelParameters[3]

	local priorProbabilityMatrix = calculateGaussianDensity(self.useLogProbabilities, featureMatrix, meanVector, standardDeviationVector)

	local multipliedProbabilityVector = AqwamTensorLibrary:multiply(priorProbabilityMatrix, probabilityVector)
	
	local initialProbability = (useLogProbabilities and 0) or 1
	
	local predictedVector = AqwamTensorLibrary:createTensor({#labelVector, 1}, initialProbability)
	
	for data = 1, #featureMatrix, 1 do
		
		for column = 1, #multipliedProbabilityVector[1], 1 do

			if (useLogProbabilities) then

				predictedVector[data][1] = predictedVector[data][1] + multipliedProbabilityVector[1][column]

			else

				predictedVector[data][1] = predictedVector[data][1] * multipliedProbabilityVector[1][column]

			end

		end

	end

	cost = calculateLogLoss(labelVector, predictedVector)
	
	return {cost}
	
end

	
function GaussianNaiveBayes:train(featureMatrix, labelVector)
	
	if (#featureMatrix ~= #labelVector) then error("The feature matrix and the label vector does not contain the same number of rows!") end
	
	local cost
	
	local ModelParameters = self.ModelParameters
	
	local standardDeviationVector, varianceVector, meanVector = AqwamTensorLibrary:standardDeviation(featureMatrix, 1)
	
	local priorProbabilityMatrix = calculateGaussianDensity(self.useLogProbabilities, featureMatrix, meanVector, standardDeviationVector)

	local probabilityVector = AqwamTensorLibrary:mean(priorProbabilityMatrix, 1)
	
	if (ModelParameters) then

		meanVector = AqwamTensorLibrary:divide(AqwamTensorLibrary:add(ModelParameters[1], meanVector), 2) 
		
		standardDeviationVector = AqwamTensorLibrary:divide(AqwamTensorLibrary:add(ModelParameters[2], standardDeviationVector), 2) 
		
		probabilityVector = AqwamTensorLibrary:divide(AqwamTensorLibrary:add(ModelParameters[3], probabilityVector), 2) 

	end
	
	self.ModelParameters = {meanVector, standardDeviationVector, probabilityVector}
	
	cost = self:calculateCost(featureMatrix, labelVector)
	
	return {cost}
	
end

function GaussianNaiveBayes:predict(featureMatrix)
	
	local numberOfData = #featureMatrix
	
	local numberOfFeatures = #featureMatrix[1]
	
	local useLogProbabilities = self.useLogProbabilities
	
	local ModelParameters = self.ModelParameters
	
	local meanVector = ModelParameters[1]
	
	local standardDeviationVector = ModelParameters[2]
	
	local probabilityVector = ModelParameters[3]
	
	local initialProbability = (useLogProbabilities and 0) or 1
	
	local priorProbabilityMatrix = calculateGaussianDensity(useLogProbabilities, featureMatrix, meanVector, standardDeviationVector)
	
	local multipliedProbabilityVector = AqwamTensorLibrary:multiply(priorProbabilityMatrix, probabilityVector)
	
	local predictedVector = AqwamTensorLibrary:createTensor({numberOfData, 1}, initialProbability)
	
	for data = 1, numberOfData, 1 do

		for column = 1, numberOfFeatures, 1 do

			if (useLogProbabilities) then

				predictedVector[data][1] = predictedVector[data][1] + multipliedProbabilityVector[1][column]

			else

				predictedVector[data][1] = predictedVector[data][1] * multipliedProbabilityVector[1][column]

			end

		end

	end

	return predictedVector
	
end

return GaussianNaiveBayes