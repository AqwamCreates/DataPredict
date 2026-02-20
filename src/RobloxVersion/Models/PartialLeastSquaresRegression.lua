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

local PartialLeastSquaresRegressionModel = {}

PartialLeastSquaresRegressionModel.__index = PartialLeastSquaresRegressionModel

setmetatable(PartialLeastSquaresRegressionModel, BaseModel)

local defaultLatentFactorCount = 1

local defaultModelParametersInitializationMode = "Zero"

function PartialLeastSquaresRegressionModel.new(parameterDictionary)
	
	parameterDictionary = parameterDictionary or {}
	
	parameterDictionary.modelParametersInitializationMode = parameterDictionary.modelParametersInitializationMode or defaultModelParametersInitializationMode

	local NewPartialLeastSquaresRegressionModel = BaseModel.new(parameterDictionary)

	setmetatable(NewPartialLeastSquaresRegressionModel, PartialLeastSquaresRegressionModel)

	NewPartialLeastSquaresRegressionModel:setName("PartialLeastSquaresRegression")
	
	NewPartialLeastSquaresRegressionModel.latentFactorCount = parameterDictionary.latentFactorCount or defaultLatentFactorCount 

	return NewPartialLeastSquaresRegressionModel

end

local function calculateOmegaVector(featureMatrix, labelVector, latentFactorCount)
	
	local transposedFeatureMatrix = AqwamTensorLibrary:transpose(featureMatrix)

	local transposedLabelVector = AqwamTensorLibrary:transpose(labelVector)

	local responseVector = AqwamTensorLibrary:dotProduct(transposedFeatureMatrix, labelVector)

	local absoluteResponseVector = AqwamTensorLibrary:applyFunction(math.abs, responseVector)

	local omegaVector = AqwamTensorLibrary:divide(responseVector, absoluteResponseVector)

	local tVector

	local transposedTVector

	local tValue

	local pVector

	local qValue

	local featureChangeMatrix

	local transposedPVector

	for k = 1, latentFactorCount, 1 do

		tVector = AqwamTensorLibrary:dotProduct(featureMatrix, omegaVector)

		transposedTVector = AqwamTensorLibrary:transpose(tVector)

		tValue = AqwamTensorLibrary:dotProduct(transposedTVector, tVector)

		tVector = AqwamTensorLibrary:divide(tVector, tValue)

		pVector = AqwamTensorLibrary:dotProduct(transposedFeatureMatrix, tVector)

		qValue = AqwamTensorLibrary:dotProduct(transposedLabelVector, tVector)

		if (qValue[1][1] == 0) then break end

		transposedPVector = AqwamTensorLibrary:transpose(pVector)

		featureChangeMatrix = AqwamTensorLibrary:dotProduct(tVector, tValue, transposedPVector)

		featureMatrix = AqwamTensorLibrary:subtract(featureMatrix, featureChangeMatrix)

		transposedFeatureMatrix = AqwamTensorLibrary:transpose(featureMatrix)

		omegaVector = AqwamTensorLibrary:dotProduct(transposedFeatureMatrix, labelVector)

	end
	
	return omegaVector, pVector, qValue
	
end


function PartialLeastSquaresRegressionModel:train(featureMatrix, labelVector)

	if (#featureMatrix ~= #labelVector) then error("The feature matrix and the label vector does not contain the same number of rows.") end
	
	local latentFactorCount = self.latentFactorCount
	
	local omegaVector
	
	local pVector
	
	local qValue
	
	local omegaMatrix
	
	local pMatrix
	
	local qVector
	
	for i = 1, latentFactorCount, 1 do
		
		omegaVector, pVector, qValue = calculateOmegaVector(featureMatrix, labelVector, latentFactorCount)
		
		if (omegaMatrix) then
			
			omegaMatrix = AqwamTensorLibrary:concatenate(omegaMatrix, omegaVector, 2)
			
		else
			
			omegaMatrix = omegaVector
			
		end
		
		if (pMatrix) then

			pMatrix = AqwamTensorLibrary:concatenate(pMatrix, pVector, 2)

		else

			pMatrix = pVector

		end
		
		if (qVector) then

			qVector = AqwamTensorLibrary:concatenate(qVector, qValue, 2)

		else

			qVector = qValue

		end
		
	end
	
	local transposedPMatrix = AqwamTensorLibrary:transpose(pMatrix)
	
	local betaMatrix = AqwamTensorLibrary:dotProduct(transposedPMatrix, omegaMatrix)
	
	betaMatrix = AqwamTensorLibrary:inverse(betaMatrix)
	
	betaMatrix = AqwamTensorLibrary:dotProduct(omegaMatrix, betaMatrix, qVector)
	
	local meanFeatureVector = AqwamTensorLibrary:mean(featureMatrix, 1)
	
	local betaBiasMatrix = AqwamTensorLibrary:dotProduct(meanFeatureVector, betaMatrix)
	
	local meanLabelValue = AqwamTensorLibrary:mean(labelVector, 1)
	
	betaBiasMatrix = AqwamTensorLibrary:subtract(meanLabelValue, betaBiasMatrix)

	self.ModelParameters = {betaMatrix, betaBiasMatrix}

end

function PartialLeastSquaresRegressionModel:predict(featureMatrix)

	local ModelParameters = self.ModelParameters
	
	if (not ModelParameters) then
		
		local featureMatrixDimensionSizeArray = AqwamTensorLibrary:getDimensionSizeArray(featureMatrix)
		
		return AqwamTensorLibrary:createTensor(featureMatrixDimensionSizeArray, math.huge)
		
	end
	
	local betaMatrix = ModelParameters[1]
	
	local betaBiasMatrix = ModelParameters[2]
	
	local betaMatrix = AqwamTensorLibrary:dotProduct(featureMatrix, betaMatrix)
	
	betaMatrix = AqwamTensorLibrary:add(betaMatrix, betaBiasMatrix)
	
	return betaMatrix

end

return PartialLeastSquaresRegressionModel
