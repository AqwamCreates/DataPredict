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

local BaseInstance = require(script.Parent.Parent.Cores.BaseInstance)

local PairwiseRankingDatasetCreator = {}

PairwiseRankingDatasetCreator.__index = PairwiseRankingDatasetCreator

setmetatable(PairwiseRankingDatasetCreator, BaseInstance)

local defaultLabelOutput = "One"

local defaultSamplingProbability = 1

local defaultUseNegativeValueBinaryLabel = false

local labelOutputFunctionList = {
	
	["One"] = function(primaryLabelValue, secondaryLabelValue) return math.sign(primaryLabelValue - secondaryLabelValue) end,
	
	["Difference"] = function(primaryLabelValue, secondaryLabelValue) return (primaryLabelValue - secondaryLabelValue) end,
	
}

function PairwiseRankingDatasetCreator.new(parameterDictionary)
	
	parameterDictionary = parameterDictionary or {}
	
	local NewPairwiseRankingDatasetCreator = BaseInstance.new(parameterDictionary)

	setmetatable(NewPairwiseRankingDatasetCreator, PairwiseRankingDatasetCreator)
	
	NewPairwiseRankingDatasetCreator:setName("PairwiseRankingDatasetCreator")

	NewPairwiseRankingDatasetCreator:setClassName("PairwiseRankingDatasetCreator")
	
	NewPairwiseRankingDatasetCreator.labelOutput = parameterDictionary.labelOutput or defaultLabelOutput
	
	NewPairwiseRankingDatasetCreator.samplingProbability = parameterDictionary.samplingProbability or defaultSamplingProbability
	
	NewPairwiseRankingDatasetCreator.useNegativeValueBinaryLabel = NewPairwiseRankingDatasetCreator:getValueOrDefaultValue(parameterDictionary.useNegativeValueBinaryLabel or defaultUseNegativeValueBinaryLabel)
	
	return NewPairwiseRankingDatasetCreator
	
end

function PairwiseRankingDatasetCreator:createDataset(featureMatrix, labelVector)
	
	local labelOutputFunctionToApply = labelOutputFunctionList[self.labelOutput]
	
	if (not labelOutputFunctionToApply) then error("Invalid label output.") end
	
	local samplingProbability = self.samplingProbability
	
	local useNegativeValueBinaryLabel = self.useNegativeValueBinaryLabel
	
	local pairwiseRankingFeatureMatrix = {}
	
	local pairwiseRankingLabelVector = {}

	local currentComparisonCount = 0
	
	local currentRandomProbability

	local primaryFeatureVector

	local primaryLabelValue

	local secondaryFeatureVector
	
	local secondaryLabelValue
	
	local pairwiseRankingLabelValue

	for i, unwrappedPrimaryFeatureVector in ipairs(featureMatrix) do

		primaryFeatureVector = {unwrappedPrimaryFeatureVector}

		primaryLabelValue = labelVector[i][1]

		for j, unwrappedSecondaryFeatureVector in ipairs(featureMatrix) do

			if (i ~= j) then
				
				secondaryLabelValue = labelVector[j][1]
				
				if (primaryLabelValue ~= secondaryLabelValue) then
					
					currentRandomProbability = math.random()

					if (samplingProbability >= currentRandomProbability) then

						currentComparisonCount = currentComparisonCount + 1

						secondaryFeatureVector = {unwrappedSecondaryFeatureVector}

						pairwiseRankingFeatureMatrix[currentComparisonCount] = AqwamTensorLibrary:subtract(primaryFeatureVector, secondaryFeatureVector)[1]

						pairwiseRankingLabelValue = labelOutputFunctionToApply(primaryLabelValue, secondaryLabelValue)
						
						if (not useNegativeValueBinaryLabel) then
							
							if (pairwiseRankingLabelValue < 0) then pairwiseRankingLabelValue = 0 end
							
						end

						pairwiseRankingLabelVector[currentComparisonCount] = {pairwiseRankingLabelValue}

					end
					
				end

			end

		end

	end

	return pairwiseRankingFeatureMatrix, pairwiseRankingLabelVector
	
end

return PairwiseRankingDatasetCreator
