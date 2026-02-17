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

local BaseSolver = require(script.Parent.BaseSolver)

local RandomCoordinateSolver = {}

RandomCoordinateSolver.__index = RandomCoordinateSolver

setmetatable(RandomCoordinateSolver, BaseSolver)

function RandomCoordinateSolver.new(parameterDictionary)
	
	local NewRandomCoordinateSolver = BaseSolver.new(parameterDictionary)
	
	setmetatable(NewRandomCoordinateSolver, RandomCoordinateSolver)
	
	NewRandomCoordinateSolver:setName("RandomCoordinate")
	
	NewRandomCoordinateSolver:setCalculateFunction(function(weightMatrix, firstDerivativeMatrix, firstDerivativeLossMatrix)
		
		-- Can only cache from linear models since the derivative is a feature matrix. Hence, these values are constant.
		
		local isLinearInput = (not NewRandomCoordinateSolver.isNonLinearInput)

		local transposedFirstDerivativeMatrix = (isLinearInput and NewRandomCoordinateSolver.cache)
		
		if (not transposedFirstDerivativeMatrix) then
			
			transposedFirstDerivativeMatrix = AqwamTensorLibrary:transpose(firstDerivativeMatrix)
			
			if (isLinearInput) then NewRandomCoordinateSolver.cache = transposedFirstDerivativeMatrix end
			
		end
		
		local weightMatrixDimensionSizeArray = AqwamTensorLibrary:getDimensionSizeArray(weightMatrix)
		
		local weightChangeMatrix = AqwamTensorLibrary:createTensor(weightMatrixDimensionSizeArray)
		
		local numberOfData = #firstDerivativeLossMatrix
		
		local numberOfFeatures = weightMatrixDimensionSizeArray[1]
		
		local numberOfOutputs = weightMatrixDimensionSizeArray[2]
		
		local randomFeatureIndex = math.random(1, numberOfFeatures)
		
		local randomOutputIndex = math.random(1, numberOfOutputs)
		
		local transposedFirstDerivativeSubTensor = {transposedFirstDerivativeMatrix[randomFeatureIndex]}
		
		local firstDerivativeLossSubTensor = AqwamTensorLibrary:extract(firstDerivativeLossMatrix, {1, randomOutputIndex}, {numberOfData, randomOutputIndex})
		
		local weightChangeValue = AqwamTensorLibrary:dotProduct(transposedFirstDerivativeSubTensor, firstDerivativeLossMatrix)[1][1]
		
		weightChangeMatrix[randomFeatureIndex][randomOutputIndex] = weightChangeValue
		
		return weightChangeMatrix
		
	end)
	
	return NewRandomCoordinateSolver
	
end

return RandomCoordinateSolver
