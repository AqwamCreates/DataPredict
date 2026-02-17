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

local GreedyCoordinateSolver = {}

GreedyCoordinateSolver.__index = GreedyCoordinateSolver

setmetatable(GreedyCoordinateSolver, BaseSolver)

function GreedyCoordinateSolver.new(parameterDictionary)
	
	local NewGreedyCoordinateSolver = BaseSolver.new(parameterDictionary)
	
	setmetatable(NewGreedyCoordinateSolver, GreedyCoordinateSolver)
	
	NewGreedyCoordinateSolver:setName("GreedyCoordinate")
	
	NewGreedyCoordinateSolver:setCalculateFunction(function(weightMatrix, firstDerivativeMatrix, firstDerivativeLossMatrix)
		
		-- Can only cache from linear models since the derivative is a feature matrix. Hence, these values are constant.
		
		local isLinearInput = (not NewGreedyCoordinateSolver.isNonLinearInput)

		local transposedFirstDerivativeMatrix = (isLinearInput and NewGreedyCoordinateSolver.cache)
		
		if (not transposedFirstDerivativeMatrix) then
			
			transposedFirstDerivativeMatrix = AqwamTensorLibrary:transpose(firstDerivativeMatrix)
			
			if (isLinearInput) then NewGreedyCoordinateSolver.cache = transposedFirstDerivativeMatrix end
			
		end
		
		local weightMatrixDimensionSizeArray = AqwamTensorLibrary:getDimensionSizeArray(weightMatrix)
		
		local coordinateWeightChangeMatrix = AqwamTensorLibrary:createTensor(weightMatrixDimensionSizeArray)
		
		local weightChangeMatrix = AqwamTensorLibrary:dotProduct(transposedFirstDerivativeMatrix, firstDerivativeLossMatrix)
		
		local maximumValueDimensionIndexArray, maximumValue = AqwamTensorLibrary:findMaximumValueDimensionIndexArray(weightChangeMatrix)
		
		local minimumValueDimensionIndexArray, minimumValue = AqwamTensorLibrary:findMinimumValueDimensionIndexArray(weightChangeMatrix)
		
		local maximumValueMagnitude = math.abs(maximumValue)
		
		local minimumValueMagnitude = math.abs(minimumValue)
		
		local targetDimensionIndexArray
		
		local targetValue
		
		if (maximumValueMagnitude > minimumValueMagnitude) then
			
			targetDimensionIndexArray = maximumValueDimensionIndexArray
			
			targetValue = maximumValue
			
		elseif (minimumValueMagnitude > minimumValueMagnitude) then
			
			targetDimensionIndexArray = minimumValueDimensionIndexArray
			
			targetValue = minimumValue
			
		else
			
			if (math.random() > 0.5) then
				
				targetDimensionIndexArray = maximumValueDimensionIndexArray
				
				targetValue = maximumValue
				
			else
				
				targetDimensionIndexArray = minimumValueDimensionIndexArray
				
				targetValue = minimumValue
				
			end
			
		end
		
		local featureIndex = targetDimensionIndexArray[1]
		
		local outputIndex = targetDimensionIndexArray[2]
		
		coordinateWeightChangeMatrix[featureIndex][outputIndex] = targetValue
		
		return coordinateWeightChangeMatrix
		
	end)
	
	return NewGreedyCoordinateSolver
	
end

return GreedyCoordinateSolver
