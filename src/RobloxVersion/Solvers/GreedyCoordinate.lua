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
	
	NewGreedyCoordinateSolver:setCalculateFunction(function(weightMatrix, inputMatrix, firstDerivativeMatrix, firstDerivativeLossMatrix)
		
		local isLinear = NewGreedyCoordinateSolver.isLinear

		local transposedJacobianMatrix = (isLinear and NewGreedyCoordinateSolver.cache)
		
		if (not transposedJacobianMatrix) then
			
			local jacobianMatrix = inputMatrix

			if (not isLinear) then jacobianMatrix = AqwamTensorLibrary:multiply(jacobianMatrix, firstDerivativeMatrix) end

			transposedJacobianMatrix = AqwamTensorLibrary:transpose(jacobianMatrix)
			
			if (isLinear) then NewGreedyCoordinateSolver.cache = transposedJacobianMatrix end
			
		end
		
		local weightMatrixDimensionSizeArray = AqwamTensorLibrary:getDimensionSizeArray(weightMatrix)
		
		local coordinateWeightChangeMatrix = AqwamTensorLibrary:createTensor(weightMatrixDimensionSizeArray)
		
		local weightChangeMatrix = AqwamTensorLibrary:dotProduct(transposedJacobianMatrix, firstDerivativeLossMatrix)
		
		local maximumValueDimensionIndexArray, maximumValue = AqwamTensorLibrary:findMaximumValueDimensionIndexArray(weightChangeMatrix)
		
		local minimumValueDimensionIndexArray, minimumValue = AqwamTensorLibrary:findMinimumValueDimensionIndexArray(weightChangeMatrix)
		
		local absoluteMaximumValue = math.abs(maximumValue)
		
		local absoluteMinimumValue = math.abs(minimumValue)
		
		local targetDimensionIndexArray
		
		local targetValue
		
		if (absoluteMaximumValue > absoluteMinimumValue) then
			
			targetDimensionIndexArray = maximumValueDimensionIndexArray
			
			targetValue = maximumValue
			
		elseif (absoluteMinimumValue > absoluteMaximumValue) then
			
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
