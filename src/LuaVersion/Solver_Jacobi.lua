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

local AqwamTensorLibrary = require("AqwamTensorLibraryLink")

local BaseSolver = require("Solver_BaseSolver")

local JacobiSolver = {}

JacobiSolver.__index = JacobiSolver

setmetatable(JacobiSolver, BaseSolver)

local function safeguardedInversionFunction(numerator)
	
	return (numerator == 0 and 0) or (1 / numerator)
	
end

local function rearrangeMatrixToDominantDiagonalMatrix(matrix)
	
	local dimensionSizeArray = AqwamTensorLibrary:getDimensionSizeArray(matrix)
	
	local numberOfRows = dimensionSizeArray[1]
	
	local numberOfColumns = dimensionSizeArray[2]
	
	local dominantDiagonalMatrix = {}
	
	local diagonaMatrixIndexArray = {}
	
	local duplicateValueBlacklistArray = {}
	
	local maximumAbsoluteValue
	
	local columnIndexWithMaximumValue
	
	local absoluteValue
	
	local sumAbsoluteValue
	
	for rowIndex, unwrappedVector in ipairs(matrix) do
		
		maximumAbsoluteValue = 0
		
		columnIndexWithMaximumValue = nil
		
		for primaryColumnIndex, value in ipairs(unwrappedVector) do
			
			absoluteValue = math.abs(value)
			
			sumAbsoluteValue = 0
			
			for secondaryColumnIndex, value in ipairs(unwrappedVector) do
				
				if (secondaryColumnIndex ~= primaryColumnIndex) then sumAbsoluteValue = sumAbsoluteValue + math.abs(value) end
				
			end
			
			if (absoluteValue > sumAbsoluteValue) then
				
				if (absoluteValue > maximumAbsoluteValue) then
					
					maximumAbsoluteValue = absoluteValue

					columnIndexWithMaximumValue = primaryColumnIndex
					
				end
				
			end
			
		end
		
		diagonaMatrixIndexArray[rowIndex] = columnIndexWithMaximumValue
		
	end
	
	for i, value in ipairs(diagonaMatrixIndexArray) do
		
		if (table.find(duplicateValueBlacklistArray, value)) then return matrix end
		
		table.insert(duplicateValueBlacklistArray, value)
		
	end
	
	for rowIndex, unwrappedVector in ipairs(matrix) do
		
		columnIndexWithMaximumValue = diagonaMatrixIndexArray[rowIndex]
		
		dominantDiagonalMatrix[columnIndexWithMaximumValue] = unwrappedVector
		
	end
	
	return dominantDiagonalMatrix
	
end

function JacobiSolver.new(parameterDictionary)
	
	local NewJacobiSolver = BaseSolver.new(parameterDictionary)
	
	setmetatable(NewJacobiSolver, JacobiSolver)
	
	NewJacobiSolver:setName("Jacobi")
	
	NewJacobiSolver:setCalculateFunction(function(weightMatrix, firstDerivativeMatrix, firstDerivativeLossMatrix)
		
		-- Can only cache from linear models since the derivative is a feature matrix. Hence, these values are constant.
		
		local isNonLinearInput = NewJacobiSolver.isNonLinearInput
		
		local isLinearInput = (not isNonLinearInput)
		
		local cache = NewJacobiSolver.cache or {}
		
		local transposedFirstDerivativeMatrix = (isLinearInput and cache[1]) or AqwamTensorLibrary:transpose(firstDerivativeMatrix)
		
		local aMatrix = (isLinearInput and cache[2])
		
		local inverseDiagonalMatrix = cache[3]
		
		local lAndUMatrix = cache[4]
		
		local weightMatrixDimensionSizeArray = AqwamTensorLibrary:getDimensionSizeArray(weightMatrix)

		local numberOfFeatures = weightMatrixDimensionSizeArray[1]

		local numberOfOutputs = weightMatrixDimensionSizeArray[2]
		
		local bMatrix = AqwamTensorLibrary:dotProduct(transposedFirstDerivativeMatrix, firstDerivativeLossMatrix)
		
		local diagonalMatrix

		if (not aMatrix) then

			local transposedFirstDerivativeMatrix = AqwamTensorLibrary:transpose(firstDerivativeMatrix)

			aMatrix = AqwamTensorLibrary:dotProduct(transposedFirstDerivativeMatrix, firstDerivativeMatrix)
			
			aMatrix = rearrangeMatrixToDominantDiagonalMatrix(aMatrix)
			
			if (isLinearInput) then cache[2] = aMatrix end

		end
		
		if (inverseDiagonalMatrix) then
			
			if (not isLinearInput) then 
				
				diagonalMatrix = AqwamTensorLibrary:multiply(diagonalMatrix, aMatrix)

				inverseDiagonalMatrix = AqwamTensorLibrary:applyFunction(safeguardedInversionFunction, diagonalMatrix)
				
			end
		
		else
			
			diagonalMatrix = AqwamTensorLibrary:createIdentityTensor(weightMatrixDimensionSizeArray)
			
			if (isLinearInput) then

				diagonalMatrix = AqwamTensorLibrary:multiply(diagonalMatrix, aMatrix)
				
				inverseDiagonalMatrix = AqwamTensorLibrary:applyFunction(safeguardedInversionFunction, diagonalMatrix)

				cache[3] = diagonalMatrix

			else

				cache[3] = diagonalMatrix

				diagonalMatrix = AqwamTensorLibrary:multiply(diagonalMatrix, aMatrix)
				
				inverseDiagonalMatrix = AqwamTensorLibrary:applyFunction(safeguardedInversionFunction, diagonalMatrix)

			end
			
		end
		
		if (lAndUMatrix) then
			
			if (not isLinearInput) then
				
				lAndUMatrix = AqwamTensorLibrary:multiply(lAndUMatrix, aMatrix)
				
				lAndUMatrix = AqwamTensorLibrary:dotProduct(lAndUMatrix, weightMatrix)
				
			end
			
		else
			
			local upperTriangularTensor = AqwamTensorLibrary:createUpperTriangularTensor(weightMatrixDimensionSizeArray, 0, 1)

			local lowerTriangularTensor = AqwamTensorLibrary:createLowerTriangularTensor(weightMatrixDimensionSizeArray, 0, 1)
			
			lAndUMatrix = AqwamTensorLibrary:add(upperTriangularTensor, lowerTriangularTensor)
			
			if (isLinearInput) then
				
				lAndUMatrix = AqwamTensorLibrary:multiply(lAndUMatrix, aMatrix)
				
				lAndUMatrix = AqwamTensorLibrary:dotProduct(lAndUMatrix, weightMatrix)
				
				cache[4] = lAndUMatrix
				
				
			else
				
				cache[4] = lAndUMatrix
				
				lAndUMatrix = AqwamTensorLibrary:multiply(lAndUMatrix, aMatrix)
				
				lAndUMatrix = AqwamTensorLibrary:dotProduct(lAndUMatrix, weightMatrix)
				
			end
			
		end
		
		local newWeightMatrix = AqwamTensorLibrary:subtract(bMatrix, lAndUMatrix)

		newWeightMatrix = AqwamTensorLibrary:dotProduct(inverseDiagonalMatrix, newWeightMatrix)
		
		NewJacobiSolver.cache = cache
		
		return AqwamTensorLibrary:subtract(newWeightMatrix, weightMatrix)
		
	end)
	
	return NewJacobiSolver
	
end

return JacobiSolver
