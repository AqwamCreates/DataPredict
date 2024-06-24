--[[

	--------------------------------------------------------------------

	Aqwam's Machine And Deep Learning Library (DataPredict)

	Version 1.32.0

	Author: Aqwam Harish Aiman
	
	YouTube: https://www.youtube.com/channel/UCUrwoxv5dufEmbGsxyEUPZw
	
	LinkedIn: https://www.linkedin.com/in/aqwam-harish-aiman/
	
	--------------------------------------------------------------------
		
	By using this library, you agree to comply with our Terms and Conditions in the link below:
	
	https://github.com/AqwamCreates/DataPredict/blob/main/docs/TermsAndConditions.md
	
	--------------------------------------------------------------------

--]]

local requiredMatrixLibraryVersion = 1.9

local AqwamMatrixLibrary = require(script.AqwamMatrixLibraryLinker.Value)

local Models = script.Models

local AqwamCustomModels = script.AqwamCustomModels

local Others = script.Others

local Optimizers = script.Optimizers

local ExperienceReplays = script.ExperienceReplays

local ModelsDictionary = {
	
	LinearRegression = require(Models.LinearRegression),
	
	LogisticRegression = require(Models.LogisticRegression),
	
	KMeans = require(Models.KMeans),
	
	SupportVectorMachine = require(Models.SupportVectorMachine),
	
	NaiveBayes = require(Models.NaiveBayes),
	
	ExpectationMaximization = require(Models.ExpectationMaximization),
	
	NeuralNetwork = require(Models.NeuralNetwork),
	
	ActorCritic = require(Models.ActorCritic),
	
	AdvantageActorCritic = require(Models.AdvantageActorCritic),
	
	AsynchronousAdvantageActorCritic = require(Models.AsynchronousAdvantageActorCritic),
	
	DeepQLearning = require(Models.DeepQLearning),
	
	DeepDoubleQLearningV1 = require(Models.DeepDoubleQLearningV1),
	
	DeepDoubleQLearningV2 = require(Models.DeepDoubleQLearningV2),
	
	DeepClippedDoubleQLearning = require(Models.DeepClippedDoubleQLearning),
	
	DeepStateActionRewardStateAction = require(Models.DeepStateActionRewardStateAction),
	
	DeepDoubleStateActionRewardStateActionV1 = require(Models.DeepDoubleStateActionRewardStateActionV1),
	
	DeepDoubleStateActionRewardStateActionV2 = require(Models.DeepDoubleStateActionRewardStateActionV2),
	
	DeepExpectedStateActionRewardStateAction = require(Models.DeepExpectedStateActionRewardStateAction),
	
	DeepDoubleExpectedStateActionRewardStateActionV1 = require(Models.DeepDoubleExpectedStateActionRewardStateActionV1),

	DeepDoubleExpectedStateActionRewardStateActionV2 = require(Models.DeepDoubleExpectedStateActionRewardStateActionV2),
	
	DeepDuelingQLearning = require(Models.DeepDuelingQLearning),
	
	REINFORCE = require(Models.REINFORCE),
	
	ProximalPolicyOptimization = require(Models.ProximalPolicyOptimization),
	
	ProximalPolicyOptimizationClip = require(Models.ProximalPolicyOptimizationClip),
	
	VanillaPolicyGradient = require(Models.VanillaPolicyGradient),
	
	KMedoids = require(Models.KMedoids),
	
	AffinityPropagation = require(Models.AffinityPropagation),
	
	AgglomerativeHierarchical = require(Models.AgglomerativeHierarchical),
	
	DensityBasedSpatialClusteringOfApplicationsWithNoise = require(Models.DensityBasedSpatialClusteringOfApplicationsWithNoise),
	
	MeanShift = require(Models.MeanShift),
	
	KNearestNeighbours = require(Models.KNearestNeighbours),
	
	NormalLinearRegression = require(Models.NormalLinearRegression),
	
	GenerativeAdversarialNetwork = require(Models.GenerativeAdversarialNetwork),
	
	ConditionalGenerativeAdversarialNetwork = require(Models.ConditionalGenerativeAdversarialNetwork),
	
	WassersteinGenerativeAdversarialNetwork = require(Models.WassersteinGenerativeAdversarialNetwork),
	
	ConditionalWassersteinGenerativeAdversarialNetwork = require(Models.ConditionalWassersteinGenerativeAdversarialNetwork),
	
}

local AqwamCustomModelsDictionary = {
	
	AdvantageLearningNeuralNetwork = require(AqwamCustomModels.AdvantageLearningNeuralNetwork),
	
	AqwamAdvantageActorCriticV1 = require(AqwamCustomModels.AqwamAdvantageActorCriticV1),
	
	AqwamAdvantageActorCriticV2 = require(AqwamCustomModels.AqwamAdvantageActorCriticV2),
	
	WeightProximalPolicyOptimizationClip = require(AqwamCustomModels.WeightProximalPolicyOptimizationClip),
	
}

local OptimizersDictionary = {
	
	RootMeanSquarePropagation = require(Optimizers.RootMeanSquarePropagation),
	
	Momentum = require(Optimizers.Momentum),
	
	AdaptiveGradient = require(Optimizers.AdaptiveGradient),
	
	AdaptiveGradientDelta = require(Optimizers.AdaptiveGradientDelta),
	
	AdaptiveMomentEstimation = require(Optimizers.AdaptiveMomentEstimation),
	
	AdaptiveMomentEstimationMaximum = require(Optimizers.AdaptiveMomentEstimationMaximum),
	
	NesterovAcceleratedAdaptiveMomentEstimation = require(Optimizers.NesterovAcceleratedAdaptiveMomentEstimation),
	
	Gravity = require(Optimizers.Gravity),
	
	LearningRateStepDecay = require(Optimizers.LearningRateStepDecay),
	
	LearningRateTimeDecay = require(Optimizers.LearningRateTimeDecay),
	
}

local ExperienceReplaysDictionary = {
	
	UniformExperienceReplay = require(ExperienceReplays.UniformExperienceReplay),
	
	PrioritizedExperienceReplay = require(ExperienceReplays.PrioritizedExperienceReplay),
	
	NStepExperienceReplay = require(ExperienceReplays.NStepExperienceReplay),
	
}

local OthersDictionary = {
	
	ModelChecker = require(Others.ModelChecker),
	
	ModelDatasetCreator = require(Others.ModelDatasetCreator),
	
	ModelParametersMerger = require(Others.ModelParametersMerger),
	
	GradientDescentModifier =  require(Others.GradientDescentModifier),
	
	RandomNetworkDistillation = require(Others.RandomNetworkDistillation),
	
	Regularization = require(Others.Regularization),
	
	StringSplitter = require(Others.StringSplitter),
	
	OnlineLearning = require(Others.OnlineLearning),
	
	DistributedGradients = require(Others.DistributedGradients),
	
	DistributedModelParameters = require(Others.DistributedModelParameters),
	
	Tokenizer = require(Others.Tokenizer),
	
	OneVsAll = require(Others.OneVsAll),
	
	ConfusionMatrixCreator = require(Others.ConfusionMatrixCreator),
	
	ReinforcementLearningQuickSetup = require(Others.ReinforcementLearningQuickSetup)
	
}

local AqwamRobloxMachineLearningLibrary = {}

AqwamRobloxMachineLearningLibrary.Models =  ModelsDictionary

AqwamRobloxMachineLearningLibrary.AqwamCustomModels = AqwamCustomModelsDictionary

AqwamRobloxMachineLearningLibrary.Optimizers = OptimizersDictionary

AqwamRobloxMachineLearningLibrary.ExperienceReplays = ExperienceReplaysDictionary

AqwamRobloxMachineLearningLibrary.Others = OthersDictionary

local function checkVersion()
	
	local matrixLibraryVersion
	
	if (AqwamMatrixLibrary == nil) then error("\n\nMatrixL (or Aqwam's Matrix Library) is not linked to this library. \nPlease read the \"Tutorial \" in DataPredict's documentation for installation details.") end
	
	local success = pcall(function()
		
		matrixLibraryVersion = AqwamMatrixLibrary:getVersion()
		
	end)
	
	if not success then matrixLibraryVersion = -1 end
	
	if (matrixLibraryVersion < requiredMatrixLibraryVersion) then warn("The matrix library is out-of-date. You may encounter some problems.") end
	
end

checkVersion()

return AqwamRobloxMachineLearningLibrary
