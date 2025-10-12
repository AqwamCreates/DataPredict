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

local AqwamMachineAndDeepLearningLibrary = {}

local Models = script.Models

local Regularizers = script.Regularizers

local Optimizers = script.Optimizers

local ValueSchedulers = script.ValueSchedulers

local ExperienceReplays = script.ExperienceReplays

local QuickSetups = script.QuickSetups

local EligibilityTraces = script.EligibilityTraces

local ReinforcementLearningStrategies = script.ReinforcementLearningStrategies

local DistributedTrainingStrategies = script.DistributedTrainingStrategies

local Others = script.Others

AqwamMachineAndDeepLearningLibrary.Models = {

	LinearRegression = require(Models.LinearRegression),
	
	PassiveAggressiveRegressor = require(Models.PassiveAggressiveRegressor),
	
	NormalLinearRegression = require(Models.NormalLinearRegression),
	
	SupportVectorRegression = require(Models.SupportVectorRegression),
	
	KNearestNeighboursRegressor = require(Models.KNearestNeighboursRegressor),
	
	LogisticRegression = require(Models.LogisticRegression),
	
	PassiveAggressiveClassifier = require(Models.PassiveAggressiveClassifier),
	
	OneClassPassiveAggressiveClassifier = require(Models.OneClassPassiveAggressiveClassifier),
	
	NeuralNetwork = require(Models.NeuralNetwork),
	
	OneClassSupportVectorMachine = require(Models.OneClassSupportVectorMachine),
	
	SupportVectorMachine = require(Models.SupportVectorMachine),
	
	NearestCentroid = require(Models.NearestCentroid),
	
	KNearestNeighboursClassifier = require(Models.KNearestNeighboursClassifier),
	
	GaussianNaiveBayes = require(Models.GaussianNaiveBayes),
	
	MultinomialNaiveBayes = require(Models.MultinomialNaiveBayes),
	
	BernoulliNaiveBayes = require(Models.BernoulliNaiveBayes),
	
	ComplementNaiveBayes = require(Models.ComplementNaiveBayes),
	
	CategoricalNaiveBayes = require(Models.CategoricalNaiveBayes),
	
	KMeans = require(Models.KMeans),
	
	FuzzyCMeans = require(Models.FuzzyCMeans),
	
	ExpectationMaximization = require(Models.ExpectationMaximization),
	
	AgglomerativeHierarchical = require(Models.AgglomerativeHierarchical),
	
	MeanShift = require(Models.MeanShift),
	
	DensityBasedSpatialClusteringOfApplicationsWithNoise = require(Models.DensityBasedSpatialClusteringOfApplicationsWithNoise),
	
	KMedoids = require(Models.KMedoids),
	
	AffinityPropagation = require(Models.AffinityPropagation),
	
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
	
	DeepMonteCarloControl = require(Models.DeepMonteCarloControl),

	DeepOffPolicyMonteCarloControl = require(Models.DeepOffPolicyMonteCarloControl),
	
	REINFORCE = require(Models.REINFORCE),
	
	VanillaPolicyGradient = require(Models.VanillaPolicyGradient),
	
	ActorCritic = require(Models.ActorCritic),
	
	SoftActorCritic = require(Models.SoftActorCritic),
	
	AdvantageActorCritic = require(Models.AdvantageActorCritic),
	
	ProximalPolicyOptimization = require(Models.ProximalPolicyOptimization),
	
	ProximalPolicyOptimizationClip = require(Models.ProximalPolicyOptimizationClip),
	
	DeepDeterministicPolicyGradient = require(Models.DeepDeterministicPolicyGradient),
	
	TwinDelayedDeepDeterministicPolicyGradient = require(Models.TwinDelayedDeepDeterministicPolicyGradient),
	
	TabularQLearning = require(Models.TabularQLearning),

	TabularStateActionRewardStateAction = require(Models.TabularStateActionRewardStateAction),
	
	TabularExpectedStateActionRewardStateAction = require(Models.TabularExpectedStateActionRewardStateAction),
	
	TabularMonteCarloControl = require(Models.TabularMonteCarloControl),
	
	TabularOffPolicyMonteCarloControl = require(Models.TabularOffPolicyMonteCarloControl),
	
	GenerativeAdversarialNetwork = require(Models.GenerativeAdversarialNetwork),
	
	ConditionalGenerativeAdversarialNetwork = require(Models.ConditionalGenerativeAdversarialNetwork),

	WassersteinGenerativeAdversarialNetwork = require(Models.WassersteinGenerativeAdversarialNetwork),

	ConditionalWassersteinGenerativeAdversarialNetwork = require(Models.ConditionalWassersteinGenerativeAdversarialNetwork),

}

AqwamMachineAndDeepLearningLibrary.Regularizers = {
	
	ElasticNet = require(Regularizers.ElasticNet),
	
	Lasso = require(Regularizers.Lasso),
	
	Ridge = require(Regularizers.Ridge),
	
}

AqwamMachineAndDeepLearningLibrary.Optimizers = {

	AdaptiveDelta = require(Optimizers.AdaptiveDelta),
	
	AdaptiveFactor = require(Optimizers.AdaptiveFactor),
	
	AdaptiveGradient = require(Optimizers.AdaptiveGradient),

	AdaptiveMomentEstimation = require(Optimizers.AdaptiveMomentEstimation),

	AdaptiveMomentEstimationMaximum = require(Optimizers.AdaptiveMomentEstimationMaximum),
	
	AdaptiveMomentEstimationWeightDecay = require(Optimizers.AdaptiveMomentEstimationWeightDecay),
	
	Gravity = require(Optimizers.Gravity),
	
	Momentum = require(Optimizers.Momentum),

	NesterovAcceleratedAdaptiveMomentEstimation = require(Optimizers.NesterovAcceleratedAdaptiveMomentEstimation),
	
	RectifiedAdaptiveMomentEstimation = require(Optimizers.RectifiedAdaptiveMomentEstimation),
	
	ResilientBackwardPropagation = require(Optimizers.ResilientBackwardPropagation),
	
	RootMeanSquarePropagation = require(Optimizers.RootMeanSquarePropagation),

}

AqwamMachineAndDeepLearningLibrary.ValueSchedulers = {
	
	Chained = require(ValueSchedulers.Chained),
	
	Constant = require(ValueSchedulers.Constant),
	
	CosineAnnealing = require(ValueSchedulers.CosineAnnealing),
	
	Exponential = require(ValueSchedulers.Exponential),
	
	InverseSquareRoot = require(ValueSchedulers.InverseSquareRoot),
	
	InverseTime = require(ValueSchedulers.InverseTime),
	
	Linear = require(ValueSchedulers.Linear),
	
	MultipleStep = require(ValueSchedulers.MultipleStep),
	
	Multiplicative = require(ValueSchedulers.Multiplicative),
	
	Polynomial = require(ValueSchedulers.Polynomial),
	
	Sequential = require(ValueSchedulers.Sequential),

	Step = require(ValueSchedulers.Step),

}

AqwamMachineAndDeepLearningLibrary.ExperienceReplays = {

	UniformExperienceReplay = require(ExperienceReplays.UniformExperienceReplay),

	PrioritizedExperienceReplay = require(ExperienceReplays.PrioritizedExperienceReplay),

	NStepExperienceReplay = require(ExperienceReplays.NStepExperienceReplay),

}

AqwamMachineAndDeepLearningLibrary.QuickSetups = {

	CategoricalPolicy = require(QuickSetups.CategoricalPolicy),

	DiagonalGaussianPolicy = require(QuickSetups.DiagonalGaussianPolicy)

}

AqwamMachineAndDeepLearningLibrary.EligibilityTraces = {
	
	AccumulatingTrace = require(EligibilityTraces.AccumulatingTrace),
	
	ReplacingTrace = require(EligibilityTraces.ReplacingTrace),
	
	DutchTrace = require(EligibilityTraces.DutchTrace),
	
}

AqwamMachineAndDeepLearningLibrary.ReinforcementLearningStrategies = {
	
	RandomNetworkDistillation = require(ReinforcementLearningStrategies.RandomNetworkDistillation),
	
	GenerativeAdversarialImitationLearning = require(ReinforcementLearningStrategies.GenerativeAdversarialImitationLearning),

	WassersteinGenerativeAdversarialImitationLearning = require(ReinforcementLearningStrategies.WassersteinGenerativeAdversarialImitationLearning),
	
}

AqwamMachineAndDeepLearningLibrary.DistributedTrainingStrategies = {
	
	DistributedGradientsCoordinator = require(DistributedTrainingStrategies.DistributedGradientsCoordinator),

	DistributedModelParametersCoordinator = require(DistributedTrainingStrategies.DistributedModelParametersCoordinator),
	
}

AqwamMachineAndDeepLearningLibrary.Others = {
	
	ModelTrainingModifier = require(Others.ModelTrainingModifier),
	
	ModelSafeguardWrapper = require(Others.ModelSafeguardWrapper),
	
	ModelParametersMerger = require(Others.ModelParametersMerger),

	ModelDatasetCreator = require(Others.ModelDatasetCreator),

	ModelChecker = require(Others.ModelChecker),
	
	OneVsAll = require(Others.OneVsAll),
	
	OneVsOne = require(Others.OneVsOne),

	OnlineLearning = require(Others.OnlineLearning),
	
	ConfusionMatrixCreator = require(Others.ConfusionMatrixCreator),

}

return AqwamMachineAndDeepLearningLibrary
