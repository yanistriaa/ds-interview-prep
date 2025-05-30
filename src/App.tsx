import { useState, useEffect } from "react";
import { Box, Button, Badge, Select } from "@chakra-ui/react";
import { Progress } from "@chakra-ui/progress";
import { 
  FiBookOpen, FiAward, FiChevronLeft, FiChevronRight, FiShuffle, 
  FiTarget, FiClock, FiCheckCircle, FiRotateCcw, FiRefreshCw, 
  FiEye, FiEyeOff 
} from "react-icons/fi";
import "@fontsource/inter";
// Remove or comment out the next line if it causes errors
// import "@fontsource/inter/variable.css";

const flashcards = [
  // Probability and Distributions
  {
    question: "Explain the difference between a probability mass function (PMF) and a probability density function (PDF).",
    answer: "A PMF is used for discrete random variables and gives the probability that a variable is exactly equal to a specific value. A PDF is used for continuous random variables and represents the relative likelihood of a variable to take on a given value. For PDFs, the probability of a single point is zero; you must integrate over an interval.",
    topic: "Probability and Distributions",
    difficulty: "Easy"
  },
  {
    question: "How do you determine if a dataset follows a particular distribution (e.g., normal)?",
    answer: "You can use visual tools like histograms, Q-Q plots, or statistical tests like the Shapiro-Wilk test, Anderson-Darling test, or Kolmogorov-Smirnov test.",
    topic: "Probability and Distributions",
    difficulty: "Medium"
  },
  {
    question: "What is the Central Limit Theorem, and why is it important in data science?",
    answer: "The Central Limit Theorem states that the sampling distribution of the sample mean approaches a normal distribution as the sample size grows, regardless of the original population's distribution. It is important in data science because it allows us to make inferences about population parameters using normal distribution assumptions, even when the population itself is not normally distributed.",
    topic: "Probability and Distributions",
    difficulty: "Medium"
  },
  {
    question: "How would you handle a situation where the underlying data distribution is heavily skewed?",
    answer: "You can apply data transformations (e.g., log, square root), use non-parametric methods, or use robust statistical techniques that are less sensitive to skewed distributions.",
    topic: "Probability and Distributions",
    difficulty: "Medium"
  },
  {
    question: "What's the difference between Bayes' Theorem and Frequentist probability?",
    answer: "Bayes' Theorem uses prior probabilities and updates them with observed data. Frequentist probability defines probability as the long-run frequency of events and does not use prior information.",
    topic: "Probability and Distributions",
    difficulty: "Medium"
  },
  {
    question: "What is a moment-generating function, and how is it used in deriving properties of distributions?",
    answer: "A moment-generating function (MGF) is a function that generates all moments of a probability distribution. It's defined as M(t) = E[e^(tX)]. MGFs are useful for identifying distributions, deriving moments, and proving convergence theorems like the Central Limit Theorem.",
    topic: "Probability and Distributions",
    difficulty: "Hard"
  },
  {
    question: "Explain how and when you would use the Law of Iterated Expectations.",
    answer: "The Law of Iterated Expectations states that E[E[Y|X]] = E[Y]. It's used when you have nested conditional expectations, particularly in econometrics and finance for modeling hierarchical data structures or when dealing with partial information.",
    topic: "Probability and Distributions",
    difficulty: "Hard"
  },
  {
    question: "How can copulas be used to model dependencies between variables beyond correlation?",
    answer: "Copulas separate the marginal distributions from the dependence structure, allowing modeling of complex dependencies like tail dependence, asymmetric relationships, and non-linear associations that correlation cannot capture. They're particularly useful in risk management and multivariate modeling.",
    topic: "Probability and Distributions",
    difficulty: "Hard"
  },
  
  // Hypothesis Testing and Statistical Inference
  {
    question: "Walk me through the steps of designing and interpreting an A/B test.",
    answer: "1. Define the objective. 2. Formulate hypotheses. 3. Determine sample size. 4. Randomly assign groups. 5. Run the test and collect data. 6. Analyze using statistical methods (e.g., t-test). 7. Draw conclusions and assess business impact.",
    topic: "Hypothesis Testing and Statistical Inference",
    difficulty: "Medium"
  },
  {
    question: "What are Type I and Type II errors? How do they relate to alpha and beta levels?",
    answer: "Type I error is rejecting a true null hypothesis (false positive), related to the alpha level. Type II error is failing to reject a false null hypothesis (false negative), related to the beta level. Reducing one typically increases the other.",
    topic: "Hypothesis Testing and Statistical Inference",
    difficulty: "Easy"
  },
  {
    question: "What factors influence the power of a statistical test?",
    answer: "Sample size, effect size, significance level (alpha), and variance in the data all affect statistical power.",
    topic: "Hypothesis Testing and Statistical Inference",
    difficulty: "Medium"
  },
  {
    question: "How would you adjust for multiple hypothesis testing in a large-scale experiment?",
    answer: "You can use corrections like the Bonferroni correction, Holm's method, or control the false discovery rate using the Benjamini-Hochberg procedure.",
    topic: "Hypothesis Testing and Statistical Inference",
    difficulty: "Hard"
  },
  {
    question: "Explain p-value. Can a low p-value guarantee practical significance?",
    answer: "A p-value measures the probability of obtaining results at least as extreme as observed, under the null hypothesis. A low p-value suggests statistical significance but does not imply practical importance.",
    topic: "Hypothesis Testing and Statistical Inference",
    difficulty: "Easy"
  },
  {
    question: "How do you adjust p-values in the context of sequential A/B tests or continuous monitoring?",
    answer: "Use sequential testing methods like alpha spending functions, group sequential designs, or always-valid p-values. These control Type I error inflation that occurs when peeking at data multiple times.",
    topic: "Hypothesis Testing and Statistical Inference",
    difficulty: "Hard"
  },
  {
    question: "When would you use a non-inferiority or equivalence test instead of a traditional hypothesis test?",
    answer: "Non-inferiority tests are used when you want to show a new treatment is not worse than standard by more than a margin. Equivalence tests show two treatments are similar within acceptable bounds. Common in pharmaceutical trials and quality control.",
    topic: "Hypothesis Testing and Statistical Inference",
    difficulty: "Hard"
  },
  
  // Estimation and Confidence
  {
    question: "What's the difference between point estimates and interval estimates?",
    answer: "A point estimate gives a single value as an estimate of a population parameter. An interval estimate provides a range (like a confidence interval) that is likely to contain the population parameter.",
    topic: "Estimation and Confidence",
    difficulty: "Easy"
  },
  {
    question: "How would you construct a confidence interval, and what does it mean in practical terms?",
    answer: "You calculate the point estimate and add/subtract the margin of error based on the standard error and critical value. It means we are confident the interval contains the population parameter a certain percent of the time (e.g., 95%).",
    topic: "Estimation and Confidence",
    difficulty: "Medium"
  },
  {
    question: "When would you use bootstrapping, and how does it work?",
    answer: "Bootstrapping is used when the theoretical distribution of a statistic is unknown. It works by repeatedly resampling the data with replacement and calculating the statistic of interest.",
    topic: "Estimation and Confidence",
    difficulty: "Medium"
  },
  {
    question: "Describe Maximum Likelihood Estimation (MLE) and its applications in modeling.",
    answer: "MLE finds parameter values that maximize the likelihood of the observed data under a statistical model. It is widely used in fitting parametric models.",
    topic: "Estimation and Confidence",
    difficulty: "Medium"
  },
  {
    question: "What's the difference between standard error and standard deviation?",
    answer: "Standard deviation measures variability within a sample or population. Standard error measures the variability of a sample statistic (like the mean) from sample to sample.",
    topic: "Estimation and Confidence",
    difficulty: "Easy"
  },
  {
    question: "What's the difference between a 95% confidence interval and a 95% Bayesian credible interval?",
    answer: "A confidence interval has a frequentist interpretation: 95% of such intervals will contain the true parameter. A credible interval has a Bayesian interpretation: there's a 95% probability the parameter lies within the interval given the data.",
    topic: "Estimation and Confidence",
    difficulty: "Medium"
  },
  {
    question: "How does the jackknife resampling technique compare to bootstrap?",
    answer: "Jackknife systematically leaves out one observation at a time to estimate bias and variance. Bootstrap randomly resamples with replacement. Bootstrap is more flexible and generally preferred, while jackknife is simpler and works well for bias correction.",
    topic: "Estimation and Confidence",
    difficulty: "Medium"
  },
  
  // Regression and Modeling Assumptions
  {
    question: "What are the assumptions behind linear regression? How would you test them?",
    answer: "Assumptions include linearity, independence, homoscedasticity, normality of residuals, and no multicollinearity. These can be tested via residual plots, Durbin-Watson test, VIF, and Q-Q plots.",
    topic: "Regression and Modeling Assumptions",
    difficulty: "Medium"
  },
  {
    question: "How do multicollinearity and heteroskedasticity affect regression results?",
    answer: "Multicollinearity inflates standard errors and makes coefficient estimates unreliable. Heteroskedasticity affects the efficiency of estimates and validity of hypothesis tests.",
    topic: "Regression and Modeling Assumptions",
    difficulty: "Medium"
  },
  {
    question: "What is the difference between L1 and L2 regularization in regression?",
    answer: "L1 regularization (Lasso) adds absolute value of coefficients to the loss function and can produce sparse models. L2 regularization (Ridge) adds squared values and generally leads to smaller coefficients without necessarily setting them to zero.",
    topic: "Regression and Modeling Assumptions",
    difficulty: "Medium"
  },
  {
    question: "How would you interpret coefficients in a logistic regression model?",
    answer: "Coefficients represent the log odds of the outcome for a one-unit increase in the predictor. Exponentiating gives the odds ratio.",
    topic: "Regression and Modeling Assumptions",
    difficulty: "Easy"
  },
  {
    question: "When would you prefer a non-parametric method over a parametric one?",
    answer: "When the assumptions of parametric methods (like normality or homoscedasticity) are violated, or when the data is ordinal, non-parametric methods are more robust.",
    topic: "Regression and Modeling Assumptions",
    difficulty: "Medium"
  },
  {
    question: "How would you statistically assess whether your regression model is overfitting?",
    answer: "Use cross-validation, compare training vs. validation performance, examine learning curves, use information criteria (AIC/BIC), or perform statistical tests for nested models.",
    topic: "Regression and Modeling Assumptions",
    difficulty: "Medium"
  },
  {
    question: "What techniques exist to make linear models more robust to outliers?",
    answer: "Use robust regression methods like Huber regression, M-estimators, or quantile regression. You can also use weighted least squares, winsorization, or robust standard errors.",
    topic: "Regression and Modeling Assumptions",
    difficulty: "Medium"
  },
  
  // Bayesian Statistics
  {
    question: "Contrast Bayesian and Frequentist approaches to statistical inference.",
    answer: "Frequentist methods rely on long-run frequencies and don't use prior information. Bayesian methods incorporate prior beliefs and update them with observed data using Bayes' theorem.",
    topic: "Bayesian Statistics",
    difficulty: "Medium"
  },
  {
    question: "How do you interpret a posterior distribution in a Bayesian model?",
    answer: "It represents the updated beliefs about the model parameters after considering the prior distribution and the observed data.",
    topic: "Bayesian Statistics",
    difficulty: "Medium"
  },
  {
    question: "In what situations would you prefer a Bayesian approach in real-world data science projects?",
    answer: "Bayesian methods are preferable when prior knowledge is available, when data is limited, or when you want probabilistic interpretations of parameters.",
    topic: "Bayesian Statistics",
    difficulty: "Medium"
  },
  {
    question: "Describe the role of priors in Bayesian inference. How do you choose them?",
    answer: "Priors express beliefs about parameters before observing data. They can be informative (based on prior knowledge) or non-informative (to let data dominate). Choice depends on context and domain knowledge.",
    topic: "Bayesian Statistics",
    difficulty: "Hard"
  },
  {
    question: "What is MCMC, and how is it used in Bayesian computation?",
    answer: "MCMC (Markov Chain Monte Carlo) is a method to approximate posterior distributions by generating samples when analytical solutions are intractable.",
    topic: "Bayesian Statistics",
    difficulty: "Hard"
  },
  {
    question: "How does empirical Bayes differ from full Bayesian analysis, and when is it useful?",
    answer: "Empirical Bayes estimates hyperparameters from data rather than specifying them a priori. It's useful when you have many similar problems (like in multiple testing) and want to borrow strength across them.",
    topic: "Bayesian Statistics",
    difficulty: "Medium"
  },
  {
    question: "How do you compare two Bayesian models? Discuss Bayes factors or WAIC/LOO.",
    answer: "Bayes factors compare marginal likelihoods of models. WAIC (Widely Applicable Information Criterion) and LOO (Leave-One-Out cross-validation) are more practical approaches that balance fit and complexity for model comparison.",
    topic: "Bayesian Statistics",
    difficulty: "Hard"
  },
  
  // Real-World Statistical Decision Making
  {
    question: "How would you decide whether a model is 'good enough' to deploy using statistical reasoning?",
    answer: "Use statistical metrics such as confidence intervals for accuracy, hypothesis testing for significance, error analysis, and consider practical significance and business constraints.",
    topic: "Real-World Statistical Decision Making",
    difficulty: "Medium"
  },
  {
    question: "Describe how you'd use statistical methods to make pricing decisions in a product.",
    answer: "Analyze demand elasticity, perform hypothesis testing on pricing strategies, use regression models to estimate impact, and A/B test different prices.",
    topic: "Real-World Statistical Decision Making",
    difficulty: "Medium"
  },
  {
    question: "What's the role of statistical thinking in detecting data drift over time?",
    answer: "Statistical tools like hypothesis testing, population stability index (PSI), and control charts can help identify shifts in data distributions or model inputs.",
    topic: "Real-World Statistical Decision Making",
    difficulty: "Medium"
  },
  {
    question: "If two models show similar accuracy, how would you use statistical reasoning to choose between them?",
    answer: "Compare variance, consistency over time, confidence intervals, complexity, interpretability, and impact on business metrics.",
    topic: "Real-World Statistical Decision Making",
    difficulty: "Medium"
  },
  {
    question: "How would you design a statistical monitoring system for a production ML model?",
    answer: "Monitor input data distributions, output predictions, and performance metrics. Use statistical tests and control charts to detect drift or anomalies.",
    topic: "Real-World Statistical Decision Making",
    difficulty: "Hard"
  },
  {
    question: "How would you distinguish correlation from causation in observational data?",
    answer: "Use causal inference methods like instrumental variables, regression discontinuity, difference-in-differences, propensity score matching, or randomized controlled trials when possible.",
    topic: "Real-World Statistical Decision Making",
    difficulty: "Hard"
  },
  {
    question: "Describe Simpson's Paradox. How can it impact statistical decision making?",
    answer: "Simpson's Paradox occurs when a trend appears in subgroups but reverses when groups are combined. It highlights the importance of considering confounding variables and can lead to incorrect conclusions if not properly addressed.",
    topic: "Real-World Statistical Decision Making",
    difficulty: "Medium"
  },
  
  // Advanced Topics
  {
    question: "Explain the concept of statistical sufficiency and its practical implications.",
    answer: "A statistic is sufficient if it captures all necessary information about a parameter from the data. Sufficient statistics simplify inference and reduce data complexity.",
    topic: "Advanced Topics",
    difficulty: "Hard"
  },
  {
    question: "What is the Curse of Dimensionality, and how does it relate to statistical inference?",
    answer: "As the number of features increases, the data becomes sparse, making statistical inference and learning harder. It affects distance metrics, model overfitting, and variance in estimators.",
    topic: "Advanced Topics",
    difficulty: "Hard"
  },
  {
    question: "How do you apply dimensionality reduction techniques like PCA from a statistical standpoint?",
    answer: "PCA identifies orthogonal components that capture the most variance. It is used to reduce noise and multicollinearity while preserving structure in the data.",
    topic: "Advanced Topics",
    difficulty: "Medium"
  },
  {
    question: "What's the relationship between entropy in information theory and statistical uncertainty?",
    answer: "Entropy quantifies the amount of uncertainty or information in a probability distribution. Higher entropy means more unpredictability.",
    topic: "Advanced Topics",
    difficulty: "Hard"
  },
  {
    question: "How would you use KL divergence or Jensen-Shannon divergence in evaluating model predictions?",
    answer: "These measure the distance between probability distributions. They are used to compare predicted vs. true distributions, useful in classification and generative modeling.",
    topic: "Advanced Topics",
    difficulty: "Hard"
  },
  {
    question: "How do concepts like VC dimension or Rademacher complexity relate to model generalization?",
    answer: "VC dimension measures the capacity of a model class - higher VC dimension means more complex models that can overfit. Rademacher complexity provides data-dependent bounds on generalization error, helping understand when models will generalize well.",
    topic: "Advanced Topics",
    difficulty: "Hard"
  },
  {
    question: "What is a concentration inequality (e.g., Hoeffding's or Chebyshev's)? How does it guide sample size decisions?",
    answer: "Concentration inequalities provide bounds on how much a random variable deviates from its expectation. They help determine sample sizes needed to achieve desired confidence levels and precision in estimation.",
    topic: "Advanced Topics",
    difficulty: "Hard"
  },
  {
    question: "Can you explain how Bayesian networks can be used to model and infer causal relationships?",
      answer: "Bayesian networks represent conditional dependencies between variables through directed acyclic graphs. While they don't automatically imply causation, when combined with causal assumptions and interventional data, they can model causal relationships and support causal inference.",
      topic: "Advanced Topics",
      difficulty: "Hard"
    },
    // Machine Learning Algorithms
  {
    question: "What is the bias-variance trade-off in machine learning?",
    answer: "The bias-variance trade-off is the balance between underfitting (high bias) and overfitting (high variance). High bias means the model is too simple, while high variance means the model is too complex. The goal is to minimize total error from both sources.",
    topic: "Machine Learning Algorithms",
    difficulty: "Easy"
  },
  {
    question: "Compare and contrast bagging and boosting.",
    answer: "Bagging trains multiple models in parallel on different subsets of data and averages their predictions to reduce variance. Boosting trains models sequentially, focusing on errors made by previous models, and aims to reduce both bias and variance.",
    topic: "Machine Learning Algorithms",
    difficulty: "Medium"
  },
  {
    question: "When would you use a decision tree over a linear model?",
    answer: "Use decision trees when the relationship between features and target is non-linear, when interpretability is important, or when the data includes categorical variables without preprocessing.",
    topic: "Machine Learning Algorithms",
    difficulty: "Medium"
  },
  {
    question: "What are support vectors in Support Vector Machines (SVMs)?",
    answer: "Support vectors are the data points closest to the decision boundary that influence the position and orientation of the hyperplane in an SVM.",
    topic: "Machine Learning Algorithms",
    difficulty: "Medium"
  },
  {
    question: "What are some limitations of k-nearest neighbors (k-NN)?",
    answer: "k-NN can be computationally expensive at prediction time, is sensitive to the choice of k and distance metric, and suffers from the curse of dimensionality.",
    topic: "Machine Learning Algorithms",
    difficulty: "Hard"
  },
  
  // Feature Engineering
  {
    question: "What techniques can you use to handle high cardinality categorical features?",
    answer: "Techniques include target encoding, frequency encoding, hashing trick, and embedding representations (especially in deep learning models).",
    topic: "Feature Engineering",
    difficulty: "Medium"
  },
  {
    question: "How do you deal with multicollinearity in your features?",
    answer: "You can use techniques like variance inflation factor (VIF) to detect it and then drop or combine correlated features, or use regularization methods like Lasso.",
    topic: "Feature Engineering",
    difficulty: "Medium"
  },
  {
    question: "What is feature selection and why is it important?",
    answer: "Feature selection is the process of choosing relevant features to improve model performance, reduce overfitting, and decrease training time.",
    topic: "Feature Engineering",
    difficulty: "Easy"
  },
  {
    question: "Explain the difference between feature selection and dimensionality reduction.",
    answer: "Feature selection chooses a subset of existing features, while dimensionality reduction creates new features by combining existing ones (e.g., via PCA).",
    topic: "Feature Engineering",
    difficulty: "Medium"
  },
  {
    question: "What is one-hot encoding and when can it cause problems?",
    answer: "One-hot encoding converts categorical variables into binary vectors. It can cause issues with high-cardinality features due to memory and model complexity.",
    topic: "Feature Engineering",
    difficulty: "Easy"
  },
  
  // Model Evaluation and Metrics
  {
    question: "What is precision-recall trade-off and when is it important?",
    answer: "The trade-off is between correctly identifying positives (precision) and capturing all actual positives (recall). It’s crucial in imbalanced datasets like fraud detection or medical diagnosis.",
    topic: "Model Evaluation and Metrics",
    difficulty: "Medium"
  },
  {
    question: "Explain ROC and AUC in model evaluation.",
    answer: "ROC is a curve plotting true positive rate vs. false positive rate. AUC measures the area under this curve; a higher AUC indicates better model performance across all thresholds.",
    topic: "Model Evaluation and Metrics",
    difficulty: "Easy"
  },
  {
    question: "What is cross-validation and why is it used?",
    answer: "Cross-validation partitions data into training and validation sets multiple times to evaluate model performance reliably and prevent overfitting.",
    topic: "Model Evaluation and Metrics",
    difficulty: "Easy"
  },
  {
    question: "What are some challenges with evaluating models on imbalanced datasets?",
    answer: "Standard metrics like accuracy become misleading. You need metrics like precision, recall, F1-score, or AUC. Also, you may need resampling methods or custom loss functions.",
    topic: "Model Evaluation and Metrics",
    difficulty: "Medium"
  },
  {
    question: "What is log loss, and how does it differ from accuracy?",
    answer: "Log loss measures the uncertainty of predicted probabilities. Unlike accuracy, it penalizes wrong predictions more when they are confident but incorrect.",
    topic: "Model Evaluation and Metrics",
    difficulty: "Medium"
  },
  
  // Interpretability and Explainability
  {
    question: "What is SHAP and how is it used?",
    answer: "SHAP (SHapley Additive exPlanations) provides consistent, game-theoretic feature attribution values, explaining the contribution of each feature to individual predictions.",
    topic: "Interpretability and Explainability",
    difficulty: "Hard"
  },
  {
    question: "How do LIME and SHAP differ in model explainability?",
    answer: "LIME approximates the model locally with a simple interpretable model, while SHAP computes exact or approximate Shapley values, offering global consistency and additive feature attributions.",
    topic: "Interpretability and Explainability",
    difficulty: "Hard"
  },
  {
    question: "Why is model interpretability important in regulated industries?",
    answer: "Interpretability helps ensure compliance, detect bias, explain decisions to stakeholders, and build trust in high-stakes areas like finance and healthcare.",
    topic: "Interpretability and Explainability",
    difficulty: "Medium"
  },
  {
    question: "What is partial dependence plot (PDP)?",
    answer: "PDP shows the marginal effect of one or two features on the predicted outcome, averaged over the data, helping to visualize model behavior.",
    topic: "Interpretability and Explainability",
    difficulty: "Medium"
  },
  
  // Model Monitoring and Drift
  {
    question: "What is concept drift and how do you detect it?",
    answer: "Concept drift is a change in the underlying relationship between input and output over time. Detection methods include statistical tests, performance degradation monitoring, and drift detection algorithms like DDM or ADWIN.",
    topic: "Model Monitoring and Drift",
    difficulty: "Hard"
  },
  {
    question: "How can you monitor model performance in production?",
    answer: "Monitor metrics like accuracy, precision, and recall, compare distributions using statistical tests, and use dashboards to alert anomalies in data or predictions.",
    topic: "Model Monitoring and Drift",
    difficulty: "Medium"
  },
  {
    question: "What is data drift and how is it different from concept drift?",
    answer: "Data drift is a change in the distribution of input data, while concept drift refers to a change in the relationship between inputs and outputs. Data drift may or may not affect model performance.",
    topic: "Model Monitoring and Drift",
    difficulty: "Medium"
  },
  
  // Neural Networks and Deep Learning
  {
    question: "What is the vanishing gradient problem in deep neural networks?",
    answer: "It occurs when gradients become very small during backpropagation, especially in deep networks, causing slow or no learning. It can be mitigated with ReLU activations or better initialization.",
    topic: "Neural Networks and Deep Learning",
    difficulty: "Hard"
  },
  {
    question: "What is batch normalization and how does it help?",
    answer: "Batch normalization normalizes the inputs of each layer to stabilize and speed up training, reduce internal covariate shift, and allow higher learning rates.",
    topic: "Neural Networks and Deep Learning",
    difficulty: "Medium"
  },
  {
    question: "Explain the role of activation functions in neural networks.",
    answer: "Activation functions introduce non-linearity, enabling the network to model complex relationships. Examples include ReLU, sigmoid, and tanh.",
    topic: "Neural Networks and Deep Learning",
    difficulty: "Easy"
  },
  {
    question: "What are attention mechanisms in deep learning?",
    answer: "Attention mechanisms dynamically focus on relevant parts of input sequences, improving performance in tasks like machine translation and text summarization. It's a core component of transformers.",
    topic: "Neural Networks and Deep Learning",
    difficulty: "Hard"
  },
  {
    question: "What is transfer learning, and why is it useful?",
    answer: "Transfer learning involves using a pre-trained model on a new, related task. It reduces training time and data requirements and often improves performance, especially in image and NLP tasks.",
    topic: "Neural Networks and Deep Learning",
    difficulty: "Medium"
  },
  // Deep Learning and Neural Networks
{
  question: "What is the vanishing gradient problem, and how is it addressed in deep neural networks?",
  answer: "The vanishing gradient problem occurs when gradients become too small during backpropagation, making it difficult for earlier layers to learn. It is mitigated using activation functions like ReLU, batch normalization, and architectures like residual networks (ResNets).",
  topic: "Deep Learning and Neural Networks",
  difficulty: "Medium"
},
{
  question: "What are the main differences between CNNs and RNNs?",
  answer: "CNNs are designed for spatial data like images, using filters and pooling. RNNs are suited for sequential data like text or time series, maintaining a hidden state across time steps.",
  topic: "Deep Learning and Neural Networks",
  difficulty: "Easy"
},
{
  question: "Explain how dropout works and why it helps prevent overfitting.",
  answer: "Dropout randomly deactivates neurons during training to prevent co-adaptation and overfitting. It acts as an ensemble of sub-networks, improving generalization.",
  topic: "Deep Learning and Neural Networks",
  difficulty: "Medium"
},
{
  question: "What is the role of backpropagation in training neural networks?",
  answer: "Backpropagation computes gradients of the loss function with respect to each weight via the chain rule, enabling efficient gradient descent updates.",
  topic: "Deep Learning and Neural Networks",
  difficulty: "Easy"
},
{
  question: "Why are residual connections important in very deep networks?",
  answer: "Residual connections allow gradients to flow directly through identity paths, reducing vanishing gradient issues and enabling deeper architectures.",
  topic: "Deep Learning and Neural Networks",
  difficulty: "Medium"
},

// Transformers and Attention
{
  question: "What is the self-attention mechanism in transformers?",
  answer: "Self-attention computes a weighted representation of inputs by attending to all positions in the sequence. It helps the model capture dependencies regardless of distance.",
  topic: "Transformers and Attention",
  difficulty: "Medium"
},
{
  question: "How does positional encoding work in transformers?",
  answer: "Positional encoding injects information about token positions into input embeddings, enabling the model to capture order in the sequence since transformers lack recurrence.",
  topic: "Transformers and Attention",
  difficulty: "Medium"
},
{
  question: "What are the benefits of multi-head attention in transformer models?",
  answer: "Multi-head attention allows the model to attend to information from different representation subspaces, enhancing the model’s ability to capture complex relationships.",
  topic: "Transformers and Attention",
  difficulty: "Medium"
},
{
  question: "Why is Layer Normalization used in transformer architectures?",
  answer: "Layer Normalization stabilizes and accelerates training by normalizing activations across features, helping with gradient flow and convergence.",
  topic: "Transformers and Attention",
  difficulty: "Easy"
},
{
  question: "Compare encoder-only, decoder-only, and encoder-decoder transformer architectures.",
  answer: "Encoder-only models (e.g., BERT) are used for understanding tasks. Decoder-only models (e.g., GPT) are for generation. Encoder-decoder models (e.g., T5) are used for seq2seq tasks like translation.",
  topic: "Transformers and Attention",
  difficulty: "Medium"
},

// Large Language Models and Generative AI
{
  question: "What is a language model, and how is it used in generative AI?",
  answer: "A language model predicts the next token in a sequence, enabling text generation. In generative AI, it is used to produce coherent and contextually relevant content.",
  topic: "Large Language Models and Generative AI",
  difficulty: "Easy"
},
{
  question: "What are the challenges associated with training large language models?",
  answer: "Challenges include computational cost, memory requirements, data quality, training stability, and ethical issues like bias and hallucination.",
  topic: "Large Language Models and Generative AI",
  difficulty: "Hard"
},
{
  question: "Explain the concept of autoregressive vs. autoencoding models in NLP.",
  answer: "Autoregressive models predict the next token (e.g., GPT), while autoencoding models predict missing parts of a sequence or reconstruct input (e.g., BERT).",
  topic: "Large Language Models and Generative AI",
  difficulty: "Medium"
},
{
  question: "What is prompt engineering, and why is it important in using LLMs?",
  answer: "Prompt engineering involves crafting effective inputs to guide the output of LLMs. It’s crucial for controlling behavior and improving performance on specific tasks.",
  topic: "Large Language Models and Generative AI",
  difficulty: "Easy"
},
{
  question: "What is 'hallucination' in LLMs, and how can it be mitigated?",
  answer: "Hallucination refers to generating plausible but incorrect or fabricated content. Mitigation involves grounding models with retrieval-augmented generation (RAG), better datasets, and training objectives.",
  topic: "Large Language Models and Generative AI",
  difficulty: "Hard"
},

// Fine-Tuning and Adaptation
{
  question: "What is the difference between fine-tuning and pretraining?",
  answer: "Pretraining involves learning general language representations on large corpora. Fine-tuning adapts the model to a specific downstream task using labeled data.",
  topic: "Fine-Tuning and Adaptation",
  difficulty: "Easy"
},
{
  question: "What is parameter-efficient fine-tuning, and why is it used?",
  answer: "It fine-tunes a small subset of model parameters (e.g., LoRA, adapters) to reduce compute cost and avoid full model retraining.",
  topic: "Fine-Tuning and Adaptation",
  difficulty: "Medium"
},
{
  question: "How does LoRA (Low-Rank Adaptation) improve fine-tuning efficiency?",
  answer: "LoRA inserts trainable low-rank matrices into linear layers, enabling adaptation with fewer parameters and preserving the base model.",
  topic: "Fine-Tuning and Adaptation",
  difficulty: "Medium"
},
{
  question: "When would you use instruction tuning vs. RLHF for fine-tuning LLMs?",
  answer: "Instruction tuning aligns model behavior to human-readable tasks via curated prompts. RLHF (Reinforcement Learning with Human Feedback) optimizes output preferences using reward models. Use RLHF when alignment to subjective quality is needed.",
  topic: "Fine-Tuning and Adaptation",
  difficulty: "Hard"
},

// Quantization and Optimization
{
  question: "What is model quantization, and how does it help with deployment?",
  answer: "Quantization reduces the precision of weights and activations (e.g., from float32 to int8), decreasing model size and inference time while preserving accuracy.",
  topic: "Quantization and Optimization",
  difficulty: "Medium"
},
{
  question: "What are the main trade-offs of post-training quantization versus quantization-aware training?",
  answer: "Post-training quantization is fast but may degrade accuracy. Quantization-aware training includes quantization effects during training, preserving accuracy but requiring more effort.",
  topic: "Quantization and Optimization",
  difficulty: "Hard"
},
{
  question: "How does knowledge distillation improve model efficiency?",
  answer: "A smaller student model is trained to mimic a larger teacher model’s outputs, enabling efficient inference with similar performance.",
  topic: "Quantization and Optimization",
  difficulty: "Medium"
},
{
  question: "Why is mixed-precision training beneficial for large models?",
  answer: "Mixed-precision training uses lower precision (e.g., FP16) for faster computation and reduced memory usage, while maintaining stability with higher precision where needed.",
  topic: "Quantization and Optimization",
  difficulty: "Medium"
},
// MLOps and Deployment
{
  question: "What are the key steps in deploying a machine learning model into production?",
  answer: "The key steps include model serialization, containerization, creating inference APIs, monitoring, scaling, and integrating with CI/CD pipelines. Additionally, security, reproducibility, and governance need to be ensured.",
  topic: "MLOps and Deployment",
  difficulty: "Medium"
},
{
  question: "How does CI/CD improve the deployment of ML models?",
  answer: "CI/CD automates testing, validation, and deployment of code and models, ensuring consistent and reliable updates. It reduces manual errors, speeds up iteration, and ensures reproducibility of ML workflows.",
  topic: "MLOps and Deployment",
  difficulty: "Medium"
},
{
  question: "What is the difference between model deployment and model serving?",
  answer: "Model deployment is the process of making a model available in a production environment. Model serving refers specifically to hosting the model behind an API or service to respond to real-time or batch prediction requests.",
  topic: "MLOps and Deployment",
  difficulty: "Easy"
},
{
  question: "What are some common tools used in MLOps pipelines?",
  answer: "Common tools include MLflow, Kubeflow, Airflow, TFX, DVC, Git, Jenkins, Docker, Kubernetes, and cloud-native ML platforms like AWS SageMaker, Azure ML, or GCP Vertex AI.",
  topic: "MLOps and Deployment",
  difficulty: "Easy"
},
{
  question: "How do you monitor ML models in production?",
  answer: "You monitor prediction latency, data drift, model performance metrics, infrastructure health, and system logs. Tools like Prometheus, Grafana, Evidently AI, and custom dashboards are used.",
  topic: "MLOps and Deployment",
  difficulty: "Medium"
},
{
  question: "What is concept drift, and how do you detect and handle it?",
  answer: "Concept drift occurs when the statistical properties of the target variable change over time. It's detected through performance degradation, drift metrics, or statistical tests, and handled through retraining, alerts, or adaptive models.",
  topic: "MLOps and Deployment",
  difficulty: "Hard"
},
{
  question: "Explain the role of feature stores in MLOps.",
  answer: "Feature stores manage, version, and serve features consistently for training and inference. They ensure feature parity, enable reuse, and improve collaboration and reproducibility.",
  topic: "MLOps and Deployment",
  difficulty: "Medium"
},
{
  question: "What are blue-green and canary deployments in ML systems?",
  answer: "Blue-green deployment runs two environments (old and new) and switches traffic when the new version is ready. Canary deployment gradually rolls out the model to a subset of users to minimize risk.",
  topic: "MLOps and Deployment",
  difficulty: "Hard"
},
{
  question: "How would you implement rollback in case a deployed model fails?",
  answer: "Maintain versioned models in a registry, monitor performance metrics post-deployment, and configure the infrastructure to automatically or manually revert to a previous stable model version.",
  topic: "MLOps and Deployment",
  difficulty: "Hard"
},
{
  question: "What are shadow deployments and why are they useful?",
  answer: "Shadow deployments route real-time traffic to a new model without affecting production decisions. It helps validate model behavior and performance under real-world conditions before full deployment.",
  topic: "MLOps and Deployment",
  difficulty: "Medium"
},

// Infrastructure and Architecture
{
  question: "What are the benefits of containerizing ML models?",
  answer: "Containers encapsulate dependencies, ensure consistent environments across development and production, simplify scaling and deployment, and integrate well with orchestration tools like Kubernetes.",
  topic: "Infrastructure and Architecture",
  difficulty: "Easy"
},
{
  question: "What are the key components of a cloud-native ML architecture?",
  answer: "Key components include data ingestion pipelines, scalable storage, feature stores, model training services, model registry, CI/CD pipelines, and inference endpoints managed via container orchestration.",
  topic: "Infrastructure and Architecture",
  difficulty: "Medium"
},
{
  question: "How does Kubernetes help in managing ML workflows?",
  answer: "Kubernetes automates deployment, scaling, and management of containerized applications. In ML, it helps orchestrate distributed training jobs, deploy models, manage resources, and enable reproducibility.",
  topic: "Infrastructure and Architecture",
  difficulty: "Medium"
},
{
  question: "What is model versioning and why is it important?",
  answer: "Model versioning tracks different iterations of models, including changes in training data, hyperparameters, and architecture. It ensures reproducibility, auditability, and safe rollbacks.",
  topic: "Infrastructure and Architecture",
  difficulty: "Easy"
},
{
  question: "How do you design a scalable inference system for real-time predictions?",
  answer: "Use stateless model servers with autoscaling, asynchronous queues, load balancing, GPU acceleration (if needed), and caching mechanisms for frequently requested inputs.",
  topic: "Infrastructure and Architecture",
  difficulty: "Hard"
},
{
  question: "Describe an ideal architecture for batch inference at scale.",
  answer: "Use distributed data processing tools like Spark or Dask to read data from data lakes, run inference using model containers, and write results back to storage, orchestrated by Airflow or similar tools.",
  topic: "Infrastructure and Architecture",
  difficulty: "Hard"
},

// APIs and Interfaces
{
  question: "How do you expose a machine learning model via a REST API?",
  answer: "You can use frameworks like Flask, FastAPI, or TensorFlow Serving to wrap the model in an endpoint that receives input data, invokes prediction, and returns results over HTTP.",
  topic: "APIs and Interfaces",
  difficulty: "Easy"
},
{
  question: "What are some best practices when designing APIs for ML services?",
  answer: "Ensure input validation, standardize error responses, include versioning, limit request size, monitor latency, and secure the API with authentication and rate limiting.",
  topic: "APIs and Interfaces",
  difficulty: "Medium"
},
{
  question: "What’s the difference between REST APIs and gRPC in the context of ML serving?",
  answer: "REST APIs use HTTP and are widely supported but may be less efficient. gRPC uses binary protocol buffers, supports multiple languages, and provides better performance for high-throughput ML workloads.",
  topic: "APIs and Interfaces",
  difficulty: "Medium"
},
{
  question: "How do you secure a model serving API in production?",
  answer: "Use HTTPS, API keys or OAuth2 for authentication, rate limiting, IP whitelisting, and monitoring for abuse or anomalies. Also consider encrypting payloads and using gateways.",
  topic: "APIs and Interfaces",
  difficulty: "Medium"
},
// Database Management and SQL Concepts
{
  question: "What is the difference between a relational and a non-relational database?",
  answer: "Relational databases store data in tables with predefined schemas and use SQL for querying. Non-relational (NoSQL) databases store data in various formats (document, key-value, graph, column-family) and offer more flexible schemas.",
  topic: "Database Management and SQL Concepts",
  difficulty: "Easy"
},
{
  question: "Explain the concept of normalization and why it's important.",
  answer: "Normalization is the process of organizing data to reduce redundancy and improve data integrity. It involves decomposing tables into smaller related tables and defining relationships between them.",
  topic: "Database Management and SQL Concepts",
  difficulty: "Medium"
},
{
  question: "What are the different types of JOINs in SQL?",
  answer: "The main types are INNER JOIN, LEFT JOIN, RIGHT JOIN, and FULL OUTER JOIN. They determine how rows from two tables are combined based on matching conditions.",
  topic: "Database Management and SQL Concepts",
  difficulty: "Easy"
},
{
  question: "What is a primary key, and how is it different from a foreign key?",
  answer: "A primary key uniquely identifies each record in a table. A foreign key is a field that links to the primary key in another table, creating a relationship between the two tables.",
  topic: "Database Management and SQL Concepts",
  difficulty: "Easy"
},
{
  question: "When would you denormalize a database, and why?",
  answer: "Denormalization involves combining tables to improve read performance, especially in OLAP systems or when join operations become a bottleneck.",
  topic: "Database Management and SQL Concepts",
  difficulty: "Medium"
},
{
  question: "How do indexes work in SQL, and what are the trade-offs of using them?",
  answer: "Indexes speed up data retrieval but add overhead for insert, update, and delete operations. They use data structures like B-trees or hash tables.",
  topic: "Database Management and SQL Concepts",
  difficulty: "Medium"
},
{
  question: "What is ACID compliance in databases?",
  answer: "ACID stands for Atomicity, Consistency, Isolation, and Durability. These properties ensure reliable transaction processing in relational databases.",
  topic: "Database Management and SQL Concepts",
  difficulty: "Medium"
},
{
  question: "Explain the difference between OLTP and OLAP systems.",
  answer: "OLTP (Online Transaction Processing) systems handle real-time transactional workloads, while OLAP (Online Analytical Processing) systems support complex queries for analytics and decision-making.",
  topic: "Database Management and SQL Concepts",
  difficulty: "Medium"
},
{
  question: "What are stored procedures and how are they used?",
  answer: "Stored procedures are precompiled SQL code stored in the database. They can encapsulate business logic and improve performance by reducing client-server communication.",
  topic: "Database Management and SQL Concepts",
  difficulty: "Medium"
},
{
  question: "How does a database handle concurrent access to the same data?",
  answer: "Databases use isolation levels and locking mechanisms (e.g., row-level locks, MVCC) to manage concurrent access and maintain data consistency.",
  topic: "Database Management and SQL Concepts",
  difficulty: "Hard"
},
{
  question: "What is a transaction, and how do you ensure it's successful?",
  answer: "A transaction is a sequence of operations that must be completed together. Success is ensured by satisfying ACID properties and using COMMIT or ROLLBACK statements.",
  topic: "Database Management and SQL Concepts",
  difficulty: "Easy"
},
{
  question: "Describe the different types of NoSQL databases.",
  answer: "NoSQL databases include document stores (e.g., MongoDB), key-value stores (e.g., Redis), column-family stores (e.g., Cassandra), and graph databases (e.g., Neo4j). Each serves different use cases.",
  topic: "Database Management and SQL Concepts",
  difficulty: "Medium"
},
{
  question: "What is sharding in databases and why is it used?",
  answer: "Sharding is a database partitioning technique that distributes data across multiple servers to improve performance and scalability.",
  topic: "Database Management and SQL Concepts",
  difficulty: "Hard"
},
{
  question: "Explain the CAP theorem in the context of distributed databases.",
  answer: "The CAP theorem states that a distributed database can provide only two of the following three guarantees at the same time: Consistency, Availability, and Partition Tolerance.",
  topic: "Database Management and SQL Concepts",
  difficulty: "Hard"
},
{
  question: "How can you optimize a slow SQL query?",
  answer: "Use EXPLAIN plans, add indexes, avoid SELECT *, reduce subqueries, and consider query refactoring or caching strategies.",
  topic: "Database Management and SQL Concepts",
  difficulty: "Medium"
},
{
  question: "What is the difference between DELETE, TRUNCATE, and DROP?",
  answer: "DELETE removes rows one at a time and can be rolled back. TRUNCATE removes all rows quickly and cannot be rolled back. DROP removes the entire table definition and data.",
  topic: "Database Management and SQL Concepts",
  difficulty: "Easy"
},
{
  question: "What is a view in SQL and when would you use one?",
  answer: "A view is a virtual table based on the result of a SQL query. It's used for abstraction, security, or simplifying complex queries.",
  topic: "Database Management and SQL Concepts",
  difficulty: "Easy"
},
{
  question: "Describe the purpose of database normalization forms (1NF, 2NF, 3NF).",
  answer: "These forms eliminate redundancy and dependencies: 1NF ensures atomic columns, 2NF removes partial dependencies, and 3NF removes transitive dependencies.",
  topic: "Database Management and SQL Concepts",
  difficulty: "Medium"
},
{
  question: "How does replication improve database availability?",
  answer: "Replication copies data from one server to others, enabling failover in case of failure and distributing read loads for better performance.",
  topic: "Database Management and SQL Concepts",
  difficulty: "Medium"
},
{
  question: "What is the purpose of a data warehouse?",
  answer: "A data warehouse stores large volumes of historical data optimized for analytical queries, reporting, and business intelligence.",
  topic: "Database Management and SQL Concepts",
  difficulty: "Medium"
},
{
  question: "What is a foreign key constraint and how does it enforce referential integrity?",
  answer: "A foreign key constraint ensures that a value in one table matches a primary key in another, enforcing consistent relationships between tables.",
  topic: "Database Management and SQL Concepts",
  difficulty: "Easy"
},
{
  question: "How does eventual consistency differ from strong consistency?",
  answer: "Eventual consistency allows temporary inconsistency across distributed nodes, while strong consistency ensures all nodes reflect the latest data at all times.",
  topic: "Database Management and SQL Concepts",
  difficulty: "Hard"
},
{
  question: "What is a composite key, and when would you use it?",
  answer: "A composite key is a combination of two or more columns used to uniquely identify a row in a table. It's used when no single column is unique by itself.",
  topic: "Database Management and SQL Concepts",
  difficulty: "Easy"
},
{
  question: "How can you handle schema evolution in NoSQL databases?",
  answer: "You can handle schema evolution by designing flexible data models, versioning documents, and using application logic to manage differences.",
  topic: "Database Management and SQL Concepts",
  difficulty: "Hard"
},
{
  question: "What are common pitfalls in database design and how can they be avoided?",
  answer: "Common pitfalls include poor normalization, lack of indexing, incorrect data types, and failing to plan for growth. Avoid them through careful planning, performance testing, and design reviews.",
  topic: "Database Management and SQL Concepts",
  difficulty: "Medium"
},
{
  question: "Explain the difference between synchronous and asynchronous replication.",
  answer: "Synchronous replication ensures data is written to all replicas before confirming success. Asynchronous replication confirms success once the primary write completes, reducing latency but risking data loss on failure.",
  topic: "Database Management and SQL Concepts",
  difficulty: "Hard"
},
// Data & AI Tools - Cloud Platforms, Orchestration, and DevOps
{
  question: "What are the key differences between IaaS, PaaS, and SaaS in cloud computing?",
  answer: "IaaS provides virtualized hardware resources (e.g., VMs, storage); PaaS offers a platform for developers to build applications without managing the underlying infrastructure; SaaS delivers ready-to-use applications over the internet.",
  topic: "Data & AI Tools",
  difficulty: "Easy"
},
{
  question: "What is AWS Lambda, and when should you use it?",
  answer: "AWS Lambda is a serverless compute service that lets you run code without provisioning servers. Use it for event-driven workloads, microservices, and lightweight backend processing.",
  topic: "Data & AI Tools",
  difficulty: "Medium"
},
{
  question: "Compare Amazon S3, Azure Blob Storage, and Google Cloud Storage.",
  answer: "All three offer object storage solutions, but differ in features like IAM integration, lifecycle policies, and regional availability. S3 supports strong consistency, Azure has tiered access levels, and GCP emphasizes seamless ML integration.",
  topic: "Data & AI Tools",
  difficulty: "Medium"
},
{
  question: "How does Databricks unify data engineering and data science workflows?",
  answer: "Databricks integrates Apache Spark with collaborative notebooks, Delta Lake for ACID-compliant data lakes, and MLflow for ML lifecycle management—supporting both batch and streaming workloads.",
  topic: "Data & AI Tools",
  difficulty: "Medium"
},
{
  question: "What is Delta Lake and how does it enhance traditional data lakes?",
  answer: "Delta Lake adds ACID transactions, schema enforcement, and time travel to data lakes, making them more reliable and suitable for production-grade data pipelines.",
  topic: "Data & AI Tools",
  difficulty: "Medium"
},
{
  question: "Explain the architecture of Snowflake and why it's considered a cloud-native data warehouse.",
  answer: "Snowflake separates storage, compute, and services layers. It uses virtual warehouses for elastic compute, stores data in a centralized storage layer, and is designed for multi-cloud environments.",
  topic: "Data & AI Tools",
  difficulty: "Medium"
},
{
  question: "What is Time Travel in Snowflake and how is it useful?",
  answer: "Time Travel lets users query historical data, recover deleted records, and perform audits by accessing data as it existed at a previous time.",
  topic: "Data & AI Tools",
  difficulty: "Medium"
},
{
  question: "How do Docker containers differ from traditional virtual machines?",
  answer: "Docker containers share the host OS kernel, making them more lightweight and faster to start than VMs, which emulate entire operating systems.",
  topic: "Data & AI Tools",
  difficulty: "Easy"
},
{
  question: "What is a Dockerfile and what is it used for?",
  answer: "A Dockerfile is a script that contains instructions to build a Docker image. It defines the base image, software packages, environment variables, and commands to run.",
  topic: "Data & AI Tools",
  difficulty: "Easy"
},
{
  question: "How do you manage container orchestration using Kubernetes?",
  answer: "Kubernetes automates deployment, scaling, and management of containerized applications using components like pods, deployments, services, and config maps.",
  topic: "Data & AI Tools",
  difficulty: "Medium"
},
{
  question: "What are the roles of ConfigMaps and Secrets in Kubernetes?",
  answer: "ConfigMaps store non-sensitive configuration data, while Secrets store sensitive data like API keys. Both can be injected into pods as environment variables or mounted volumes.",
  topic: "Data & AI Tools",
  difficulty: "Medium"
},
{
  question: "Describe the difference between Git fetch and Git pull.",
  answer: "Git fetch retrieves updates from a remote repository without merging them, while Git pull fetches and then merges the changes into your current branch.",
  topic: "Data & AI Tools",
  difficulty: "Easy"
},
{
  question: "What is Git rebase and how does it differ from merge?",
  answer: "Git rebase moves or combines a sequence of commits to a new base commit, producing a linear history. Merge preserves the history but creates a new merge commit.",
  topic: "Data & AI Tools",
  difficulty: "Medium"
},
{
  question: "How does Airflow manage task dependencies and workflow scheduling?",
  answer: "Airflow uses Directed Acyclic Graphs (DAGs) to define workflows. Task dependencies are explicitly set, and scheduling is controlled via cron expressions or time intervals.",
  topic: "Data & AI Tools",
  difficulty: "Medium"
},
{
  question: "What are Airflow sensors and when should you use them?",
  answer: "Sensors are specialized Airflow operators that wait for a condition to be met before proceeding. They're used for checking external file existence, database rows, or API status.",
  topic: "Data & AI Tools",
  difficulty: "Medium"
},
{
  question: "What is Collibra and how does it support data governance?",
  answer: "Collibra is a data intelligence platform that enables data cataloging, stewardship, lineage, and compliance, helping organizations manage metadata and enforce governance policies.",
  topic: "Data & AI Tools",
  difficulty: "Medium"
},
{
  question: "Explain the concept of data lineage and its importance.",
  answer: "Data lineage tracks the flow of data from source to destination, showing transformations along the way. It’s critical for debugging, compliance, and impact analysis.",
  topic: "Data & AI Tools",
  difficulty: "Easy"
},
{
  question: "How does Tableau enable interactive data exploration?",
  answer: "Tableau allows users to build dashboards with drag-and-drop interfaces, create calculated fields, apply filters, and interactively explore data through visualizations.",
  topic: "Data & AI Tools",
  difficulty: "Easy"
},
{
  question: "What are Tableau extracts and when should you use them?",
  answer: "Extracts are snapshots of data optimized for fast performance. Use them when querying live data is slow or when you need offline access.",
  topic: "Data & AI Tools",
  difficulty: "Medium"
},
{
  question: "What is ELT and how does it differ from ETL?",
  answer: "In ELT, data is loaded into the destination system before transformations occur (common in cloud data warehouses). ETL transforms data before loading it. ELT leverages scalable compute in modern platforms.",
  topic: "Data & AI Tools",
  difficulty: "Easy"
},
{
  question: "How would you ensure CI/CD for data pipelines?",
  answer: "Use version control (e.g., Git), automated testing, containerization (Docker), orchestration tools (Airflow, dbt), and CI/CD tools like GitHub Actions, Jenkins, or Azure DevOps.",
  topic: "Data & AI Tools",
  difficulty: "Hard"
},
{
  question: "What is dbt (data build tool) and how does it help with transformation logic?",
  answer: "dbt allows analysts to write modular SQL transformations, manage dependencies, test data, and document models, all within version-controlled repositories.",
  topic: "Data & AI Tools",
  difficulty: "Medium"
},
{
  question: "Explain how caching works in Snowflake and its effect on performance.",
  answer: "Snowflake uses result, metadata, and data cache to reduce query time. Caches persist within virtual warehouses and across sessions, speeding up repeated queries.",
  topic: "Data & AI Tools",
  difficulty: "Medium"
},
{
  question: "What is the purpose of using Helm in Kubernetes environments?",
  answer: "Helm is a package manager for Kubernetes that simplifies application deployment and management by packaging configurations into charts.",
  topic: "Data & AI Tools",
  difficulty: "Medium"
},
{
  question: "What are Git hooks and how can they be used for enforcing code quality?",
  answer: "Git hooks are scripts triggered by Git events (e.g., pre-commit, pre-push) and can enforce formatting, run tests, or check secrets before code changes are committed or pushed.",
  topic: "Data & AI Tools",
  difficulty: "Hard"
},
{
  question: "How does role-based access control (RBAC) work in Kubernetes?",
  answer: "RBAC defines roles and permissions, binding them to users or service accounts. It restricts access to Kubernetes resources, supporting least-privilege principles.",
  topic: "Data & AI Tools",
  difficulty: "Hard"
},
{
  question: "How can you secure sensitive information in Airflow DAGs?",
  answer: "Use Airflow Connections and Variables to store secrets, integrate with Vaults, and avoid hardcoding sensitive data in DAG files.",
  topic: "Data & AI Tools",
  difficulty: "Hard"
},
{
  question: "What are tags in Tableau and how can they improve governance?",
  answer: "Tags are metadata labels that help organize, search, and manage Tableau content (e.g., dashboards, data sources). They enhance governance and discoverability.",
  topic: "Data & AI Tools",
  difficulty: "Easy"
},
{
  question: "How does autoscaling work in Kubernetes?",
  answer: "Kubernetes autoscaling adjusts pod replicas based on metrics like CPU or memory usage using the Horizontal Pod Autoscaler, ensuring efficient resource utilization.",
  topic: "Data & AI Tools",
  difficulty: "Medium"
},
{
  question: "What are Snowflake's virtual warehouses and how do they impact cost and performance?",
  answer: "Virtual warehouses are independent compute clusters in Snowflake. They allow concurrent processing and isolated workloads, affecting cost and performance based on size and auto-suspend/resume policies.",
  topic: "Data & AI Tools",
  difficulty: "Medium"
},
{
  question: "How does version control integrate with data science workflows?",
  answer: "Version control (e.g., Git) tracks changes in notebooks, scripts, and datasets. It enables collaboration, reproducibility, and rollback in data science projects.",
  topic: "Data & AI Tools",
  difficulty: "Easy"
},
{
  question: "How do you find the second highest salary from a table without using LIMIT or OFFSET?",
  answer: "You can use a subquery: SELECT MAX(salary) FROM employees WHERE salary < (SELECT MAX(salary) FROM employees);",
  topic: "SQL for Data Analysis and ML Pipelines",
  difficulty: "Medium"
},
{
  question: "Write a SQL query to detect duplicate rows in a table based on multiple columns.",
  answer: "SELECT col1, col2, COUNT(*) FROM table GROUP BY col1, col2 HAVING COUNT(*) > 1;",
  topic: "SQL for Data Analysis and ML Pipelines",
  difficulty: "Easy"
},
{
  question: "How would you calculate a rolling average over time in SQL?",
  answer: "Use a window function: SELECT date, AVG(sales) OVER (ORDER BY date ROWS BETWEEN 6 PRECEDING AND CURRENT ROW) AS rolling_avg FROM sales_data;",
  topic: "SQL for Data Analysis and ML Pipelines",
  difficulty: "Medium"
},
{
  question: "How can you pivot data in SQL?",
  answer: "Use CASE statements with aggregation or use PIVOT if the SQL dialect supports it. Example: SELECT dept, SUM(CASE WHEN month='Jan' THEN sales ELSE 0 END) AS JanSales, ... FROM sales GROUP BY dept;",
  topic: "SQL for Data Analysis and ML Pipelines",
  difficulty: "Medium"
},
{
  question: "What is a CTE and when should you use it?",
  answer: "A Common Table Expression (CTE) is a temporary result set defined using WITH. Use it for modularizing complex queries and improving readability.",
  topic: "SQL for Data Analysis and ML Pipelines",
  difficulty: "Easy"
},
{
  question: "How do you detect and remove outliers in SQL?",
  answer: "Use statistical thresholds like IQR: SELECT * FROM table WHERE value BETWEEN (q1 - 1.5*IQR) AND (q3 + 1.5*IQR); after computing q1, q3, and IQR.",
  topic: "SQL for Data Analysis and ML Pipelines",
  difficulty: "Hard"
},
{
  question: "How can you calculate correlation between two variables in SQL?",
  answer: "Use: SELECT CORR(var1, var2) FROM table; if the SQL dialect supports CORR(). Otherwise, implement Pearson manually using aggregates.",
  topic: "SQL for Data Analysis and ML Pipelines",
  difficulty: "Hard"
},
{
  question: "Write a query to find customers who made purchases every month this year.",
  answer: "Use COUNT(DISTINCT month) = 12 in a HAVING clause grouped by customer_id.",
  topic: "SQL for Data Analysis and ML Pipelines",
  difficulty: "Hard"
},
{
  question: "How do you handle missing values in a pandas DataFrame?",
  answer: "Use methods like df.dropna(), df.fillna(value), or imputation techniques like mean/median/mode filling.",
  topic: "Python for Data Manipulation",
  difficulty: "Easy"
},
{
  question: "What's the difference between loc and iloc in pandas?",
  answer: "loc accesses by label (e.g., df.loc['row']) while iloc accesses by integer index (e.g., df.iloc[0]).",
  topic: "Python for Data Manipulation",
  difficulty: "Easy"
},
{
  question: "How can you detect and handle outliers in Python?",
  answer: "Use IQR method, z-score, or visualization tools like boxplots. Example: df[(df['col'] < Q3 + 1.5*IQR) & (df['col'] > Q1 - 1.5*IQR)]",
  topic: "Python for Data Manipulation",
  difficulty: "Medium"
},
{
  question: "How do you merge two pandas DataFrames with different join conditions?",
  answer: "Use pd.merge(df1, df2, on='key', how='left') for left join or change 'how' to 'inner', 'outer', or 'right' as needed.",
  topic: "Python for Data Manipulation",
  difficulty: "Easy"
},
{
  question: "How would you efficiently apply a function row-wise in a large DataFrame?",
  answer: "Use vectorized operations if possible, otherwise apply with axis=1: df.apply(func, axis=1).",
  topic: "Python for Data Manipulation",
  difficulty: "Medium"
},
{
  question: "How do you evaluate a binary classification model?",
  answer: "Use metrics such as accuracy, precision, recall, F1 score, ROC AUC. Prefer F1 for imbalanced data.",
  topic: "Python for Machine Learning",
  difficulty: "Medium"
},
{
  question: "What is the purpose of cross-validation in ML?",
  answer: "To estimate the generalization performance of a model and reduce overfitting. k-Fold CV is a common method.",
  topic: "Python for Machine Learning",
  difficulty: "Easy"
},
{
  question: "How would you tune hyperparameters in scikit-learn?",
  answer: "Use GridSearchCV or RandomizedSearchCV from sklearn.model_selection.",
  topic: "Python for Machine Learning",
  difficulty: "Medium"
},
{
  question: "How can you detect and fix data leakage in ML pipelines?",
  answer: "Ensure feature engineering is done within cross-validation loops, and leakage-prone features (like future data) are removed.",
  topic: "Python for Machine Learning",
  difficulty: "Hard"
},
{
  question: "What is a pipeline in scikit-learn and why is it useful?",
  answer: "A pipeline chains preprocessing and modeling steps, ensuring consistent transformations and preventing data leakage.",
  topic: "Python for Machine Learning",
  difficulty: "Medium"
},
{
  question: "How do you handle class imbalance in classification problems?",
  answer: "Use techniques like SMOTE, class weights, or downsampling majority class.",
  topic: "Python for Machine Learning",
  difficulty: "Hard"
},
{
  question: "How would you interpret feature importance from a tree-based model?",
  answer: "Use built-in model.feature_importances_ or permutation importance to evaluate effect of each feature on predictions.",
  topic: "Python for Machine Learning",
  difficulty: "Medium"
},
{
  question: "What’s the difference between bagging and boosting?",
  answer: "Bagging trains models independently in parallel (e.g., Random Forest), boosting trains sequentially correcting errors (e.g., XGBoost).",
  topic: "Python for Machine Learning",
  difficulty: "Medium"
},
{
  question: "Explain regularization in linear models and its impact.",
  answer: "Regularization (L1, L2) penalizes large coefficients to reduce overfitting. L1 (Lasso) can shrink some weights to zero.",
  topic: "Python for Machine Learning",
  difficulty: "Medium"
},
{
  question: "How do you deploy a trained scikit-learn model?",
  answer: "Serialize it with joblib or pickle, then load it in a web API (e.g., Flask or FastAPI) for inference.",
  topic: "Python for Machine Learning",
  difficulty: "Hard"
},
{
  question: "How do you define a simple neural network using PyTorch?",
  answer: "Subclass `nn.Module`, define layers in `__init__`, and implement forward pass in `forward()`. Example: class Net(nn.Module): def __init__(self): ... def forward(self, x): return self.fc(x)",
  topic: "AI and Generative AI",
  difficulty: "Easy"
},
{
  question: "What are the key differences between PyTorch and Keras?",
  answer: "PyTorch offers more flexibility and dynamic computation graphs; Keras (on top of TensorFlow) is more user-friendly and suitable for rapid prototyping.",
  topic: "AI and Generative AI",
  difficulty: "Easy"
},
{
  question: "How can you fine-tune a Hugging Face transformer model for text classification?",
  answer: "Load a pre-trained model using `AutoModelForSequenceClassification`, prepare data with a tokenizer, use `Trainer` or native PyTorch training loop.",
  topic: "AI and Generative AI",
  difficulty: "Medium"
},
{
  question: "What does 'attention' mean in transformer architectures?",
  answer: "Attention is a mechanism that allows models to weigh the importance of different input tokens when making predictions, enabling context-aware learning.",
  topic: "AI and Generative AI",
  difficulty: "Medium"
},
{
  question: "How do you use OpenAI’s GPT API to generate text?",
  answer: "Use the `openai.ChatCompletion.create()` or `openai.Completion.create()` methods with a prompt, model name, and temperature/parameters for control.",
  topic: "AI and Generative AI",
  difficulty: "Easy"
},
{
  question: "What is LangChain and how is it used in GenAI applications?",
  answer: "LangChain is a framework for building LLM-driven applications by chaining prompts, models, memory, and tools together for complex reasoning workflows.",
  topic: "AI and Generative AI",
  difficulty: "Medium"
},
{
  question: "How would you implement retrieval-augmented generation (RAG)?",
  answer: "Use vector embedding search to retrieve relevant context, then pass the results as part of the prompt to the LLM for final answer generation.",
  topic: "AI and Generative AI",
  difficulty: "Hard"
},
{
  question: "What is a prompt template in LangChain and why is it useful?",
  answer: "A prompt template standardizes inputs to LLMs by dynamically formatting instructions and user inputs, improving consistency and modularity.",
  topic: "AI and Generative AI",
  difficulty: "Medium"
},
{
  question: "Explain how tokenization affects LLM input/output.",
  answer: "Tokenization breaks text into subword units. The model processes tokens, not raw text. Token limits affect context length, cost, and memory.",
  topic: "AI and Generative AI",
  difficulty: "Medium"
},
{
  question: "How do you use embeddings with OpenAI API?",
  answer: "Use `openai.Embedding.create()` to get vector representations of text, then apply similarity search (e.g., cosine similarity) for retrieval tasks.",
  topic: "AI and Generative AI",
  difficulty: "Medium"
},
{
  question: "What is the role of a vector store in GenAI pipelines?",
  answer: "A vector store indexes embeddings for fast similarity search. It's essential for RAG and contextual search in LLM applications.",
  topic: "AI and Generative AI",
  difficulty: "Medium"
},
{
  question: "How do you cache LLM responses using LangChain?",
  answer: "Use LangChain’s built-in caching wrappers, like `langchain.cache`, or external cache systems like Redis to avoid repeated calls to LLMs.",
  topic: "AI and Generative AI",
  difficulty: "Medium"
},
{
  question: "Describe the difference between greedy decoding, beam search, and sampling.",
  answer: "Greedy picks the highest probability token at each step. Beam search explores multiple paths. Sampling randomly picks based on token probabilities.",
  topic: "AI and Generative AI",
  difficulty: "Medium"
},
{
  question: "How can you use Hugging Face's `pipeline()` for zero-shot classification?",
  answer: "Use `pipeline('zero-shot-classification')` with your text and candidate labels; it uses models like BART or RoBERTa behind the scenes.",
  topic: "AI and Generative AI",
  difficulty: "Easy"
},
{
  question: "What are guardrails in the context of LLM applications?",
  answer: "Guardrails are safety mechanisms to constrain or filter LLM outputs using tools like prompt engineering, moderation, output parsing, or content filters.",
  topic: "AI and Generative AI",
  difficulty: "Hard"
},
{
  question: "How do you use agents in LangChain?",
  answer: "Agents use reasoning steps to choose tools and respond to queries. You set up an LLM, tools, and an agent executor to run tasks dynamically.",
  topic: "AI and Generative AI",
  difficulty: "Hard"
},
{
  question: "How would you finetune a transformer model using PyTorch Lightning?",
  answer: "Wrap the model in a `LightningModule`, handle training/validation steps, and use `Trainer` for training, which abstracts boilerplate and adds features like checkpointing.",
  topic: "AI and Generative AI",
  difficulty: "Hard"
},
// Data Visualization Best Practices
{
  question: "What are the key principles of effective data visualization?",
  answer: "Effective data visualizations should be clear, accurate, and purposeful. They should minimize chartjunk, use appropriate scales, highlight important patterns, and be tailored to the audience’s level of expertise.",
  topic: "Data Visualization Best Practices",
  difficulty: "Easy"
},
{
  question: "When should you use a bar chart versus a line chart?",
  answer: "Use a bar chart for comparing discrete categories and a line chart to show trends over a continuous dimension, such as time. Line charts emphasize the flow or trajectory of data.",
  topic: "Data Visualization Best Practices",
  difficulty: "Easy"
},
{
  question: "What are common mistakes to avoid in visualizing data?",
  answer: "Common mistakes include misleading axes (e.g., truncating the y-axis), using inappropriate chart types, excessive labeling, poor color choices, and cluttered visuals that obscure insights.",
  topic: "Data Visualization Best Practices",
  difficulty: "Medium"
},
{
  question: "How can color be used effectively in data visualizations?",
  answer: "Color should be used to highlight patterns, differentiate groups, or convey magnitude. It's important to consider colorblind-safe palettes and avoid overuse of colors that can confuse interpretation.",
  topic: "Data Visualization Best Practices",
  difficulty: "Medium"
},
{
  question: "What is the 'data-ink ratio' and why does it matter?",
  answer: "The data-ink ratio, introduced by Edward Tufte, refers to the proportion of a graphic’s ink used to display data versus decorative elements. A high ratio ensures that the visualization focuses on conveying information, not aesthetics.",
  topic: "Data Visualization Best Practices",
  difficulty: "Medium"
},
{
  question: "Why is it important to consider the audience when designing visualizations?",
  answer: "Different audiences have varying levels of data literacy. Tailoring visualizations to the audience ensures the message is clear, relevant, and engaging. Analysts may prefer detailed visuals, while executives need high-level summaries.",
  topic: "Data Visualization Best Practices",
  difficulty: "Medium"
},
{
  question: "How do you choose between a heatmap, scatter plot, and box plot?",
  answer: "Choose a heatmap for visualizing matrix-style data and correlations, a scatter plot for exploring relationships between two continuous variables, and a box plot for comparing distributions across groups.",
  topic: "Data Visualization Best Practices",
  difficulty: "Medium"
},
{
  question: "What is the benefit of using interactive data visualizations?",
  answer: "Interactive visualizations allow users to explore data dynamically, uncover hidden insights, filter by segments, and drill down into specific areas. This improves engagement and understanding, especially with large datasets.",
  topic: "Data Visualization Best Practices",
  difficulty: "Hard"
}
];

const difficultyColors: Record<string, string> = {
  Easy: "bg-green-100 text-green-800 hover:bg-green-200",
  Medium: "bg-yellow-100 text-yellow-800 hover:bg-yellow-200", 
  Hard: "bg-red-100 text-red-800 hover:bg-red-200"
};

const topicEmojis: Record<string, string> = {
  "Probability and Distributions": "📘",
  "Hypothesis Testing and Statistical Inference": "📗",
  "Estimation and Confidence": "📙",
  "Regression and Modeling Assumptions": "📕",
  "Bayesian Statistics": "📒",
  "Real-World Statistical Decision Making": "📓",
  "Advanced Topics": "🧠"
};

export default function FlashcardApp() {
  const [topicFilter, setTopicFilter] = useState("All");
  const [difficultyFilter, setDifficultyFilter] = useState("All");
  const [currentIndex, setCurrentIndex] = useState(0);
  const [showAnswer, setShowAnswer] = useState(false);
  const [learned, setLearned] = useState(new Set());
  const [reviewLearned, setReviewLearned] = useState(false);
  const [isShuffled, setIsShuffled] = useState(false);
  const [sessionStats, setSessionStats] = useState({ started: Date.now(), cardsReviewed: 0 });

  const topics = ["All", ...new Set(flashcards.map((f) => f.topic))];
  const difficulties = ["All", ...new Set(flashcards.map((f) => f.difficulty))];

  const filtered = flashcards.filter((f, index) => {
    const topicMatch = topicFilter === "All" || f.topic === topicFilter;
    const difficultyMatch = difficultyFilter === "All" || f.difficulty === difficultyFilter;
    const notLearned = reviewLearned || !learned.has(index);
    return topicMatch && difficultyMatch && notLearned;
  });

  const current = filtered[currentIndex] || null;
  const progress = filtered.length ? ((currentIndex + 1) / filtered.length) * 100 : 0;
  const learnedCount = learned.size;
  const totalCards = flashcards.length;

  useEffect(() => {
    if (currentIndex >= filtered.length && filtered.length > 0) {
      setCurrentIndex(0);
    }
  }, [filtered.length, currentIndex]);

  const markLearned = () => {
    const globalIndex = flashcards.findIndex(
      (f) => f.question === current?.question && f.answer === current?.answer
    );
    if (globalIndex !== -1) {
      const newSet = new Set(learned);
      newSet.add(globalIndex);
      setLearned(newSet);
      setSessionStats((prev: typeof sessionStats) => ({ ...prev, cardsReviewed: prev.cardsReviewed + 1 }));
    }
    nextCard();
  };

  const nextCard = () => {
    setCurrentIndex((i: number) => (i + 1) % filtered.length);
    setShowAnswer(false);
  };

  const prevCard = () => {
    setCurrentIndex((i: number) => (i - 1 + filtered.length) % filtered.length);
    setShowAnswer(false);
  };

  const shuffleCards = () => {
    setIsShuffled(!isShuffled);
    setCurrentIndex(0);
    setShowAnswer(false);
  };

  const resetLearned = () => {
    setLearned(new Set());
    setCurrentIndex(0);
    setShowAnswer(false);
    setSessionStats({ started: Date.now(), cardsReviewed: 0 });
  };

  const startOver = () => {
    setTopicFilter("All");
    setDifficultyFilter("All");
    setCurrentIndex(0);
    setShowAnswer(false);
    setReviewLearned(false);
    setIsShuffled(false);
    setSessionStats({ started: Date.now(), cardsReviewed: 0 });
  };

  const getSessionTime = () => {
    const minutes = Math.floor((Date.now() - sessionStats.started) / 60000);
    return minutes;
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 via-indigo-50 to-purple-50">
      <div className="max-w-4xl mx-auto p-6">
        {/* Header */}
        <div className="text-center mb-8">
          <h1 className="text-4xl font-bold text-gray-900 mb-2 flex items-center justify-center gap-3">
            <FiBookOpen className="text-blue-600" />
            Data Science Interview Prep
          </h1>
          <p className="text-gray-600">Master DS/ML/AI concepts with interactive flashcards</p>
        </div>

        {/* Stats Dashboard */}
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-6">
          <Box className="p-4 text-center bg-white/80 backdrop-blur-sm rounded-lg shadow-md">
            <div className="flex items-center justify-center gap-2 mb-1">
              <FiAward className="w-4 h-4 text-yellow-600" />
              <span className="text-sm font-medium text-gray-600">Learned</span>
            </div>
            <p className="text-2xl font-bold text-gray-900">{learnedCount}/{totalCards}</p>
          </Box>
          <Box className="p-4 text-center bg-white/80 backdrop-blur-sm rounded-lg shadow-md">
            <div className="flex items-center justify-center gap-2 mb-1">
              <FiTarget className="w-4 h-4 text-blue-600" />
              <span className="text-sm font-medium text-gray-600">Current</span>
            </div>
            <p className="text-2xl font-bold text-gray-900">{filtered.length ? currentIndex + 1 : 0}/{filtered.length}</p>
          </Box>
          <Box className="p-4 text-center bg-white/80 backdrop-blur-sm rounded-lg shadow-md">
            <div className="flex items-center justify-center gap-2 mb-1">
              <FiCheckCircle className="w-4 h-4 text-green-600" />
              <span className="text-sm font-medium text-gray-600">Reviewed</span>
            </div>
            <p className="text-2xl font-bold text-gray-900">{sessionStats.cardsReviewed}</p>
          </Box>
          <Box className="p-4 text-center bg-white/80 backdrop-blur-sm rounded-lg shadow-md">
            <div className="flex items-center justify-center gap-2 mb-1">
              <FiClock className="w-4 h-4 text-purple-600" />
              <span className="text-sm font-medium text-gray-600">Time</span>
            </div>
            <p className="text-2xl font-bold text-gray-900">{getSessionTime()}m</p>
          </Box>
        </div>

        {/* Progress Bar */}
        <div className="mb-6">
          <div className="flex justify-between text-sm text-gray-600 mb-2">
            <span>Progress</span>
            <span>{Math.round(progress)}%</span>
          </div>
          <Progress value={progress} className="h-2" />
        </div>

        {/* Filters */}
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mb-6">
          <div className="space-y-2">
            <label className="text-sm font-medium text-gray-700">Topic</label>
            <select
              value={topicFilter}
              onChange={(e: React.ChangeEvent<HTMLSelectElement>) => {
                setTopicFilter(e.target.value);
                setCurrentIndex(0);
                setShowAnswer(false);
              }}
              className="bg-white/80 backdrop-blur-sm"
            >
              {topics.map((topic) => (
                <option key={topic} value={topic} className="flex items-center gap-2">
                  {topic !== "All" && topicEmojis[topic]} {topic}
                </option>
              ))}
            </select>
          </div>
          <div className="space-y-2">
            <label className="text-sm font-medium text-gray-700">Difficulty</label>
            <select
              value={difficultyFilter}
              onChange={(e: React.ChangeEvent<HTMLSelectElement>) => {
                setDifficultyFilter(e.target.value);
                setCurrentIndex(0);
                setShowAnswer(false);
              }}
              className="bg-white/80 backdrop-blur-sm"
            >
              {difficulties.map((level) => (
                <option key={level} value={level}>
                  {level}
                </option>
              ))}
            </select>
          </div>
        </div>

        {/* Control Buttons */}
        <div className="flex flex-wrap gap-2 mb-6 justify-center">          <Button 
            variant="outline" 
            onClick={resetLearned}
            className="bg-white/80 backdrop-blur-sm hover:bg-white px-6 py-3 flex items-center justify-center min-w-[140px]"
          >
            <FiRotateCcw className="w-4 h-4 mr-2" />
            Reset Learned
          </Button>
          <Button 
            variant="outline" 
            onClick={startOver}
            className="bg-white/80 backdrop-blur-sm hover:bg-white px-6 py-3 flex items-center justify-center min-w-[140px]"
          >
            <FiRefreshCw className="w-4 h-4 mr-2" />
            Start Over
          </Button>
          <Button
            variant={reviewLearned ? "solid" : "outline"}
            onClick={() => {
              setReviewLearned(!reviewLearned);
              setCurrentIndex(0);
              setShowAnswer(false);
            }}
            className={`${reviewLearned ? "" : "bg-white/80 backdrop-blur-sm hover:bg-white"} px-6 py-3 flex items-center justify-center min-w-[140px]`}
          >
            {reviewLearned ? <FiEyeOff className="w-4 h-4 mr-2" /> : <FiEye className="w-4 h-4 mr-2" />}
            {reviewLearned ? "Hide Learned" : "Review Learned"}
          </Button>
          <Button
            variant={isShuffled ? "solid" : "outline"}
            onClick={shuffleCards}
            className={`${isShuffled ? "" : "bg-white/80 backdrop-blur-sm hover:bg-white"} px-6 py-3 flex items-center justify-center min-w-[140px]`}
          >
            <FiShuffle className="w-4 h-4 mr-2" />
            Shuffle
          </Button>
        </div>

        {/* Main Flashcard */}
        {current ? (
          <Box className="shadow-2xl border-0 bg-white/90 backdrop-blur-sm rounded-lg">
            <div className="p-8">
              {/* Topic and Difficulty Badges */}
              <div className="flex flex-wrap gap-2 mb-6 justify-between items-start">
                <Badge variant="subtle" className="text-xs">
                  {topicEmojis[current.topic]} {current.topic}
                </Badge>
                <Badge className={`text-xs ${difficultyColors[current.difficulty]}`}>
                  {current.difficulty}
                </Badge>
              </div>

              {/* Question */}
              <div className="mb-8">
                <h3 className="text-xl font-semibold text-gray-900 leading-relaxed">
                  {current.question}
                </h3>
              </div>

              {/* Answer Section */}
              {!showAnswer ? (
                <div className="text-center">
                  <Button
                    size="lg"
                    onClick={() => setShowAnswer(true)}
                    className="bg-blue-600 hover:bg-blue-700 text-white px-8 py-3"
                  >
                    <FiEye className="w-5 h-5 mr-2" />
                    Reveal Answer
                  </Button>
                </div>
              ) : (
                <div className="space-y-6">
                  <div className="bg-gray-50 rounded-xl p-6 border-l-4 border-blue-500">
                    <p className="text-gray-800 leading-relaxed whitespace-pre-wrap">
                      {current.answer}
                    </p>
                  </div>
                </div>
              )}

              {/* Navigation and Action Buttons */}
              <div className="flex flex-col sm:flex-row gap-4 mt-8">
                <div className="flex gap-2 flex-1">
                  <Button
                    variant="outline"
                    onClick={prevCard}
                    disabled={filtered.length <= 1}
                    className="flex-1 bg-white/80 backdrop-blur-sm hover:bg-white"
                  >
                    <FiChevronLeft className="w-4 h-4 mr-2" />
                    Previous
                  </Button>
                  <Button
                    variant="outline"
                    onClick={nextCard}
                    disabled={filtered.length <= 1}
                    className="flex-1 bg-white/80 backdrop-blur-sm hover:bg-white"
                  >
                    Next
                    <FiChevronRight className="w-4 h-4 ml-2" />
                  </Button>
                </div>
                
                {showAnswer && (
                  <Button
                    onClick={markLearned}
                    className="bg-green-600 hover:bg-green-700 text-white sm:w-auto w-full"
                  >
                    <FiCheckCircle className="w-4 h-4 mr-2" />
                    Mark as Learned
                  </Button>
                )}
              </div>
            </div>
          </Box>
        ) : (
          <Box className="shadow-2xl border-0 bg-white/90 backdrop-blur-sm rounded-lg">
            <div className="p-12 text-center">
              <div className="space-y-4">
                <FiAward className="w-16 h-16 text-yellow-500 mx-auto" />
                <h3 className="text-2xl font-bold text-gray-900">
                  Great job! 🎉
                </h3>
                <p className="text-gray-600 text-lg">
                  {reviewLearned 
                    ? "You've reviewed all your learned cards!"
                    : "No more flashcards matching your current filters."}
                </p>
                <div className="flex flex-col sm:flex-row gap-3 justify-center mt-6">
                  <Button onClick={startOver} size="lg" className="bg-blue-600 hover:bg-blue-700">
                    <FiRefreshCw className="w-4 h-4 mr-2" />
                    Start New Session
                  </Button>
                  {!reviewLearned && learnedCount > 0 && (
                    <Button
                      variant="outline"
                      onClick={() => {
                        setReviewLearned(true);
                        setCurrentIndex(0);
                        setShowAnswer(false);
                      }}
                      size="lg"
                      className="bg-white/80 backdrop-blur-sm hover:bg-white"
                    >
                      <FiEye className="w-4 h-4 mr-2" />
                      Review Learned Cards
                    </Button>
                  )}
                </div>
              </div>
            </div>
          </Box>
        )}

        {/* Footer */}
        <div className="text-center mt-8 text-gray-500 text-sm">
          <p>Data Science Interview Preparation • {totalCards} Total Questions</p>
        </div>
      </div>
    </div>
  );
}