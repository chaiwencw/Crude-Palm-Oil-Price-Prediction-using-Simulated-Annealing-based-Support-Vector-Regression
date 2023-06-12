# CPO Price Prediction using Simulated Annealing-based Support Vector Regression

## Introduction
The study use a metaheuristic algorithm, called Simulated Annealing (SA) to optimize the hyperparameter of the Support Vector Regression (SVR) model. 

### Support Vector Regression
Support vector regression (SVR) is an extension of the support vector machine (SVM) applied to regression problems. Linear, polynomial, radial basis function (RBF), and sigmoid kernels are the most commonly used kernels in an SVM implementation. There is no direct way to determine the best kernel choice for a specific data pattern. According to Ojemakinde (2006), without prior knowledge about the data, the RBF kernel is the preferable choice justified by some valid reasons. First, it requires less tuneable hyperparameters than polynomials. RBF also has fewer numerical difficulties since the kernel value (𝛾) ranges from 0 to 1, whereas the range of these values of the polynomial kernel can fall between 0 and ∞. Besides, although the sigmoid kernel is successfully applicable, it is not always fulfilling the requirement for an SVR kernel, called Mercer’s condition. The sigmoid kernel is also similar to the RBF kernel when the kernel width is small. In addition, Ali Alahmari (2020) and Saadah et al. (2021) revealed that SVR with RBF kernel demonstrated an outstanding prediction performance in price prediction problems.

For any kernel type, the SVR model complexity can be affected by the values of 𝐶 and 𝜀. In this project, the intention is applying the RBF kernel for CPO price forecasting. Therefore, the tuneable hyperparameters of the RBF-SVR model are C, ε, and γ. Simulated annealing (SA) will be used to find the near optimal solutions for these hyperparameters to improve the prediction performance.

### Simulated Annealing
The simulated annealing (SA) algorithm is an iterative improvement algorithm. It uses a random search, which always accepts changes (solutions) that improve the objective function but sometimes also keeps some changes that are not ideal in the search process based on the acceptance probability function. SA parameters that impact the result of hyperparameter tuning include cooling factor (𝛼), number of iterations, initial temperature (_T_<sub>0</sub>) and minimum temperature  (_𝑇_<sub>min</sub>).

According to Fischetti and Stringher (2019), 𝑇 can be updated using a simple formula 𝑇 = 𝛼 × 𝑇, with cooling factor 𝛼 ∈ (0,1) such that 𝛼 ∈ (0.7, 0.8) when cooling is applied after several SA iterations with a constant 𝑇. Hence, it would be ideal to take an average of the two boundaries, 0.7 and 0.8, which is 0.75.

Note that SA is an iteration-intensive algorithm where the number of iterations at any given temperature will affect the duration and the quality of the obtained solution. The number of iterations needed to achieve global optima depends on the size of the problems, as the number of iterations might be as large as millions. However, Martinez-Rios and Frausto-Solis (2012) were able to use SA with only 100 iterations in solving a nondeterministic polynomial-time complete (NP-complete) problem, which is the “Boolean Satisfiability problem”. Thus, we consider using 100 iterations in hyperparameter tuning is worthwhile as this process might involve the search space of a million numbers.

Finally, choose both initial (_T_<sub>0</sub>) and minimum (_T_<sub>min</sub>) temperatures wisely since they affect the acceptance probability, which impacts the overall tuning result. _T_<sub>0</sub> should be large enough to make the initial acceptance probability closer to 1, and _T_<sub>min</sub> should be much lower so that the acceptance probability decreases gradually throughout the annealing process. Fischetti and Stringher (2019) chose _T_<sub>0</sub> = 1 and considered a temperature reduction of 3 to 5 times, which has reduced the acceptance probability low enough. Therefore, we decided to set _T_<sub>0</sub> = 100 and _T_<sub>min</sub> = 30 (achieved after five times temperature reductions) for this project.

### Hyperparameter Tuning Procedure
The procedure of hyperparameter tuning is as follows:
1. Randomly choose values for all hyperparameters, assuming it as current state and evaluating model performance with the selected evaluation metric.
2. Obtain new current state by randomly updating the value of one hyperparameter by randomly selecting a value in the neighbourhood to get neighbouring state.
3. If the combination is repeated, repeat Step 2 until a new combination is generated.
4. Evaluate model performance on the neighbouring state.
5. Compare the model performance of neighbouring state to the current state and decide whether to accept the neighbouring state as current state or reject it based on the value of the evaluation metric.
6. According to the result of Step 5, repeating Steps 2 through 5.

For further analysis, the algorithm can intake a set of previously found hyperparameters, so the algorithm will continue the search of new current state around the neighbourhood of previously found hyperparameters.

The steps of tuning are presented in the flowchart below.

<img width="500" alt="Screenshot 2023-05-22 at 10 38 06 PM" src="https://github.com/chaiwencw/Crude-Palm-Oil-Price-Prediction-using-Simulated-Annealing-based-Support-Vector-Regression/assets/85020127/6c098899-0396-41ef-9e7a-7f43d3e6fc41">

## References
1. Ali Alahmari, S. (2020). Predicting the Price of Cryptocurrency using Support Vector Regression Methods. JOURNAL OF MECHANICS OF CONTINUA AND MATHEMATICAL SCIENCES, 15. https://doi.org/10.26782/jmcms.2020.04.00023
2. Fischetti, M., & Stringher, M. (2019). Embedded hyper-parameter tuning by Simulated Annealing. ArXiv E-Prints, arXiv:1906.01504.
3. Martinez-Rios, F., & Frausto-Solis, J. (2012). A Simulated Annealing Algorithm for the Satisfiability Problem Using Dynamic Markov Chains with Linear Regression Equilibrium. In Simulated Annealing - Advances, Applications and Hybridizations. https://doi.org/10.5772/46175
4. Ojemakinde, B. (2006). Support Vector Regression for Non-Stationary Time Series. Masters Theses. https://trace.tennessee.edu/utk_gradthes/1756
5. Saadah, S., Z, F., & Z, H. (2021). Support Vector Regression (SVR) Dalam Memprediksi Harga Minyak Kelapa Sawit di Indonesia dan Nilai Tukar Mata Uang EUR/USD: Support Vector Machine (SVM) To Predict Crude Oil Palm in Indonesia and Exchange Rate of EUR/USD. Journal of Computer Science and Informatics Engineering (J-Cosine), 5, 85–92. https://doi.org/10.29303/jcosine.v5i1.403



