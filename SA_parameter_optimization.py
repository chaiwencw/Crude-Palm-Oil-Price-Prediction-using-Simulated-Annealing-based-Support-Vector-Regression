from argparse import OPTIONAL
from xmlrpc.client import boolean
import pandas as pd
import numpy as np
from math import ceil
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, mean_squared_error

MAE = mean_absolute_error
MAPE = mean_absolute_percentage_error
MSE = mean_squared_error

def RegressionMetric(valid_data, predictions):

    metrics = [MAE, MAPE, MSE]
    metric_value = []
    
    for index, metric in enumerate(metrics):
        metric_value.append(metric(valid_data, predictions))
    print("MAE: ", metric_value[0])
    print("MAPE:", metric_value[1]) 
    print("MSE: ", metric_value[2])
    print("RMSE:", np.sqrt(metric_value[2]))


# Function to train model and output best model with metric
def model_training(Xtrain, Ytrain, Xvalid, Yvalid, model_to_train,
                    current_parameters = dict(), constant_parameters = dict(), **optional_output):
    """
    Train the model with given set of hyperparameters
    current_parameters - Dict of hyperparameters and chosen values
    constant_parameters - Dict of hyperparameters that are kept constant
    Xtrain - Train Data
    Xvalid - Validation Data
    Ytrain - Train labels
    Yvalid - Validaion labels
    metric - Metric to compute model performance
    """
    output = []
    #copy the constant parameters dictionary
    parameters_copy = constant_parameters.copy()
    #combine the two constant and current parameters dictionary
    parameters_copy.update(current_parameters)
    #"**" is used to input a dict that contains all the key which are same as the arguments of the mdoel    
    model = model_to_train(**parameters_copy)
    #"fit" is used to train the model
    trained_model = model.fit(Xtrain, Ytrain)
    output.append(trained_model)
    #"predict" is used to make prediction
    predictions = trained_model.predict(Xvalid)

    if optional_output:
        model_param = optional_output.keys()
    
        if "prediction" in model_param and optional_output['prediction'] == True:
            output.append(predictions)

        if "metric" in model_param:
            #"metric" here equals the F1_Score function imported from sklearn, it's used to measure how effective the model is in making prediction by comparing true and predicted target values
            measure = optional_output['metric']
            metric_value = measure(Yvalid, predictions)
            output.append(metric_value)

    if len(output) == 1:
        return output[0]
    else:
        return tuple(output)

#Function to choose parameters for optimization
def parameters_choosing(potential_parameters, current_parameters = None):
    """
    Function to choose parameters for next iteration
    
    Inputs:
    potential_parameters - Ordered dictionary of hyperparameter search space
    current_parameters - Dict of current hyperparameters

    Output:
    Dictionary of New Parameters
    """
       
    #when current parameters is non-empty dict
    if current_parameters:
        #copy current parameters dict to another dict called "new parameters"
        new_parameters = current_parameters.copy()
        #change the "keys" of the potential parameters dict into list
        potential_param_keys = list(potential_parameters.keys())
        #randomly choose a "key" out of the potential_param_keys and set the "key" as the parameter-to-be-updated
        param_to_update = np.random.choice(potential_param_keys)
        #obtain the potential values for the parameter-to-be-updated
        potential_param_values = potential_parameters[param_to_update]
        #find the current index of the value of the parameter-to-be-updated
        curr_param_val_index = potential_param_values.index(current_parameters[param_to_update])
        no_potential_param_val = len(potential_param_values)
        random_range = [x for x in np.arange(-no_potential_param_val, no_potential_param_val + 1) if x != 0]
        #if the value of the parameter is the first in the list of potential values 
        if curr_param_val_index == 0:
            positive_random_index = np.random.choice([x for x in random_range if x > 0])
            #set the value of the parameter-to-be-updated as the second in the list of potential values
            new_parameters[param_to_update] = potential_param_values[positive_random_index - 1]
        #if the value of the parameter is the last in the list of potential values 
        elif curr_param_val_index == len(potential_param_values) - 1:
            negative_random_index = np.random.choice([x for x in random_range if x < 0])
            #set the value of the parameter-to-be-updated as the second last in the list of potential values
            new_parameters[param_to_update] = potential_param_values[negative_random_index + 1]
        else:
            #set the value of the parameter-to-be-updated as the value with index +1 or -1 in the list of potential values
            restrict = np.arange(ceil(len(random_range)/5))
            final_index = curr_param_val_index + np.random.choice(restrict)

            if final_index >= no_potential_param_val: 
                new_parameters[param_to_update] = potential_param_values[-1]
            elif final_index <= 0:
                new_parameters[param_to_update] = potential_param_values[0]
            else:
                new_parameters[param_to_update] = potential_param_values[final_index]
    #when current parameters is empty dict
    else:
        #create a new empty dict
        new_parameters = dict()
        #randomly assign the potential values to the parameters
        for k, v in potential_parameters.items():
            new_parameters[k] = np.random.choice(v)

    return new_parameters


def simulate_annealing(param_dict, constant_params, 
                        X_train, Y_train,  X_valid, Y_valid, 
                        ML_model, benchmark_metric = MSE,
                        no_iters: int = 100, alpha = 0.95, beta = 1, 
                        initial_temperature = 100, min_temperature = 1, previous_parameters: dict = None,
                        **optional_output: bool):
    """
    Function to perform hyperparameter search using simulated annealing (minimization)

    Inputs:
    param_dict - Ordered dictionary of Hyperparameter search space
    const_param - Static parameters of the model
    Xtrain - Train Data
    Xvalid - Validation Data
    Ytrain - Train labels
    Yvalid - Validaion labels
    training_function - Function to train the model
    no_iters - Number of iterations to perform the parameter search
    alpha - factor to reduce temperature
    beta - constant in probability estimate
    initial_temperature - Initial temperature
    min_temperature - Minimum temperature
    
    Output:
    Dataframe of the parameters explored and corresponding model performance
    """
    output = []
    T = initial_temperature
    T_min = min_temperature

    columns_name = ['Number of Temperature Reduction'] + [*param_dict.keys()] + ['Metric', 'Best Metric']
    results = pd.DataFrame(columns = columns_name)
    ori_model, ori_metric = model_training(X_train, Y_train, X_valid, Y_valid, 
                                            ML_model, metric = benchmark_metric)
    prev_params = previous_parameters
    prev_metric = ori_metric
    best_metric = ori_metric
    best_params = dict()
    weights = tuple(10**x for x in range(len(param_dict)))
    hash_values = set()
    result_list = []
    j = 0

    while T >= T_min:
        print("Current Temperature is: %.2f" %T)
        print("\n")
        
        for i in range(no_iters):
            print('Starting Iteration ' + str(i+1))

            curr_params = parameters_choosing(param_dict, prev_params)
            indices = tuple(param_dict[k].index(v) for k, v in curr_params.items())
            hash_val = sum([i * j for (i, j) in zip(weights, indices)])

            if hash_val in hash_values:
                print('Combination revisited.')
                print('\n\n')

            else:
                hash_values.add(hash_val)
                            
                model, metric = model_training(X_train, Y_train, X_valid, Y_valid, 
                                            ML_model, curr_params, constant_params,
                                            metric = benchmark_metric)

                if metric < prev_metric:
                    print('Local Improvement in metric from {:8.6f} to {:8.6f} '
                            .format(prev_metric, metric) + ' - parameters accepted' + '\n')
                    prev_metric = metric
                    prev_params = curr_params.copy()
                    
                    if metric < best_metric:
                        print('Global Improvement in metric from {:8.6f} to {:8.6f} '
                                .format(best_metric, metric) + ' - best parameters updated' + '\n\n')
                        best_metric = metric
                        best_params = curr_params.copy()
                        best_model = model
                
                else:
                    random_no = np.random.uniform()
                    diff = metric - prev_metric
                    Metropolis = np.exp(- beta * diff / T)
                    if random_no < Metropolis:
                        print("No Improvement but parameters are ACCEPTED.") 
                        prev_metric = metric
                        prev_params = curr_params
                        
                    else:
                        print("No Improvement and parameters are REJECTED.") 
                    
                    print("Metric change:   %.6f" % diff)
                    print("Threshold:       %.6f" % Metropolis)
                    print("Random Number:   %.6f" % random_no)
                    print('\n')

            results.loc[i, 'Number of Temperature Reduction'] = j
            results.loc[i, list(curr_params.keys())] = list(curr_params.values())
            results.loc[i, 'Metric'] = metric
            results.loc[i, 'Best Metric'] = best_metric
            print("\n")
        
        result_copy = results.copy()
        result_list.append(result_copy)
        
        T = alpha * T

        print("Temperature has been reduced.")
        print("The number of temperature reduced: " + str(j + 1))
        j = j + 1

        if T < T_min: print("Minimum temperature is reached. The algorithm is terminated.")

    output.append(best_model)

    if optional_output:
        SA_param = optional_output.keys()
        
        if "result" in SA_param and optional_output['result'] == True:
            final_result = pd.concat(result_list)
            output.append(final_result)

        if "best_parameter" in SA_param and optional_output['best_parameter'] == True:
            #final_parameter = final_result[final_result["Metric"] == final_result['Best Metric'].min()]
            output.append(best_params)
            
        if "total_iterations" in SA_param and optional_output['total_iterations'] == True:
            total_no_temp_drop = np.log(T_min/T)/np.log(alpha)
            total_no_iter = ceil(total_no_temp_drop)*no_iters
            output.append(total_no_iter)
        
    if len(output) == 1:
        return output[0]
    else:
        return tuple(output)
