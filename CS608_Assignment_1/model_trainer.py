import cornac
from cornac.data import Reader
from cornac.data import Dataset
from cornac.models import MF, PMF, BPR, WMF, MMMF, EASE
from cornac.eval_methods import BaseMethod, RatioSplit
from cornac.hyperopt import RandomSearch, GridSearch, Discrete, Continuous
from cornac.metrics import RMSE, AUC, FMeasure, NCRR, NDCG, Recall
from cornac.experiment import Experiment

import pandas as pd
import numpy as np
import ast
import argparse
import regex

# Evaluate input arguments
def evaluate_input(args):
    """
    Evaluate the input arguments for the model and parameters.
    Args:
        args: parsed command line arguments.
    Returns:
        model_name (str): The name of the model to be used.
        param_dict (dict): A dictionary of parameters for the model.
    """
    # Validate model name
    valid_models = ['MF', 'PMF', 'BPR', 'WMF', 'MMMF', 'EASE']
    if args.model not in valid_models:
        raise ValueError(f"Model {args.model} is not supported. Choose from {valid_models}.")
    # Validate params argument
    if not args.params:
        raise ValueError("Parameters must be provided as a string.")
    # Extract parameters from the string
    r = regex.compile(r"(\w+):([\w.]+)")
    params = regex.findall(r, args.params)
    param_dict = {}
    # Convert model parameters to a dictionary & validate their types
    for param in params:
        if len(param) != 2:
            raise ValueError(f"Invalid parameter format: {param}. Expected format is 'key:value'.")
        key, value = param
        # Convert value to appropriate type
        if value.isdigit():
            value = int(value)
        elif value.replace('.', '', 1).isdigit():
            value = float(value)
        elif value.lower() == 'true' or value.lower() == 'false':
            value = bool(value)
        param_dict[key] = value
    # Validate eval_params argument
    eval_dict = {}
    if args.eval_params:
        # Extract training parameters from the string
        r = regex.compile(r"(\w+):([\w.]+)")
        eval_params = regex.findall(r, args.eval_params)
        # Convert training parameters to a dictionary & validate their types
        for param in eval_params:
            if len(param) != 2:
                raise ValueError(f"Invalid parameter format: {param}. Expected format is 'key:value'.")
            key, value = param
            # Convert value to appropriate type
            if value.isdigit():
                value = int(value)
            elif value.replace('.', '', 1).isdigit():
                value = float(value)
            elif value.lower() == 'true' or value.lower() == 'false':
                value = bool(value)
            eval_dict[key] = value
    return args.model, param_dict, eval_dict


# Loading datasets
def load_data():
    """
    Load the training and probe datasets and combine them into a single dataset.
    Returns:
        train_data (list(tuples)): The train dataset containing user-item interactions.
        test_data (list(tuples)): The test dataset containing user-item interactions.
        data (pd.DataFrame): A DataFrame representation of the combined dataset.
    """
    # Load the datasets
    train_data = Reader().read(fpath='./cs608_ip_train_v3.csv', fmt='UIR', sep=',', skip_lines=1, id_inline=False, parser=None)
    test_data = Reader().read(fpath='./cs608_ip_probe_v3.csv', fmt='UIR', sep=',', skip_lines=1, id_inline=False, parser=None)
    # Combine the train and probe datasets
    full_data = train_data + test_data
    # Convert the combined dataset to a DataFrame
    data_df = pd.DataFrame(full_data, columns=['user', 'item', 'rating'])
    print("Data loaded successfully.")
    print(f"Number of unique users: {data_df['user'].nunique()}")
    print(f"Number of unique items: {data_df['item'].nunique()}")
    print(f"Number of training interactions: {len(train_data)}")
    print(f"Number of test interactions: {len(test_data)}")
    return train_data, test_data, full_data


def choose_model(model_name: str, parameters: dict):
    """
    Choose a model based on the provided model name and parameters.
    Args:
        model_name (str): The name of the model to be used.
        parameters (dict): The parameters for the chosen model.
    Returns:
        cornac.models.Model: The chosen model.
    """
    if model_name == 'MF':
        return MF(**parameters)
    elif model_name == 'PMF':
        return PMF(**parameters)
    elif model_name == 'BPR':
        return BPR(**parameters)
    elif model_name == 'WMF':
        return WMF(**parameters)
    elif model_name == 'MMMF':
        return MMMF(**parameters)
    elif model_name == 'EASE':
        return EASE(**parameters)
    else:
        raise ValueError(f"Model {model_name} is not supported.")


def perform_experiment(model, eval_method, metrics, verbose=True, save_results=False):
    """
    Perform an experiment with the given model and dataset.
    Args:
        model (cornac.models.Model): The model to be evaluated.
        dataset (list(tuples)): The dataset to be used for evaluation.
        eval_method (cornac.eval_methods.BaseMethod): The evaluation method to be used.
        metrics (list): A list of metrics to be used for evaluation.
    Returns:
        cornac.experiment.Experiment: The experiment object containing the results.
    """
    # check if inputs are valid
    if not isinstance(model, cornac.models.recommender.Recommender):
        raise ValueError("Model must be an instance of cornac.models.Model")
    if not isinstance(eval_method, cornac.eval_methods.base_method.BaseMethod):
        raise ValueError("Eval method must be an instance of cornac.eval_methods.BaseMethod")
    if not isinstance(metrics, list):
        raise ValueError("Metrics must be a list of cornac.metrics.Metric instances")
    # Create an experiment
    experiment = Experiment(
        models=[model],
        eval_method=eval_method,
        metrics=metrics,
        user_based=True,
        verbose=verbose,
        save_dir="models" if save_results else None,
    )
    # Run the experiment
    experiment.run()
    return

if __name__ == "__main__":

    # Set up argument parser
    parser = argparse.ArgumentParser(description="Train and evaluate a recommendation model.")
    parser.add_argument('--model', type=str, required=True, help='Model name (MF, PMF, BPR, WMF, MMMF, EASE)')
    parser.add_argument('--params', type=str, required=True, help='Model parameters as a string (e.g., k:10,max_iter:100')
    parser.add_argument('--eval_params', type=str, default=None, help='Evaluation parameters as a string (e.g., threshold:1.5)')
    args = parser.parse_args()

    model_name, param_dict, eval_dict = evaluate_input(args)

    # Load the data
    train_data, test_data, full_data = load_data()

    # Choose the model
    model = choose_model(model_name, param_dict)
    print(f"Built model: {model} with parameters: {param_dict}")

    # Hyperparameter tuning (optional)  // currently not used
    # hyperparameter_tuning = input("Do you want to perform hyperparameter tuning? (random/grid/no): ").strip().lower()
    hyperparameter_tuning = 'no'
    if hyperparameter_tuning not in ['random', 'grid']:
        print("No hyperparameter tuning...")
        # evaluation method without validation set
        eval_method = BaseMethod.from_splits(
            train_data=full_data, test_data=test_data, val_data=None, 
            fmt='UIR', rating_threshold=eval_dict.get('threshold', 2.5), exclude_unknowns=eval_dict.get('exclude_unknowns', False),
            random_state=42
        )

    else:
        # Split the data into train, validation, and test sets
        eval_method = RatioSplit(
            data=full_data,
            test_size=0.15,
            val_size=0.15,
            exclude_unknowns=True,
            random_state=42
        )

        # Get user input for hyperparameter tuning
        hyperparameters = [
            Discrete(name='lamb', values=[1500, 3000]),
            ]

        try:
            if hyperparameter_tuning == 'random':
                print("Performing hyperparameter tuning using random search...")
                trials = int(input("Enter the number of trials for random search (e.g., 20): "))
                # Random Search
                model = RandomSearch(
                    model=model,
                    space=hyperparameters,
                    metric=NDCG(k=50),
                    eval_method=eval_method,
                    n_trails=trials,
                )

            elif hyperparameter_tuning == 'grid':
                print("Performing hyperparameter tuning using grid search...")
                # Grid Search
                model = GridSearch(
                    model=model,
                    space=hyperparameters,
                    metric=NDCG(k=50),
                    eval_method=eval_method,
                )

        except Exception as e:
            raise ValueError(f"Hyperparameter Tuning setup broken: {e}")


    # Define metrics
    metrics = [RMSE(), AUC(), FMeasure(k=50), NCRR(k=50), NDCG(k=50), Recall(k=50)]
    
    # Perform the experiment
    perform_experiment(model, eval_method, metrics, verbose=True, save_results=True)
    print("Experiment completed successfully.")