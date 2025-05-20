import cornac
from cornac.models import MF, PMF, BPR, WMF, MMMF, EASE
import argparse
import pandas as pd


# Evaluate input arguments
def evaluate_input(args):
    """
    Evaluate the input arguments for the model and parameters.
    Args:
        args: parsed command line arguments.
    Returns:
        model = The loaded model.
    """
    # Load the model based on the provided model name
    if args.modelname == 'MF':
        model = MF.load(args.modelpath)
    elif args.modelname == 'PMF':
        model = PMF.load(args.modelpath)
    elif args.modelname == 'BPR':
        model = BPR.load(args.modelpath)
    elif args.modelname == 'WMF':       
        model = WMF.load(args.modelpath)
    elif args.modelname == 'EASE':
        model = EASE.load(args.modelpath)
    else:
        raise ValueError(f"Model {args.modelname} is not supported.")
    return model


def perform_predictions(models):
    """
    Perform predictions for each model.
    Args:
        models: list of loaded models.
    Returns:
        recommendations: Dataframe of recommendations with columns=['user_id, 'item_1', ..., 'item_k'].
    """
    K = 50
    columns = ['user_id'] + [f'item_{i}' for i in range(1, K+1)]
    user_ranking = pd.DataFrame(columns=columns)

    for user_id in range(1, 11526):
        item_ranking = {} # item_id: [ranking 1, ranking 2]

        # Step 1: Predict scores per model
        for model in models:
            top_items = model.recommend(user_id=str(user_id), k=K*2)
            
            # Add top_items with their rankings to item_ranking
            for item_id in top_items:
                if item_id in item_ranking:
                    item_ranking[item_id].append(top_items.index(item_id) + 1)
                else:
                    item_ranking[item_id] = [top_items.index(item_id) + 1]

        # Step 2: Calculate Borda Count:
        borda_ranking = [] # (item_id, score)
        for item_id, rankings in item_ranking.items():
            borda_score = sum(K - rank for rank in rankings)
            borda_ranking.append((item_id, borda_score))

        # Step 3: Sort by Borda Count
        borda_ranking.sort(key=lambda x: x[1], reverse=True)

        # Step 4: Add item_ids to user_ranking
        new_row = [user_id] + [item_id for item_id, _ in borda_ranking[:K]]
        user_ranking.loc[len(user_ranking)] = new_row

    return user_ranking


if __name__ == "__main__":
    # Setting up the command line argument parser
    parser = argparse.ArgumentParser(description="Perform prediction with multiple models.")
    parser.add_argument('--modelnames', type=str, nargs='+', required=True, help='List of model names (MF, PMF, BPR, WMF, EASE)')
    parser.add_argument('--modelpaths', type=str, nargs='+', required=True, help='List of paths to the model files (order must match modelnames)')
    args = parser.parse_args()

    # Check that the number of model names matches the number of model paths
    if len(args.modelnames) != len(args.modelpaths):
        raise ValueError("The number of model names must match the number of model paths.")

    # Evaluate input arguments for each model
    models = []
    for modelname, modelpath in zip(args.modelnames, args.modelpaths):
        single_args = argparse.Namespace(modelname=modelname, modelpath=modelpath)
        models.append(evaluate_input(single_args))
    print("Models loaded successfully.")
    print(models)

    # Perform predictions for each model
    print("Performing predictions...")
    recommendations = perform_predictions(models)
    print("Recommendations generated successfully.")

    # Write recommendations to file
    recommendations_sorted = recommendations.sort_values(by='user_id')
    recommendations_sorted.to_csv('user_ranking.txt', sep='\t', index=False, columns=list(recommendations.columns[1:]), header=False)
    print("Recommendations written to user_ranking.txt")

