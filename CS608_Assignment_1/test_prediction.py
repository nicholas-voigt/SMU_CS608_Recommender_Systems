# Importing necessary libraries
import cornac
from cornac.models import MF, PMF, BPR, WMF, MMMF, EASE
import argparse


# Setting up the command line argument parser
parser = argparse.ArgumentParser(description="Perform prediction.")
parser.add_argument('--modelname', type=str, required=True, help='Model name (MF, PMF, BPR, WMF, EASE)')
parser.add_argument('--modelpath', type=str, required=True, help='Path to the model file')
args = parser.parse_args()
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

print("Model loaded successfully.")
print("Performing prediction...")

# make recommendations for each user and write them to file
recommendations = []
for user in range(1, 11526):
    # get the top 50 items for the user
    top_items = model.recommend(user_id=str(user), k=50)
    recommendations.append(top_items)

# write recommendations to file
with open("recommendations.txt", "w") as f:
    for rec in recommendations:
        # Convert each list of recommendations to a space-separated string
        line = " ".join(str(item) for item in rec)
        # Write the line followed by a newline
        f.write(line + "\n")

print("Recommendations written to recommendations.txt")