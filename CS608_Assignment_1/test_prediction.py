# Importing necessary libraries
import cornac
from cornac.models import MF, BPR

# load model from file
model = BPR.load("CS608_Assignment_1/models/BPR/2025-05-11_05-48-16-657191.pkl")
print("Model loaded successfully.")

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