from dashboard import views

import pandas as pd
import numpy as np
from sklearn.metrics import precision_recall_curve, confusion_matrix
import matplotlib.pyplot as plt

texts = [
    "I absolutely loved the experience!", "Very disappointed in the product.",
    "It's okay, nothing special.", "Fantastic service!", "Worst app ever.",
    "Good value for the money.", "Update broke everything!", "Looks beautiful.",
    "Mediocre performance.", "Great support team.", "Neutral opinion.",
    "Packaging damaged, product fine.", "Arrived on time.", "Not impressed.",
    "Exceeded expectations.", "Not bad overall.", "Too laggy.", "Highly recommend.",
    "Still unsure about it.", "Flawless execution.", "Too buggy.",
    "Responsive team.", "Could be better.", "Good balance.", "Worst purchase.",
    "Decent for daily use.", "Support unresponsive.", "Beautiful UI.",
    "Poor under load.", "Clear instructions.", "Terrible experience.",
    "Issues fixed quickly.", "Seems okay so far.", "Hard to use.",
    "Helpful and easy install.", "Nice look, poor function.",
    "Simple and effective.", "Crashes often.", "Amazing product.",
    "Confusing UI.", "Meets expectations.", "No service reply.",
    "Minimal design, fast.", "Laggy and slow.", "Perfect result!",
    "Overpriced.", "Works fine.", "Buggy keyboard.", "Love the update!",
    "Disappointed again.", "Smooth and fast.", "Just average.",
    "Helped a lot.", "Terrible first impression.", "More reliable than expected.",
    "Connection issues.", "Install was easy.", "Needs more options.",
    "Too cluttered.", "Exactly right.", "Bad menu layout.",
    "Meets promises.", "Worst UI design.", "Fast and clean.",
    "No support for old devices.", "Happy with results.", "Missing features.",
    "Fast and intuitive.", "Random errors.", "Setup was easy.",
    "Feels unfinished.", "Very reliable.", "Poor battery usage.",
    "Effective and usable.", "Not compatible.", "Best update ever.",
    "Settings hard to find.", "Consistent performance.", "Lacks documentation.",
    "User-friendly.", "Frequent disconnects.", "Fast load time.",
    "Not what I expected.", "Very pleased.", "Rough edges remain.",
    "Great layout.", "Connection problems.", "Super efficient.",
    "Not intuitive.", "Easy configuration.", "Sync issues.",
    "Well-designed.", "Battery drains quickly.", "Worked instantly!",
    "Too many features.", "Clean interface.", "Hard to set up.",
    "Excellent experience.", "Too basic.", "Would use again.",
    "Stuck loading.", "Stable performance.", "Too slow.",
    "Refreshing change.", "Platform inconsistency.", "Solid feature set.",
    "Crashes gone.", "Too many permissions."
]

# Ground truth labels (manually assigned or randomly for demo)
y_true=[
    1,0,1,1,0,
    1,0,1,1,1,
    1,1,1,0,1,
    1,0,1,1,1,
    0,1,1,1,0,
    1,0,1,0,1,
    0,1,1,0,1,
    1,1,0,1,0,
    1,0,1,0,1,
    0,1,0,1,0,
    1,1,1,0,1,
    0,1,1,0,1,
    0,1,0,1,0,
    1,0,1,0,1,
    0,1,0,1,0,
    1,0,1,0,1,
    0,1,0,1,0,
    1,0,1,0,1,
    0,1,0,1,0,
    1,0,1,0,1,
    0,1,0,1,0,
    1,1,0
    ]
# Map labels from RoBERTa to binary
label_map = {"LABEL_0": 0, "LABEL_1": 1, "LABEL_2": 1}  # Negative=0, Neutral+Positive=1

y_scores = []
y_pred = []

for text in texts:
    result = views.sentiment(text)[0]
    label = result["label"]
    score = result["score"]
    binary_label = label_map[label]
    
    y_pred.append(binary_label)
    y_scores.append(score if binary_label == 1 else 1 - score)  # Ensure higher score = positive

precision, recall, thresholds = precision_recall_curve(y_true, y_scores)

plt.figure(figsize=(8, 5))
plt.plot(recall, precision, marker='.', label='RoBERTa')
plt.title('Precision vs. Recall Curve')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.grid(True)
plt.legend()
plt.show()

cm = confusion_matrix(y_true, y_pred)
print("Confusion Matrix:\n", cm)