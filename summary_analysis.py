from dashboard import views

from rouge_score import rouge_scorer
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df1 = pd.read_csv("summarization_long_paragraphs.csv") 

# User Input
texts = df1["Paragraph"].tolist()
reference_summaries = df1["Reference_Summary"].tolist() # reference summaries(ground truth)

generated_summaries = []
# Your generated summaries
for text in texts:
    generated_summary = views.summarizer(text)
    generated_summaries.append(generated_summary)


# Initialize scorer
scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)

results = []

# Compare each generated vs. reference
for ref, gen in zip(reference_summaries, generated_summaries):
    scores = scorer.score(ref, gen)
    results.append({
        "ROUGE-1 (F)": scores["rouge1"].fmeasure,
        "ROUGE-2 (F)": scores["rouge2"].fmeasure,
        "ROUGE-L (F)": scores["rougeL"].fmeasure
    })

# Display as DataFrame
df = pd.DataFrame(results)
print(df)
print("\nAverage ROUGE Scores:")
print(df.mean())

plt.figure(figsize=(10, 6))
plt.plot(df["ROUGE-1 (F)"], label="ROUGE-1")
plt.plot(df["ROUGE-2 (F)"], label="ROUGE-2")
plt.plot(df["ROUGE-L (F)"], label="ROUGE-L")
plt.title("ROUGE Score Across Summarized Examples")
plt.xlabel("Example Index")
plt.ylabel("ROUGE F1 Score")
plt.legend()
plt.grid(True)
plt.show()