import pandas as pd
import os
import re
from tqdm import tqdm
import numpy as np
from itertools import combinations

from sentence_transformers import SentenceTransformer
from bert_score import BERTScorer

model = SentenceTransformer("all-MiniLM-L6-v2")

# Initialize BERTScorer for Chinese language
scorer = BERTScorer(model_type='bert-base-chinese', lang='zh', rescale_with_baseline=True)

def calculate_total_diversity(selected_questions):
    """
    Calculate the total diversity of a set of selected questions.
    
    Parameters:
    selected_questions (list): A list of selected questions.
    
    Returns:
    float: The total diversity score based on pairwise distances.
    """
    
    precision_scores = []
    
    # Calculate the pairwise distance for each unique pair of questions
    for i in range(len(selected_questions)):
        for j in range(i + 1, len(selected_questions)):
            precision, recall, F1 = scorer.score([selected_questions[i]], [selected_questions[j]])
            
            precision_scores.append(1 - precision.mean().item())  # Semantic distance

    return np.sum(precision_scores)

def select_best_combination(candidates, selection_size):
    best_combination = None
    best_diversity_score = -1
    
    # Evaluate all combinations of the given selection size
    for combination in combinations(candidates, selection_size):
        diversity_score = calculate_total_diversity(combination)
        if diversity_score > best_diversity_score:
            best_diversity_score = diversity_score
            best_combination = combination
    
    return best_combination, best_diversity_score

if __name__ == "__main__":
    budget = 5  # Total budget
    file_path = './selection_eval_data.csv'
    output_path = f'./best_select_questions_20_{budget}.csv'
    
    df = pd.read_csv(file_path)
    
    results = []

    if not os.path.isfile(output_path):
        results_df = pd.DataFrame(columns=['question', 'selected', 'diversity'])
        results_df.to_csv(output_path, index=False)
    
    for index, row in tqdm(df.iterrows(), total=df.shape[0], desc="Processing Rows"):
        candidates = re.split(r'\s*\d+\.\s*', row['generated similar question'])  # Adjust the column name as needed
        
        cleaned_questions = [q.strip() for q in candidates if q.strip()]

        if len(cleaned_questions) < budget:
            selected_questions = cleaned_questions
            total_diversity = calculate_total_diversity(selected_questions)
        else:
            selected_questions, total_diversity = select_best_combination(cleaned_questions, budget)
        
        print(f"Best Selected Questions:")
        for question in selected_questions:
            print(question)
        
        result = {
            "question": row["question"],  # Original question from the row
            "selected": ", ".join(selected_questions),  # Join selected questions into a string
            "diversity": total_diversity  # Total diversity calculated from pairwise F1 scores
        }
        results.append(result)

    results_df = pd.DataFrame(results)
    
    print(results_df["diversity"].mean())

    results_df.to_csv(output_path, index=False)

    print(f"Results saved to {output_path}.")
    
    # Calculate the mean of the 'diversity' column
    mean_diversity = results_df['diversity'].mean()

    print(f"#############\nMean diversity of {output_path}:", mean_diversity)
