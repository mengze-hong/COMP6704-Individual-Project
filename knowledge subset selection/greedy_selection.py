import pandas as pd
import os
import re
from tqdm import tqdm
import numpy as np

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
            precison, recall, F1 = scorer.score([selected_questions[i]], [selected_questions[j]])
            
            precision_scores.append(1 - precison.mean().item())  # Semantic distance

    return np.sum(precision_scores)



# Define the cost function
def cost(q):
    return 1  # Assuming each question has a cost of 1 for simplicity

# Define the distance function using BERTScore
def dist(candidate, selected):
    if not selected:
        return 0.0
    
    # Score returns precision, recall, F1 as tensors
    P, R, F1 = scorer.score([candidate] * len(selected), selected)
    return 1 - P.mean().item()

# Greedy algorithm for maximizing pairwise diversity
def greedy_maximize_diversity(candidates, budget, selection_size):
    S = []  # Selected subset
    remaining_budget = budget

    while remaining_budget > 0 and len(S) < selection_size:
        best_candidate = None
        best_value = -1  # Initialize to a very low value

        for candidate in candidates:
            if candidate in S:
                continue
            
            # Calculate the diversity gain over the cost
            diversity_gain = dist(candidate, S)  # Use the updated distance function
            candidate_cost = cost(candidate)

            if candidate_cost <= remaining_budget:
                value = diversity_gain / candidate_cost
                
                if value > best_value:
                    best_value = value
                    best_candidate = candidate

        if best_candidate is not None:
            S.append(best_candidate)
            remaining_budget -= cost(best_candidate)

    return S

if __name__ == "__main__":
    budget = 10  # Total budget
    file_path = './selection_eval_data.csv'
    output_path = f'./greedy_selected_questions_20_{budget}.csv'
    
    df = pd.read_csv(file_path)
    
    results = []
    
    # Preprocess the data to remove duplicate titles
    df = df.drop_duplicates(subset=['question']).reset_index(drop=True)

    # Check if the output file already exists
    if not os.path.isfile(output_path):
        results_df = pd.DataFrame(columns=['question', 'selected', 'diversity'])
        results_df.to_csv(output_path, index=False)
    
    # Process each row in the DataFrame
    for index, row in tqdm(df.iterrows(), total=df.shape[0], desc="Processing Rows"):
        # Split the generated similar questions, removing the leading numbers
        candidates = re.split(r'\s*\d+\.\s*', row['generated similar question'])  # Adjust the column name as needed
        
        # Clean up the questions (remove any leading/trailing whitespace and empty strings)
        cleaned_questions = [q.strip() for q in candidates if q.strip()]

        # Get selected questions and total diversity
        selected_questions = greedy_maximize_diversity(cleaned_questions, budget, budget)
        avg_diversity = calculate_total_diversity(selected_questions)
        
        print(cleaned_questions)
        print(f"Total Diversity: {avg_diversity:.4f}")
        print("Selected Questions:")
        for question in selected_questions:
            print(question)
        
        result = {
            "question": row["question"],  # Original question from the row
            "selected": ", ".join(selected_questions),  # Join selected questions into a string
            "diversity": avg_diversity  # Total diversity
        }
        results.append(result)

    results_df = pd.DataFrame(results)

    results_df.to_csv(output_path, index=False)

    print(f"Results saved to {output_path}.")
    
    # Calculate the mean of the 'diversity' column
    mean_diversity = results_df['diversity'].mean()

    print(f"#############\nMean diversity of {output_path}:", mean_diversity)
