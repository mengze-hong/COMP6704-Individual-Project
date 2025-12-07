import numpy as np
import pandas as pd
import os
import re
from itertools import combinations
from tqdm import tqdm

from sentence_transformers import SentenceTransformer
from bert_score import BERTScorer

# Load the SentenceTransformer model
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


if __name__ == "__main__":
    # Define selection size
    selection_size = 10  # Number of questions to select

    file_path = './selection_eval_data.csv'
    output_path = f'./exhaustive_selected_questions_20_{selection_size}.csv'
    
    df = pd.read_csv(file_path)
    
    # Initialize results list
    results = []

    # Check if the output file already exists
    if not os.path.isfile(output_path):
        results_df = pd.DataFrame(columns=['question', 'selected', 'diversity'])
        results_df.to_csv(output_path, index=False)
    
    # Process each row in the DataFrame
    for index, row in tqdm(df.iterrows(), total=df.shape[0], desc="Processing Rows"):
        # Split the generated similar questions
        candidates = re.split(r'\s*\d+\.\s*', row['generated similar question'])  # Adjust the column name as needed
        
        # Clean up the questions (remove any leading/trailing whitespace and empty strings)
        cleaned_questions = [q.strip() for q in candidates if q.strip()]
        
        # Generate all possible combinations of the cleaned questions
        all_combinations = combinations(cleaned_questions, selection_size)
        # y = [i for i in all_combinations]
        # print(y[0])
        # print(len(y))
        # print(calculate_total_diversity((y[0])))
                
        highest_diversity = -1
        
        # Evaluate each combination
        for selected_questions in tqdm(all_combinations):
            # Calculate total diversity for the selected combination
            total_diversity = calculate_total_diversity(selected_questions)
            if total_diversity > highest_diversity:
                highest_diversity = total_diversity
                print(highest_diversity)
            
            # Prepare results for saving to CSV
            result = {
                "question": row["question"],  # Original question from the row
                "selected": ", ".join(selected_questions),  # Join selected questions into a string
                "diversity": total_diversity  # Total diversity calculated from pairwise F1 scores
            }
            results.append(result)  # Append the result dictionary to the results list
        break
    # Create a DataFrame from the results
    results_df = pd.DataFrame(results)
    
    # Print the mean diversity score
    print(f"Mean Diversity: {results_df['diversity'].mean()}")

    # Save the DataFrame to CSV
    results_df.to_csv(output_path, index=False)

    print(f"Results saved to {output_path}.")
    
    # Calculate the mean of the 'diversity' column
    mean_diversity = results_df['diversity'].mean()

    print(f"#############\nMean diversity of {output_path}:", mean_diversity)
