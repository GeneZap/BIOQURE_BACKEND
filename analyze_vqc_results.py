import os
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import numpy as np

def analyze_vqc_predictions(folder_path):
    """Analyze VQC validation predictions from a given folder"""
    
    # Find the CSV file in the folder
    csv_files = [f for f in os.listdir(folder_path) if f.endswith('vqc_validation_predictions.csv')]
    
    if not csv_files:
        print(f"No vqc_validation_predictions.csv found in {folder_path}")
        return None
    
    csv_file = csv_files[0]
    file_path = os.path.join(folder_path, csv_file)
    
    # Read the CSV file
    df = pd.read_csv(file_path)
    
    # Extract parameters from folder name
    folder_name = os.path.basename(folder_path)
    
    # Calculate metrics
    y_true = df['y_true']
    y_pred_050 = df['prediction_at_0_50']
    y_pred_threshold = df['prediction_at_selected_threshold']
    
    # Calculate accuracy for both prediction methods
    acc_050 = accuracy_score(y_true, y_pred_050)
    acc_threshold = accuracy_score(y_true, y_pred_threshold)
    
    # Calculate precision and recall for both methods
    prec_050 = precision_score(y_true, y_pred_050, zero_division=0)
    prec_threshold = precision_score(y_true, y_pred_threshold, zero_division=0)
    
    rec_050 = recall_score(y_true, y_pred_050, zero_division=0)
    rec_threshold = recall_score(y_true, y_pred_threshold, zero_division=0)
    
    f1_050 = f1_score(y_true, y_pred_050, zero_division=0)
    f1_threshold = f1_score(y_true, y_pred_threshold, zero_division=0)
    
    # Extract seed and method from folder name
    params = {
        'folder': folder_name,
        'seed': None,
        'method': 'threshold',
        'accuracy_050': acc_050,
        'accuracy_threshold': acc_threshold,
        'precision_050': prec_050,
        'precision_threshold': prec_threshold,
        'recall_050': rec_050,
        'recall_threshold': rec_threshold,
        'f1_050': f1_050,
        'f1_threshold': f1_threshold
    }
    
    # Parse folder name to extract seed and method
    if 'seed' in folder_name:
        seed_str = [s for s in folder_name.split('_') if s.startswith('seed')]
        if seed_str:
            params['seed'] = int(seed_str[0].replace('seed', ''))
    
    if 'linear' in folder_name:
        params['method'] = 'linear'
    elif 'circular' in folder_name:
        params['method'] = 'circular'
    elif 'threshold' in folder_name:
        params['method'] = 'threshold'
    elif 'baseline' in folder_name:
        params['method'] = 'baseline'
    
    return params

def main():
    # Define the base directory
    base_dir = "D:/PROGRAMMING/PROJECTS/SIH/gdc_download_20260928_181302.288321/training_outputs"
    
    # Get all subdirectories
    folders = [os.path.join(base_dir, d) for d in os.listdir(base_dir) if 
               os.path.isdir(os.path.join(base_dir, d))]
    
    results = []
    
    print("Analyzing VQC validation predictions across all parameter sets...")
    print("=" * 80)
    
    # Analyze each folder
    for folder in folders:
        try:
            result = analyze_vqc_predictions(folder)
            if result:
                results.append(result)
                print(f"Folder: {result['folder']}")
                print(f"  Method: {result['method']}")
                print(f"  Seed: {result['seed']}")
                print(f"  Accuracy (0.50 threshold): {result['accuracy_050']:.4f}")
                print(f"  Accuracy (selected threshold): {result['accuracy_threshold']:.4f}")
                print(f"  F1 Score (0.50 threshold): {result['f1_050']:.4f}")
                print(f"  F1 Score (selected threshold): {result['f1_threshold']:.4f}")
                print("-" * 50)
        except Exception as e:
            print(f"Error processing {folder}: {str(e)}")
    
    # Create summary DataFrame
    if results:
        df_results = pd.DataFrame(results)
        
        # Sort by accuracy with selected threshold (highest first)
        df_results_sorted = df_results.sort_values('accuracy_threshold', ascending=False)
        
        print("\n" + "=" * 80)
        print("SUMMARY - Top 5 Parameter Sets by Accuracy (Selected Threshold)")
        print("=" * 80)
        
        top_5 = df_results_sorted.head(5)
        for idx, row in top_5.iterrows():
            print(f"{idx+1}. Folder: {row['folder']}")
            print(f"   Method: {row['method']}, Seed: {row['seed']}")
            print(f"   Accuracy: {row['accuracy_threshold']:.4f}")
            print(f"   F1 Score: {row['f1_threshold']:.4f}")
            print()
        
        # Save results to CSV
        output_file = "vqc_analysis_results.csv"
        df_results_sorted.to_csv(output_file, index=False)
        print(f"Full results saved to {output_file}")
        
        # Find the best performing set of parameters
        best_result = df_results_sorted.iloc[0]
        print("=" * 80)
        print("BEST PERFORMING PARAMETERS")
        print("=" * 80)
        print(f"Best folder: {best_result['folder']}")
        print(f"Best method: {best_result['method']}")
        print(f"Best seed: {best_result['seed']}")
        print(f"Best accuracy: {best_result['accuracy_threshold']:.4f}")
        print(f"Best F1 score: {best_result['f1_threshold']:.4f}")
        
        return df_results_sorted
    else:
        print("No results to analyze")
        return None

if __name__ == "__main__":
    main()