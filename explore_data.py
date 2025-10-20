import pandas as pd

def explore_data(parquet_file_path):
    """Reads a parquet file and prints out information about the data."""
    try:
        df = pd.read_parquet(parquet_file_path)
        print(f"Successfully loaded {parquet_file_path}\n")

        print("--- Columns ---")
        print(df.columns)
        print("\n")

        print("--- First 5 Rows ---")
        print(df.head())
        print("\n")

        print("--- Data Info ---")
        df.info()
        print("\n")

        print("--- Descriptive Statistics for Key Columns ---")
        key_cols = [
            'context_score', 'bid_cpm', 'pay_price_cpm', 'attention_shift', 
            'alignment_score_norm', 'embeddings_shift_norm', 'lexicon_shift_norm',
            'sentiment_after', 'sentiment_before', 'cosine_distance', 'llm_overall_score'
        ]
        # Filter out columns that are not in the dataframe
        key_cols_exist = [col for col in key_cols if col in df.columns]
        print(df[key_cols_exist].describe())
        print("\n")

        if 'clicked' in df.columns:
            print("--- Clicked Value Counts ---")
            print(df['clicked'].value_counts(normalize=True))
            print("\n")

    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    # The analysis script saves the evaluated data to this file
    explore_data('sampled_evaluations_openai.parquet')
