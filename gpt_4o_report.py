import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from datetime import datetime
import json
import re

# Set up plotting style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

def load_and_clean_data():
    """Load the evaluation results and clean the data"""
    try:
        df = pd.read_parquet('sampled_evaluations_openai.parquet')
        print(f"Loaded {len(df)} records from parquet")
    except FileNotFoundError:
        print("Error: 'sampled_evaluations_openai.parquet' not found.")
        print("Please run the gpt_4o_analysis.py script first.")
        return None

    results = {}
    valid_scores_from_jsonl = 0
    try:
        with open('openai_batch_results.jsonl', 'r') as f:
            for line in f:
                try:
                    data = json.loads(line)
                    custom_id = data.get('custom_id')
                    if custom_id and data.get('response') and data['response']['body']['choices']:
                        content_str = data['response']['body']['choices'][0]['message']['content']
                        
                        # Clean the string to extract JSON
                        match = re.search(r'\{.*\}', content_str, re.DOTALL)
                        if match:
                            json_str = match.group(0)
                            try:
                                content_json = json.loads(json_str)
                                score = content_json.get('overall_score')
                                explanation = content_json.get('explanation')
                                category_scores = content_json.get('category_scores')
                                results[custom_id] = {'score': score, 'explanation': explanation, 'category_scores': category_scores}
                                if score is not None:
                                    valid_scores_from_jsonl += 1
                            except json.JSONDecodeError:
                                results[custom_id] = {'score': None, 'explanation': 'Error decoding JSON content', 'category_scores': None}
                except (json.JSONDecodeError, KeyError, IndexError):
                    if 'custom_id' in locals() and custom_id:
                        results[custom_id] = {'score': None, 'explanation': 'Error processing batch result line', 'category_scores': None}
                    continue
        print(f"Found {valid_scores_from_jsonl} valid scores in openai_batch_results.jsonl")
    except FileNotFoundError:
        print("Warning: 'openai_batch_results.jsonl' not found. LLM scores will not be loaded.")

    # Map results to the DataFrame
    df['llm_overall_score'] = df.index.map(lambda i: results.get(f'row_{i}', {}).get('score'))
    df['llm_explanation_clean'] = df.index.map(lambda i: results.get(f'row_{i}', {}).get('explanation'))
    df['llm_category_scores'] = df.index.map(lambda i: results.get(f'row_{i}', {}).get('category_scores'))

    # --- Data Cleaning and Type Conversion ---
    df['llm_overall_score_clean'] = pd.to_numeric(df['llm_overall_score'], errors='coerce')
    print(f"{df['llm_overall_score_clean'].notna().sum()} scores successfully mapped to dataframe.")

    df['bid_cpm_clean'] = pd.to_numeric(df['bid_cpm'], errors='coerce')
    df['pay_price_cpm_clean'] = pd.to_numeric(df['pay_price_cpm'], errors='coerce')
    df['cosine_distance_clean'] = pd.to_numeric(df['cosine_distance'], errors='coerce')
    df['sentiment_before_clean'] = pd.to_numeric(df['sentiment_before'], errors='coerce')
    df['sentiment_after_clean'] = pd.to_numeric(df['sentiment_after'], errors='coerce')
    df['attention_shift_clean'] = pd.to_numeric(df['attention_shift'], errors='coerce')
    df['embeddings_shift_norm_clean'] = pd.to_numeric(df['embeddings_shift_norm'], errors='coerce')

    if 'clicked' in df.columns:
        df['clicked_clean'] = df['clicked'].astype(str).str.lower().map({'true': True, 'false': False, '1': True, '0': False}).fillna(False).astype(bool)
    else:
        df['clicked_clean'] = False

    return df

def create_visualizations(df):
    """Create histograms for ad quality scores, bid prices, and other metrics"""
    fig, axes = plt.subplots(3, 2, figsize=(15, 18))
    fig.suptitle('Ad Quality Analysis - OpenAI GPT-4o Results', fontsize=16, fontweight='bold')
    
    # 1. LLM Overall Score Distribution
    valid_scores = df['llm_overall_score_clean'].dropna()
    if not valid_scores.empty:
        sns.histplot(valid_scores, bins=20, ax=axes[0, 0], color='skyblue', kde=True)
        axes[0, 0].set_title('Distribution of LLM Overall Quality Scores')
        axes[0, 0].axvline(valid_scores.mean(), color='red', linestyle='--', label=f'Mean: {valid_scores.mean():.1f}')
        axes[0, 0].legend()
    else:
        axes[0, 0].text(0.5, 0.5, 'No valid LLM scores', ha='center', va='center', transform=axes[0, 0].transAxes)

    # 2. Bid CPM Distribution
    valid_bids = df['bid_cpm_clean'].dropna()
    if not valid_bids.empty:
        sns.histplot(valid_bids, bins=20, ax=axes[0, 1], color='lightgreen', kde=True)
        axes[0, 1].set_title('Distribution of Bid CPM Prices')
        axes[0, 1].axvline(valid_bids.mean(), color='red', linestyle='--', label=f'Mean: ${valid_bids.mean():.2f}')
        axes[0, 1].legend()
    else:
        axes[0, 1].text(0.5, 0.5, 'No valid bid prices available', ha='center', va='center', transform=axes[0, 1].transAxes)

    # 3. Pay Price CPM Distribution
    valid_pay = df['pay_price_cpm_clean'].dropna()
    if not valid_pay.empty:
        sns.histplot(valid_pay, bins=20, ax=axes[1, 0], color='lightcoral', kde=True)
        axes[1, 0].set_title('Distribution of Pay Price CPM')
        axes[1, 0].axvline(valid_pay.mean(), color='red', linestyle='--', label=f'Mean: ${valid_pay.mean():.2f}')
        axes[1, 0].legend()
    else:
        axes[1, 0].text(0.5, 0.5, 'No pay price data', ha='center', va='center', transform=axes[1, 0].transAxes)

    # 4. Cosine Distance Distribution
    valid_cosine = df['cosine_distance_clean'].dropna()
    if not valid_cosine.empty:
        sns.histplot(valid_cosine, bins=20, ax=axes[1, 1], color='lightgreen', kde=True)
        axes[1, 1].set_title('Distribution of Cosine Distance')
        axes[1, 1].axvline(valid_cosine.mean(), color='red', linestyle='--', label=f'Mean: {valid_cosine.mean():.2f}')
        axes[1, 1].legend()
    else:
        axes[1, 1].text(0.5, 0.5, 'No cosine distance data', ha='center', va='center', transform=axes[1, 1].transAxes)

    # 5. Clicks by Ad Quality Bin
    if not valid_scores.empty and 'clicked_clean' in df.columns:
        df['quality_bin'] = pd.cut(df['llm_overall_score_clean'], bins=5, labels=['Very Low', 'Low', 'Medium', 'High', 'Very High'])
        
        ads_per_bin = df['quality_bin'].value_counts().sort_index()
        clicks_per_bin = df.groupby('quality_bin')['clicked_clean'].sum()
        
        ax = axes[2, 0]
        if clicks_per_bin.sum() > 0:
            clicks_per_bin.plot(kind='bar', ax=ax, color='gold')
            ax.set_title(f'Total Clicks by Ad Quality Bin (Total Ads: {len(df)})')
            ax.set_ylabel('Number of Clicks')

            # Create new labels with ad counts
            new_labels = [f'{label.get_text()}\n(n={ads_per_bin.get(label.get_text(), 0)})' for label in ax.get_xticklabels()]
            ax.set_xticklabels(new_labels, rotation=0)
        else:
            ax.text(0.5, 0.5, 'No clicks recorded in this sample', ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Clicks by Ad Quality Bin')
    else:
        axes[2, 0].text(0.5, 0.5, 'Insufficient data for click analysis', ha='center', va='center', transform=axes[2, 0].transAxes)
        axes[2, 0].set_title('Clicks by Ad Quality Bin')

    # Hide the empty subplot
    axes[2, 1].set_visible(False)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig('ad_quality_visuals_openai.png', dpi=300)
    plt.show()

def create_category_scores_chart(df):
    """Create a bar chart of the average category scores."""
    # Explode the category scores into separate columns, ensuring it's a list of dicts
    valid_category_scores = df['llm_category_scores'].dropna().tolist()
    if not valid_category_scores:
        print("No category scores found to plot.")
        return

    category_scores_df = pd.DataFrame(valid_category_scores)
    
    if not category_scores_df.empty:
        avg_category_scores = category_scores_df.mean()
        fig, ax = plt.subplots(figsize=(10, 6))
        avg_category_scores.plot(kind='bar', ax=ax, color='purple', alpha=0.7)
        ax.set_title('Average LLM Category Scores')
        ax.set_xlabel('Category')
        ax.set_ylabel('Average Score (0-10)')
        ax.tick_params(axis='x', rotation=45)
        plt.tight_layout()
        plt.savefig('category_scores_analysis_openai.png', dpi=300, bbox_inches='tight')
        plt.show()

def generate_summary_stats(df):
    stats = {'data_health': {}}
    
    # Data Health Check
    for col in ['sentiment_before', 'sentiment_after', 'attention_shift', 'embeddings_shift_norm']:
        stats['data_health'][col] = {'missing': df[col].isnull().sum(), 'total': len(df)}

    # Key Metrics
    stats['total_ads'] = len(df)
    stats['ctr'] = df['clicked_clean'].mean()
    stats['avg_bid_cpm'] = df['bid_cpm_clean'].mean()
    stats['avg_pay_cpm'] = df['pay_price_cpm_clean'].mean()
    stats['avg_llm_score'] = df['llm_overall_score_clean'].mean()
    stats['avg_cosine_distance'] = df['cosine_distance_clean'].mean()
    stats['avg_sentiment_shift'] = (df['sentiment_after_clean'] - df['sentiment_before_clean']).mean()

    # Correlations
    stats['correlations'] = {
        'score_vs_ctr': df['llm_overall_score_clean'].corr(df['clicked_clean']),
        'score_vs_cosine': df['llm_overall_score_clean'].corr(df['cosine_distance_clean']),
        'score_vs_pay_cpm': df['llm_overall_score_clean'].corr(df['pay_price_cpm_clean'])
    }
    return stats

def create_markdown_report(stats, df):
    report = f"""# Ad Quality and Performance Report

**Generated on:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## 1. Executive Summary
This report provides an analysis of {stats['total_ads']} advertisements, evaluating their quality via LLM scoring and their performance on key metrics.

- **Average Ad Quality Score:** {stats['avg_llm_score']:.1f}/100
- **Overall Click-Through Rate (CTR):** {stats['ctr']:.2%}
- **Average Cosine Distance:** {stats['avg_cosine_distance']:.3f}

## 2. Data Health Check
This section highlights columns with significant missing data, which impacts the analysis.

"""
    for col, data in stats['data_health'].items():
        if data['missing'] == data['total']:
            report += f"- **{col}:** 🚨 100% MISSING DATA. Analysis based on this metric is not possible.\n"
    report += "\n## 3. Key Performance Indicators\n"
    report += f"- **Click-Through Rate (CTR):** {stats['ctr']:.2%}\n"
    report += f"- **Average Bid CPM:** ${stats['avg_bid_cpm']:.2f}\n"
    report += f"- **Average Paid CPM:** ${stats['avg_pay_cpm']:.2f}\n"

    report += "\n## 4. Ad Impact Analysis\n"
    report += f"- **Average Cosine Distance:** {stats['avg_cosine_distance']:.3f} (Lower is better, indicates less topic drift)\n"
    if not np.isnan(stats['avg_sentiment_shift']):
        report += f"- **Average Sentiment Shift:** {stats['avg_sentiment_shift']:.3f} (Positive indicates improved sentiment after ad)\n"
    else:
        report += "- **Average Sentiment Shift:** Not available due to missing data.\n"

    report += "\n## 5. Correlation Analysis\nKey correlations with the LLM Overall Score:\n\n"
    report += f"- **Score vs. CTR:** {stats['correlations']['score_vs_ctr']:.3f}\n"
    report += f"- **Score vs. Cosine Distance:** {stats['correlations']['score_vs_cosine']:.3f}\n"
    report += f"- **Score vs. Paid CPM:** {stats['correlations']['score_vs_pay_cpm']:.3f}\n"

    report += "\n## 6. Visual Analysis\n![Ad Quality Visuals](ad_quality_visuals_openai.png)\n"
    report += "\n### Average LLM Category Scores\n![Category Scores](category_scores_analysis_openai.png)"

    report += "\n## 7. Sampled Ads for Review\n"
    # Get sample of lowest scoring ads for human review
    low_scoring_ads = df.nsmallest(3, 'llm_overall_score_clean')[['creative', 'llm_overall_score_clean', 'llm_explanation_clean', 'query', 'context', 'user_message_before', 'bot_message_before']].copy()
    
    # Get sample of highest scoring ads for comparison
    high_scoring_ads = df.nlargest(2, 'llm_overall_score_clean')[['creative', 'llm_overall_score_clean', 'llm_explanation_clean', 'query', 'context', 'user_message_before', 'bot_message_before']].copy()

    for title, sample_df in [("Top 3 Highest Scoring Ads", high_scoring_ads), 
                             ("Top 3 Lowest Scoring Ads", low_scoring_ads)]:
        report += f"### {title}\n"
        for _, row in sample_df.iterrows():
            report += f"- **Score:** {row['llm_overall_score_clean']:.1f} | **Creative:** {row['creative'][:100]}... | **Explanation:** {row['llm_explanation_clean']}\n"
            report += f"  - **User Message Before:** {str(row['user_message_before'])[:256]}\n"
            report += f"  - **Bot Message Before:** {str(row['bot_message_before'])[:256]}\n"

    return report

def main():
    """Main analysis function"""
    print("Loading and cleaning ad quality data...")
    df = load_and_clean_data()
    
    if df is None:
        return

    print("Generating summary statistics...")
    stats = generate_summary_stats(df)
    
    print("Creating visualizations...")
    create_visualizations(df)
    create_category_scores_chart(df)
    
    print("Creating markdown report...")
    report = create_markdown_report(stats, df)
    
    with open('ad_quality_report_openai.md', 'w', encoding='utf-8') as f:
        f.write(report)
    
    print("\nAnalysis complete!")
    print(f"- Report saved to: ad_quality_report_openai.md")
    print(f"- Visuals saved to: ad_quality_visuals_openai.png")
    print(f"- Category scores chart saved to: category_scores_analysis_openai.png")

if __name__ == "__main__":
    main()