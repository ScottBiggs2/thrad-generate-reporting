import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from datetime import datetime
import json

# Set up plotting style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

def load_and_clean_data():
    """Load the evaluation results and clean the data"""
    df = pd.read_parquet('sampled_evaluations_openai.parquet')
    print(f"Loaded {len(df)} records from parquet")

    results = {}
    with open('openai_batch_results.jsonl', 'r') as f:
        for line in f:
            try:
                data = json.loads(line)
                custom_id = data.get('custom_id')
                if custom_id and data.get('response') and data['response']['body']['choices']:
                    content_str = data['response']['body']['choices'][0]['message']['content']
                    
                    # Clean the string to extract JSON
                    if '```json' in content_str:
                        content_str = content_str.split('```json')[1].strip()
                    if '```' in content_str:
                        content_str = content_str.split('```')[0].strip()

                    try:
                        content_json = json.loads(content_str)
                        score = content_json.get('overall_score')
                        explanation = content_json.get('explanation')
                        category_scores = content_json.get('category_scores')
                        results[custom_id] = {'score': score, 'explanation': explanation, 'category_scores': category_scores}
                    except json.JSONDecodeError:
                        results[custom_id] = {'score': None, 'explanation': 'Error decoding JSON content', 'category_scores': None}

            except (json.JSONDecodeError, KeyError, IndexError):
                # Handle cases where the line is not valid JSON or structure is wrong
                if 'custom_id' in locals() and custom_id:
                    results[custom_id] = {'score': None, 'explanation': 'Error processing batch result line', 'category_scores': None}
                continue

    # Map results to the DataFrame
    # Assuming custom_id is like 'row_0', 'row_1', etc.
    df['llm_overall_score'] = df.index.map(lambda i: results.get(f'row_{i}', {}).get('score'))
    df['llm_explanation'] = df.index.map(lambda i: results.get(f'row_{i}', {}).get('explanation'))
    df['llm_category_scores'] = df.index.map(lambda i: results.get(f'row_{i}', {}).get('category_scores'))

    print(f"Columns: {df.columns.tolist()}")
    
    # Check LLM results
    valid_llm_scores = df['llm_overall_score'].notna()
    print(f"Valid LLM scores from JSONL: {valid_llm_scores.sum()}/{len(df)}")
    
    # Clean LLM scores - convert to numeric, handle errors
    df['llm_overall_score_clean'] = pd.to_numeric(df['llm_overall_score'], errors='coerce')
    
    # Clean bid prices
    df['bid_cpm_clean'] = pd.to_numeric(df['bid_cpm'], errors='coerce')
    df['pay_price_cpm_clean'] = pd.to_numeric(df['pay_price_cpm'], errors='coerce')
    
    # Clean clicked column - convert boolean strings to actual booleans
    if 'clicked' in df.columns:
        df['clicked_clean'] = df['clicked'].astype(str).str.lower().map({'true': True, 'false': False, '1': True, '0': False})
    else:
        df['clicked_clean'] = False
    
    return df

def create_histograms(df):
    """Create histograms for ad quality scores and bid prices"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Ad Quality Analysis - OpenAI GPT-4o Results', fontsize=16, fontweight='bold')
    
    # 1. LLM Overall Score Distribution
    valid_scores = df['llm_overall_score_clean'].dropna()
    if len(valid_scores) > 0:
        axes[0, 0].hist(valid_scores, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
        axes[0, 0].set_title('Distribution of LLM Overall Quality Scores (OpenAI GPT-4o)')
        axes[0, 0].set_xlabel('Overall Score (0-100)')
        axes[0, 0].set_ylabel('Frequency')
        axes[0, 0].axvline(valid_scores.mean(), color='red', linestyle='--',
                          label=f'Mean: {valid_scores.mean():.1f}')
        axes[0, 0].legend()
    else:
        axes[0, 0].text(0.5, 0.5, 'No valid LLM scores available',
                       ha='center', va='center', transform=axes[0, 0].transAxes)
        axes[0, 0].set_title('Distribution of LLM Overall Quality Scores')
    
    # 2. Bid CPM Distribution
    valid_bids = df['bid_cpm_clean'].dropna()
    if len(valid_bids) > 0:
        axes[0, 1].hist(valid_bids, bins=20, alpha=0.7, color='lightgreen', edgecolor='black')
        axes[0, 1].set_title('Distribution of Bid CPM Prices')
        axes[0, 1].set_xlabel('Bid CPM ($)')
        axes[0, 1].set_ylabel('Frequency')
        axes[0, 1].axvline(valid_bids.mean(), color='red', linestyle='--',
                          label=f'Mean: ${valid_bids.mean():.2f}')
        axes[0, 1].legend()
    else:
        axes[0, 1].text(0.5, 0.5, 'No valid bid prices available',
                       ha='center', va='center', transform=axes[0, 1].transAxes)
        axes[0, 1].set_title('Distribution of Bid CPM Prices')
    
    # 3. Pay Price CPM Distribution
    valid_pay = df['pay_price_cpm_clean'].dropna()
    if len(valid_pay) > 0:
        axes[1, 0].hist(valid_pay, bins=20, alpha=0.7, color='lightcoral', edgecolor='black')
        axes[1, 0].set_title('Distribution of Pay Price CPM')
        axes[1, 0].set_xlabel('Pay Price CPM ($)')
        axes[1, 0].set_ylabel('Frequency')
        axes[1, 0].axvline(valid_pay.mean(), color='red', linestyle='--',
                          label=f'Mean: ${valid_pay.mean():.2f}')
        axes[1, 0].legend()
    else:
        axes[1, 0].text(0.5, 0.5, 'No valid pay prices available',
                       ha='center', va='center', transform=axes[1, 0].transAxes)
        axes[1, 0].set_title('Distribution of Pay Price CPM')
    
    # 4. Click Rate by Quality Score (if we have both)
    if len(valid_scores) > 0 and 'clicked_clean' in df.columns:
        # Create quality score bins
        df['quality_bin'] = pd.cut(df['llm_overall_score_clean'], 
                                  bins=5, labels=['Very Low', 'Low', 'Medium', 'High', 'Very High'])
        click_rates = df.groupby('quality_bin')['clicked_clean'].mean()
        
        click_rates.plot(kind='bar', ax=axes[1, 1], color='gold', alpha=0.7)
        axes[1, 1].set_title('Click Rate by Ad Quality Score')
        axes[1, 1].set_xlabel('Quality Score Range')
        axes[1, 1].set_ylabel('Click Rate')
        axes[1, 1].tick_params(axis='x', rotation=45)
    else:
        axes[1, 1].text(0.5, 0.5, 'Insufficient data for click rate analysis',
                       ha='center', va='center', transform=axes[1, 1].transAxes)
        axes[1, 1].set_title('Click Rate by Ad Quality Score')
    
    plt.tight_layout()
    plt.savefig('ad_quality_analysis_openai.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return fig

def create_augmented_visualizations(df):
    """Create augmented visualizations for the report"""
    # Explode the category scores into separate columns
    category_scores_df = df['llm_category_scores'].apply(pd.Series)
    
    # Create a bar chart of the average category scores
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
    """Generate summary statistics for the report"""
    stats = {}
    
    # Basic dataset info
    stats['total_ads'] = len(df)
    stats['unique_users'] = df['user'].nunique() if 'user' in df.columns else 'N/A'
    stats['unique_campaigns'] = df['campaign_id'].nunique() if 'campaign_id' in df.columns else 'N/A'
    
    # LLM Score statistics
    valid_scores = df['llm_overall_score_clean'].dropna()
    if len(valid_scores) > 0:
        stats['llm_scores'] = {
            'count': len(valid_scores),
            'mean': valid_scores.mean(),
            'median': valid_scores.median(),
            'std': valid_scores.std(),
            'min': valid_scores.min(),
            'max': valid_scores.max()
        }
    else:
        stats['llm_scores'] = {'count': 0, 'error': 'No valid LLM scores available'}
    
    # Bid price statistics
    valid_bids = df['bid_cpm_clean'].dropna()
    if len(valid_bids) > 0:
        stats['bid_prices'] = {
            'count': len(valid_bids),
            'mean': valid_bids.mean(),
            'median': valid_bids.median(),
            'std': valid_bids.std(),
            'min': valid_bids.min(),
            'max': valid_bids.max()
        }
    else:
        stats['bid_prices'] = {'count': 0, 'error': 'No valid bid prices available'}
    
    # Pay price statistics
    valid_pay = df['pay_price_cpm_clean'].dropna()
    if len(valid_pay) > 0:
        stats['pay_prices'] = {
            'count': len(valid_pay),
            'mean': valid_pay.mean(),
            'median': valid_pay.median(),
            'std': valid_pay.std(),
            'min': valid_pay.min(),
            'max': valid_pay.max()
        }
    else:
        stats['pay_prices'] = {'count': 0, 'error': 'No valid pay prices available'}
    
    # Click statistics
    if 'clicked_clean' in df.columns:
        stats['click_rate'] = df['clicked_clean'].mean()
        stats['total_clicks'] = df['clicked_clean'].sum()
    else:
        stats['click_rate'] = 'N/A'
        stats['total_clicks'] = 'N/A'
    
    return stats

def create_markdown_report(stats, df):
    """Create a comprehensive markdown report"""
    
    # Get sample of lowest scoring ads for human review
    low_scoring_ads = df.nsmallest(3, 'llm_overall_score_clean')[['creative', 'llm_overall_score_clean', 'llm_explanation', 'query', 'context']].copy()
    
    # Get sample of highest scoring ads for comparison
    high_scoring_ads = df.nlargest(2, 'llm_overall_score_clean')[['creative', 'llm_overall_score_clean', 'llm_explanation', 'query', 'context']].copy()
    
    report = f"""# Ad Quality Analysis Report - OpenAI GPT-4o

**Generated on:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Executive Summary

This report analyzes the quality and performance of advertisements based on OpenAI GPT-4o evaluation and bid pricing data. The analysis covers {stats['total_ads']} advertisements across {stats['unique_users']} unique users and {stats['unique_campaigns']} campaigns.

## Dataset Overview

- **Total Advertisements:** {stats['total_ads']}
- **Unique Users:** {stats['unique_users']}
- **Unique Campaigns:** {stats['unique_campaigns']}
- **Click Rate:** {stats['click_rate']:.2%} ({stats['total_clicks']} total clicks)

## Ad Quality Analysis

### LLM Overall Quality Scores

"""
    
    if 'error' not in stats['llm_scores']:
        llm_stats = stats['llm_scores']
        report += f"""
- **Valid Evaluations:** {llm_stats['count']} out of {stats['total_ads']} ads
- **Average Score:** {llm_stats['mean']:.1f}/100
- **Median Score:** {llm_stats['median']:.1f}/100
- **Score Range:** {llm_stats['min']:.1f} - {llm_stats['max']:.1f}
- **Standard Deviation:** {llm_stats['std']:.1f}

**Quality Distribution:**
- High Quality (80-100): {len(df[(df['llm_overall_score_clean'] >= 80) & (df['llm_overall_score_clean'] <= 100)])} ads
- Medium Quality (50-79): {len(df[(df['llm_overall_score_clean'] >= 50) & (df['llm_overall_score_clean'] < 80)])} ads
- Low Quality (0-49): {len(df[(df['llm_overall_score_clean'] >= 0) & (df['llm_overall_score_clean'] < 50)])} ads
"""
    else:
        report += f"""
**⚠️ Issue:** {stats['llm_scores']['error']}

This suggests there may be an issue with the LLM evaluation process. Please check the API processing results.
"""
    
    report += f"""

### LLM Category Scores

![Category Scores](category_scores_analysis_openai.png)

"""

    report += f"""

## Bid Pricing Analysis

### Bid CPM Prices

"""
    
    if 'error' not in stats['bid_prices']:
        bid_stats = stats['bid_prices']
        report += f"""
- **Valid Bids:** {bid_stats['count']} out of {stats['total_ads']} ads
- **Average Bid:** ${bid_stats['mean']:.2f} CPM
- **Median Bid:** ${bid_stats['median']:.2f} CPM
- **Bid Range:** ${bid_stats['min']:.2f} - ${bid_stats['max']:.2f} CPM
- **Standard Deviation:** ${bid_stats['std']:.2f}
"""
    else:
        report += f"""
**⚠️ Issue:** {stats['bid_prices']['error']}
"""
    
    report += f"""

### Pay Price CPM

"""
    
    if 'error' not in stats['pay_prices']:
        pay_stats = stats['pay_prices']
        report += f"""
- **Valid Pay Prices:** {pay_stats['count']} out of {stats['total_ads']} ads
- **Average Pay Price:** ${pay_stats['mean']:.2f} CPM
- **Median Pay Price:** ${pay_stats['median']:.2f} CPM
- **Pay Price Range:** ${pay_stats['min']:.2f} - ${pay_stats['max']:.2f} CPM
- **Standard Deviation:** ${pay_stats['std']:.2f}
"""
    else:
        report += f"""
**⚠️ Issue:** {stats['pay_prices']['error']}
"""
    
    # Add correlation analysis if we have both quality scores and prices
    if ('error' not in stats['llm_scores'] and 'error' not in stats['bid_prices'] and 
        len(df['llm_overall_score_clean'].dropna()) > 0 and len(df['bid_cpm_clean'].dropna()) > 0):
        
        correlation = df['llm_overall_score_clean'].corr(df['bid_cpm_clean'])
        report += f"""

## Quality vs. Pricing Correlation

- **Correlation between Quality Score and Bid Price:** {correlation:.3f}
- **Interpretation:** {'Positive correlation' if correlation > 0.1 else 'Negative correlation' if correlation < -0.1 else 'Weak correlation'} between ad quality and bid prices
"""
    
    report += f"""

## Visualizations

The following histograms show the distribution of:
1. **LLM Overall Quality Scores** - Distribution of ad quality ratings (0-100) using OpenAI GPT-4o
2. **Bid CPM Prices** - Distribution of bid prices per thousand impressions
3. **Pay Price CPM** - Distribution of actual prices paid per thousand impressions
4. **Click Rate by Quality** - Performance correlation between quality and clicks

![Ad Quality Analysis](ad_quality_analysis_openai.png)

## Click Analysis

"""
    clicked_ads = df[df['clicked_clean'] == True]
    if not clicked_ads.empty:
        report += f"""
### Clicked Ads Analysis

- **Total Clicks:** {stats['total_clicks']}
- **Average Quality Score of Clicked Ads:** {clicked_ads['llm_overall_score_clean'].mean():.1f}
- **Median Quality Score of Clicked Ads:** {clicked_ads['llm_overall_score_clean'].median():.1f}

"""
        for idx, (_, ad) in enumerate(clicked_ads.iterrows(), 1):
            report += f"""
#### Clicked Ad #{idx} - Score: {ad['llm_overall_score_clean']:.1f}/100

**Ad Creative:**
> {ad['creative'][:200]}{'...' if len(str(ad['creative'])) > 200 else ''}

**User Context:**
- **Query:** {ad['query'][:150]}{'...' if len(str(ad['query'])) > 150 else ''}
- **Context:** {ad['context'][:150]}{'...' if len(str(ad['context'])) > 150 else ''}

**LLM Explanation:**
> {ad['llm_explanation']}

---
"""
    else:
        report += "No ads were clicked in this dataset.\n"

    report += f"""

## Owner Feedback Analysis

"""
    if 'owner_like' in df.columns and 'owner_dislike' in df.columns:
        likes = df['owner_like'].sum()
        dislikes = df['owner_dislike'].sum()
        report += f"""
### Owner Feedback

- **Total Likes:** {likes}
- **Total Dislikes:** {dislikes}

"""
    else:
        report += "No owner feedback data available in this dataset.\n"


    report += f"""

## Key Findings

"""
    
    # Add key findings based on the data
    findings = []
    
    if 'error' not in stats['llm_scores']:
        avg_quality = stats['llm_scores']['mean']
        if avg_quality >= 70:
            findings.append(f"✅ **High Overall Quality:** Average ad quality score is {avg_quality:.1f}/100, indicating generally high-quality advertisements.")
        elif avg_quality >= 50:
            findings.append(f"⚠️ **Moderate Quality:** Average ad quality score is {avg_quality:.1f}/100, suggesting room for improvement.")
        else:
            findings.append(f"❌ **Low Quality:** Average ad quality score is {avg_quality:.1f}/100, indicating significant quality issues.")
    
    if 'error' not in stats['bid_prices'] and 'error' not in stats['pay_prices']:
        avg_bid = stats['bid_prices']['mean']
        avg_pay = stats['pay_prices']['mean']
        if avg_pay < avg_bid * 0.8:
            findings.append(f"💰 **Efficient Bidding:** Pay prices (${avg_pay:.2f}) are significantly lower than bid prices (${avg_bid:.2f}), indicating efficient auction participation.")
        else:
            findings.append(f"📊 **Competitive Bidding:** Pay prices (${avg_pay:.2f}) are close to bid prices (${avg_bid:.2f}), suggesting competitive auction environment.")
    
    if stats['click_rate'] != 'N/A':
        click_rate = stats['click_rate']
        if click_rate >= 0.05:
            findings.append(f"🎯 **Good Click Performance:** Click rate of {click_rate:.2%} is above industry average.")
        elif click_rate >= 0.02:
            findings.append(f"📈 **Moderate Click Performance:** Click rate of {click_rate:.2%} is within typical range.")
        else:
            findings.append(f"📉 **Low Click Performance:** Click rate of {click_rate:.2%} suggests need for optimization.")
    
    if not findings:
        findings.append("⚠️ **Limited Analysis:** Insufficient valid data for comprehensive findings.")
    
    report += "\n".join(findings)
    
    report += f"""

## Recommendations

1. **Quality Improvement:** Focus on improving ad creative quality based on LLM evaluation criteria
2. **Bid Optimization:** Analyze the relationship between bid prices and ad performance
3. **Performance Monitoring:** Track click rates and conversion metrics by quality score
4. **A/B Testing:** Test different creative approaches to improve overall quality scores

## Sample Advertisements for Human Review

### Lowest Scoring Advertisements

"""
    
    # Add lowest scoring ads
    for idx, (_, ad) in enumerate(low_scoring_ads.iterrows(), 1):
        report += f"""
#### Ad #{idx} - Score: {ad['llm_overall_score_clean']:.1f}/100

**Ad Creative:**
> {ad['creative'][:200]}{'...' if len(str(ad['creative'])) > 200 else ''}

**User Context:**
- **Query:** {ad['query'][:150]}{'...' if len(str(ad['query'])) > 150 else ''}
- **Context:** {ad['context'][:150]}{'...' if len(str(ad['context'])) > 150 else ''}

**LLM Explanation:**
> {ad['llm_explanation']}

---
"""
    
    report += f"""
### Highest Scoring Advertisements (for comparison)

"""
    
    # Add highest scoring ads
    for idx, (_, ad) in enumerate(high_scoring_ads.iterrows(), 1):
        report += f"""
#### Ad #{idx} - Score: {ad['llm_overall_score_clean']:.1f}/100

**Ad Creative:**
> {ad['creative'][:200]}{'...' if len(str(ad['creative'])) > 200 else ''}

**User Context:**
- **Query:** {ad['query'][:150]}{'...' if len(str(ad['query'])) > 150 else ''}
- **Context:** {ad['context'][:150]}{'...' if len(str(ad['context'])) > 150 else ''}

**LLM Explanation:**
> {ad['llm_explanation']}

---
"""
    # 7. **Private Activities** (0-10): Assessment of whether the user is engaging in romantic, therapeutic, fantastical, or sexual roleplay/writing

    report += f"""
## Metric Calculation Details

### LLM Quality Score Calculation

The LLM evaluation uses a 7-criteria rubric, each scored 0-10:

1. **Appropriateness** (0-10): How suitable the ad is for the conversation context
2. **Relevance** (0-10): How relevant the ad content is to the user's query/context  
3. **Timing & Disruption** (0-10): Whether the ad appears at a natural point without excessive disruption
4. **Clarity & Appeal** (0-10): How clear, understandable, and visually/verbally appealing the ad is
5. **Potential Effectiveness** (0-10): Likelihood the ad will achieve its goal (click, conversion, engagement)
6. **Experience** (0-10): How the user would rate their experience with this advertisement

**Important Note:** The overall score is determined by the LLM using a **heuristic approach**, not a mathematical formula. The LLM considers all criteria holistically and may apply additional logic such as:
- Penalizing ads served during inappropriate contexts (romantic/sexual roleplay)
- Weighting certain criteria more heavily based on context
- Applying business logic beyond simple averaging

**Example Heuristic Process:**
- Individual criteria scores: Appropriateness: 2, Relevance: 1, Timing: 3, Clarity: 4, Effectiveness: 1, Experience: 2, Private Activities: 0
- LLM considers: Low relevance + inappropriate context + poor effectiveness
- **Heuristic Result:** 22/100 (not a simple mathematical average)

### Bid Price Metrics

- **Bid CPM:** The maximum price the advertiser is willing to pay per thousand impressions
- **Pay Price CPM:** The actual price paid per thousand impressions (usually lower due to auction dynamics)
- **Efficiency Ratio:** Pay Price / Bid Price (lower ratios indicate more efficient bidding)

### Quality vs. Price Correlation

The correlation coefficient measures the linear relationship between ad quality scores and bid prices:
- **Range:** -1.0 to +1.0
- **Interpretation:**
  - > 0.3: Strong positive correlation (higher quality = higher bids)
  - 0.1 to 0.3: Moderate positive correlation
  - -0.1 to 0.1: Weak correlation
  - < -0.1: Negative correlation (higher quality = lower bids)

## Technical Notes

- **LLM Model:** OpenAI GPT-4o (Note: GPT-5 is not yet available)
- **Evaluation Criteria:** Appropriateness, Relevance, Timing & Disruption, Clarity & Appeal, Potential Effectiveness, Experience, Private Activities
- **Data Source:** Sampled from {stats['total_ads']} total advertisements
- **Analysis Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
- **Sampling Method:** Random sampling with 5% of total dataset
- **Evaluation Method:** Direct API calls with rate limiting (0.1s delay between requests)
- **Score Calculation:** Heuristic-based evaluation by LLM, not mathematical averaging

---
*This report was generated automatically from OpenAI API evaluation results.*
"""
    
    return report

def main():
    """Main analysis function"""
    print("Loading and analyzing ad quality data from OpenAI results...")
    
    # Load and clean data
    df = load_and_clean_data()
    
    # Create visualizations
    print("Creating visualizations...")
    create_histograms(df)
    create_augmented_visualizations(df)
    
    # Generate summary statistics
    print("Generating summary statistics...")
    stats = generate_summary_stats(df)
    
    # Create markdown report
    print("Creating markdown report...")
    report = create_markdown_report(stats, df)
    
    # Save report
    with open('ad_quality_report_openai.md', 'w', encoding='utf-8') as f:
        f.write(report)
    
    print("Analysis complete!")
    print(f"- Visualizations saved to: ad_quality_analysis_openai.png")
    print(f"- Augmented visualizations saved to: category_scores_analysis_openai.png")
    print(f"- Report saved to: ad_quality_report_openai.md")
    print(f"- Analyzed {stats['total_ads']} advertisements")
    
    return df, stats, report

if __name__ == "__main__":
    df, stats, report = main()