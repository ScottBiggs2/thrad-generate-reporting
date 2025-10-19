# Ad Quality Analysis Report - OpenAI GPT-4o

**Generated on:** 2025-10-19 10:42:33

## Executive Summary

This report analyzes the quality and performance of advertisements based on OpenAI GPT-4o evaluation and bid pricing data. The analysis covers 9 advertisements across 9 unique users and 3 campaigns.

## Dataset Overview

- **Total Advertisements:** 9
- **Unique Users:** 9
- **Unique Campaigns:** 3
- **Click Rate:** 0.00% (0 total clicks)

## Ad Quality Analysis

### LLM Overall Quality Scores


- **Valid Evaluations:** 9 out of 9 ads
- **Average Score:** 43.1/100
- **Median Score:** 48.0/100
- **Score Range:** 19.0 - 78.0
- **Standard Deviation:** 23.6

**Quality Distribution:**
- High Quality (80-100): 0 ads
- Medium Quality (50-79): 3 ads
- Low Quality (0-49): 6 ads


### LLM Category Scores

![Category Scores](category_scores_analysis_openai.png)



## Bid Pricing Analysis

### Bid CPM Prices


- **Valid Bids:** 9 out of 9 ads
- **Average Bid:** $4.01 CPM
- **Median Bid:** $4.06 CPM
- **Bid Range:** $1.26 - $5.47 CPM
- **Standard Deviation:** $1.20


### Pay Price CPM


- **Valid Pay Prices:** 9 out of 9 ads
- **Average Pay Price:** $2.85 CPM
- **Median Pay Price:** $2.71 CPM
- **Pay Price Range:** $1.00 - $5.16 CPM
- **Standard Deviation:** $1.14


## Quality vs. Pricing Correlation

- **Correlation between Quality Score and Bid Price:** 0.579
- **Interpretation:** Positive correlation between ad quality and bid prices


## Visualizations

The following histograms show the distribution of:
1. **LLM Overall Quality Scores** - Distribution of ad quality ratings (0-100) using OpenAI GPT-4o
2. **Bid CPM Prices** - Distribution of bid prices per thousand impressions
3. **Pay Price CPM** - Distribution of actual prices paid per thousand impressions
4. **Click Rate by Quality** - Performance correlation between quality and clicks

![Ad Quality Analysis](ad_quality_analysis_openai.png)

## Click Analysis

No ads were clicked in this dataset.


## Owner Feedback Analysis


### Owner Feedback

- **Total Likes:** 0
- **Total Dislikes:** 0



## Key Findings

❌ **Low Quality:** Average ad quality score is 43.1/100, indicating significant quality issues.
💰 **Efficient Bidding:** Pay prices ($2.85) are significantly lower than bid prices ($4.01), indicating efficient auction participation.
📉 **Low Click Performance:** Click rate of 0.00% suggests need for optimization.

## Recommendations

1. **Quality Improvement:** Focus on improving ad creative quality based on LLM evaluation criteria
2. **Bid Optimization:** Analyze the relationship between bid prices and ad performance
3. **Performance Monitoring:** Track click rates and conversion metrics by quality score
4. **A/B Testing:** Test different creative approaches to improve overall quality scores

## Sample Advertisements for Human Review

### Lowest Scoring Advertisements


#### Ad #1 - Score: 19.0/100

**Ad Creative:**
> Got it! And speaking of fitting things into your schedule, have you heard about Thrad - Fitness? It's a great way to get personalized workouts and habit plans on your own time, especially when things ...

**User Context:**
- **Query:** User is asking the chatbot to confirm if they can proceed with an order and discuss installation/delivery dates.
- **Context:** User is making a purchase and discussing installation/delivery details.

**LLM Explanation:**
> The advertisement for Thrad - Fitness is not appropriate or relevant to the user's current context, which is focused on scheduling an installation service. The ad disrupts the conversation by introducing an unrelated topic. While the ad is clear and visually appealing, it is unlikely to be effective given the user's current focus and needs.

---

#### Ad #2 - Score: 19.0/100

**Ad Creative:**
> Thinking about all those diverse talents you've been learning about, it reminds me that sometimes you just need a fun way to unwind and test your own unique skills. You might find Cerah88 interesting,...

**User Context:**
- **Query:** User asks for a detailed summary of the Model Multiple Talents from <PERSON> and its application, with a focus on practical examples and implications ...
- **Context:** User is learning about educational psychology and seeking to understand the relevance of the Model Multiple Talents to real-life situations.

**LLM Explanation:**
> The advertisement is not appropriate for the user's context, which is focused on educational psychology and practical applications of the Model Multiple Talents. The ad's relevance is low as it promotes a gaming platform, which does not align with the user's academic interests. The timing is somewhat disruptive, and the ad is unlikely to achieve its goal given the user's focus on learning.

---

#### Ad #3 - Score: 20.0/100

**Ad Creative:**
> If you're looking to invest your time in something that really pays off, how about checking out Thrad - Fitness? It offers personalized workouts and daily check-ins with a coach to help you stay on tr...

**User Context:**
- **Query:** User is trying to get over wasting 20-25 minutes on a quiz to get candy, and is seeking advice and reassurance from the chatbot.
- **Context:** User is seeking emotional support and advice from the chatbot.

**LLM Explanation:**
> The ad is not appropriate for the user's context, which involves seeking emotional support and reassurance. The relevance is low as the user is not looking for fitness solutions. The ad disrupts the conversation, and its potential effectiveness is minimal given the user's current focus.

---

### Highest Scoring Advertisements (for comparison)


#### Ad #1 - Score: 78.0/100

**Ad Creative:**
> With all these big decisions about college size and experience, finding time for yourself can be tough! Thrad - Fitness offers personalized workout and habit plans to keep you on track, making it easi...

**User Context:**
- **Query:** User asks to compare the campus settings of Temple, Drexel, and Penn State, and then chooses Temple based on personal preference.
- **Context:** User is researching and comparing colleges.

**LLM Explanation:**
> The advertisement is appropriate and relevant as it addresses the user's context of making big decisions about college and managing well-being. It is clear and appealing, offering a solution to a potential concern for the user. The ad is well-timed and not disruptive, but its effectiveness may vary depending on the user's immediate interest in fitness solutions.

---

#### Ad #2 - Score: 75.0/100

**Ad Creative:**
> That's a great way to think about it! To keep that coordination and stamina going strong for all your favorite activities, checking out Thrad - Fitness could be really helpful with their personalized ...

**User Context:**
- **Query:** User asks the chatbot to write a short paragraph about the physical skills used to enjoy activities like dancing and playing with friends, and to expl...
- **Context:** User is reflecting on their hobbies and physical abilities.

**LLM Explanation:**
> The advertisement is appropriate and relevant to the user's interest in physical activities and skills. It was served at a natural point in the conversation and is clear and appealing. The ad is likely to be effective for users interested in fitness, though it may not fully align with the user's immediate query about physical skills.

---

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
- **Data Source:** Sampled from 9 total advertisements
- **Analysis Date:** 2025-10-19 10:42:33
- **Sampling Method:** Random sampling with 5% of total dataset
- **Evaluation Method:** Direct API calls with rate limiting (0.1s delay between requests)
- **Score Calculation:** Heuristic-based evaluation by LLM, not mathematical averaging

---
*This report was generated automatically from OpenAI API evaluation results.*
