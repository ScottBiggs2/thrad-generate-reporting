import os
import time
import json
import pandas as pd
import requests
from dotenv import load_dotenv
from openai import OpenAI

# Load environment variables from .env file
load_dotenv()

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise RuntimeError("Please set environment variable OPENAI_API_KEY")

# Initialize OpenAI client
client = OpenAI(api_key=OPENAI_API_KEY)

def load_parquet(parquet_file_path: str = 'data/2025-10-16.parquet') -> pd.DataFrame:
    df = pd.read_parquet(parquet_file_path, engine='pyarrow')
    print(f"Parquet file successfully read from {parquet_file_path}")
    return df.drop_duplicates(subset=['user'], keep='last')

def build_prompt(row: pd.Series, rubric: str) -> str:
    creative = row.get('creative', '')
    context = row.get('context', '')
    query = row.get('query', '')
    user_msg = row.get('user_message_before', '')
    bot_msg = row.get('bot_message_before', '')
    prompt = f"""
You are an expert ad-evaluation assistant with a deep knowledge of effective ad services in niche consumption contexts as well as broad campaigns.
Use the rubric below to evaluate how "good" the advertisement is in the following chat context.
Be sure to detect users engaging in roleplay, creative writing, therapy, casual chitchat, romantic or sexual, and other inappropriate subjects and reduce the scores of ads served to these users. 
Be sure to detect users working on content invisible to you. For example, writing emails, analyzing documents, writing code, or working with other private materials, and reduce the scores of ads served to these users.

Rubric:
{rubric}

Conversation / context:
User message before: {user_msg}
Bot message before: {bot_msg}
User query: {query}
Context summary: {context}

Ad creative shown: {creative}

Again, be sure to detect users engaging in roleplay, creative writing, therapy, casual chitchat, romantic or sexual, and other inappropriate subjects and reduce the scores of ads served to these users. 
Again, be sure to detect users working on content invisible to you. For example, writing emails, analyzing documents, writing code, or working with other private materials, and reduce the scores of ads served to these users.
Please output a JSON object with keys:
- category_scores: an object mapping each rubric category name to a number between 0 and 10, rounding down when in doubt.
- overall_score: a number between 0 and 100, rounding down when in doubt and returning 0 overall if you detect romantic, sexual, or fantastical roleplay, writing, or content.
- explanation: a short text explanation (2-4 sentences).

Only output the JSON object (no extra text).
"""
    return prompt.strip()

def sample_rows(df: pd.DataFrame, k_percent: float, seed: int = 42) -> pd.DataFrame:
    n = max(1, int(len(df) * (k_percent / 100.0)))
    return df.sample(n=n, random_state=seed).reset_index(drop=True)

def evaluate_single_ad(row: pd.Series, rubric: str, idx: int) -> dict:
    """Evaluate a single ad using OpenAI API"""
    prompt = build_prompt(row, rubric)
    
    try:
        response = client.chat.completions.create(
            model="gpt-4o", 
            messages=[
                {"role": "user", "content": prompt}
            ],
            temperature=0.0,
            max_tokens=1000
        )

        # response = client.chat.completions.create(
        #     model="gpt-5",
        #     messages=[
        #         {"role": "user", "content": prompt}
        #     ],
        #     max_completion_tokens=1000
        # )
        
        content = response.choices[0].message.content.strip()
        print(f"Raw response for row {idx}: {content[:200]}...")
        
        # Check if content is empty
        if not content:
            return {
                "custom_id": f"row_{idx}",
                "status": "failed",
                "response": None,
                "error": "Empty response from API"
            }
        
        # Parse JSON response
        try:
            # Clean the content - remove any markdown formatting or extra text
            content_clean = content.strip()
            if content_clean.startswith('```json'):
                content_clean = content_clean[7:]
            if content_clean.endswith('```'):
                content_clean = content_clean[:-3]
            content_clean = content_clean.strip()
            
            result = json.loads(content_clean)
            # Validate that we have the expected structure
            if not isinstance(result, dict) or 'overall_score' not in result:
                return {
                    "custom_id": f"row_{idx}",
                    "status": "failed",
                    "response": None,
                    "error": f"Invalid JSON structure: missing required fields. Got: {list(result.keys()) if isinstance(result, dict) else type(result)}"
                }
            
            return {
                "custom_id": f"row_{idx}",
                "status": "completed",
                "response": {
                    "status_code": 200,
                    "body": {
                        "choices": [{
                            "message": {
                                "content": content
                            }
                        }]
                    }
                },
                "error": None
            }
        except json.JSONDecodeError as e:
            print(f"JSON parse error for row {idx}: {str(e)}")
            print(f"Content that failed to parse: {content}")
            return {
                "custom_id": f"row_{idx}",
                "status": "failed",
                "response": None,
                "error": f"JSON parse error: {str(e)}"
            }
            
    except Exception as e:
        print(f"API error for row {idx}: {str(e)}")
        return {
            "custom_id": f"row_{idx}",
            "status": "failed",
            "response": None,
            "error": f"API error: {str(e)}"
        }

def evaluate_ads_batch(sampled_df: pd.DataFrame, rubric: str, result_jsonl_path: str):
    """Evaluate all ads and save results to JSONL file"""
    print(f"Evaluating {len(sampled_df)} ads using OpenAI API...")
    
    results = []
    for idx, row in sampled_df.iterrows():
        print(f"Evaluating ad {idx + 1}/{len(sampled_df)}...")
        result = evaluate_single_ad(row, rubric, idx)
        results.append(result)
        
        # Add a small delay to respect rate limits
        time.sleep(0.1)
    
    # Save results to JSONL file
    with open(result_jsonl_path, 'w', encoding='utf-8') as f:
        for result in results:
            f.write(json.dumps(result, ensure_ascii=False) + "\n")
    
    print(f"Saved evaluation results to {result_jsonl_path}")
    return results

def merge_results(sampled_df: pd.DataFrame, result_jsonl_path: str) -> pd.DataFrame:
    # read the JSON lines file, map custom_id -> result
    results = {}
    with open(result_jsonl_path, 'r', encoding='utf-8') as f:
        for line in f:
            j = json.loads(line)
            cid = j.get("custom_id")
            results[cid] = j
    
    # build new columns
    category_scores_col = []
    overall_scores_col = []
    explanations_col = []
    
    for idx, _ in sampled_df.iterrows():
        cid = f"row_{idx}"
        j = results.get(cid)
        
        if j and j.get("status") == "completed" and j.get("response"):
            # Parse the response structure
            try:
                response_body = j["response"]["body"]
                if "choices" in response_body and len(response_body["choices"]) > 0:
                    content = response_body["choices"][0]["message"]["content"]
                    resp_obj = json.loads(content)
                    category_scores_col.append(resp_obj.get("category_scores"))
                    overall_scores_col.append(resp_obj.get("overall_score"))
                    explanations_col.append(resp_obj.get("explanation"))
                else:
                    category_scores_col.append(None)
                    overall_scores_col.append(None)
                    explanations_col.append(f"No choices in response for {cid}")
            except Exception as e:
                category_scores_col.append(None)
                overall_scores_col.append(None)
                explanations_col.append(f"JSON parse error for {cid}: {str(e)}")
        else:
            error_msg = j.get("error", "Unknown error") if j else "Missing result"
            category_scores_col.append(None)
            overall_scores_col.append(None)
            explanations_col.append(f"Error for {cid}: {error_msg}")
    
    sampled_df = sampled_df.copy()
    sampled_df["llm_category_scores"] = category_scores_col
    sampled_df["llm_overall_score"] = overall_scores_col
    sampled_df["llm_explanation"] = explanations_col
    return sampled_df

def main():
    df = load_parquet()
    print(f"Loaded dataframe with {len(df)} unique users.")
    
    rubric = (
        "1. Appropriateness: Was the advertisement appropriate given the user's conversation and context?\n"
        "2. Relevance: How relevant was the advertisement creative to the user's query and context?\n"
        "3. Timing & Disruption: Did the advertisement serve at a natural point and not disrupt the conversation excessively?\n"
        "4. Clarity & Appeal: Was the advertisement clear, understandable, visually/verbally appealing and persuasive?\n"
        "5. Potential Effectiveness: Based on context and user, how likely is the advertisement to achieve its goal (click, conversion, engagement)?\n"
        "6. Experience: Based on the context and user, if you were in their shoes, how would you rate your experience with this advertisement?\n"
    )
        # "7. Private Activities: Based on the context and user history, are they likely to be participating in romantic, therapeutic, fantastical, or sexual roleplay, writing, or reflection?"


    k = 1  # percentage to sample
    sampled_df = sample_rows(df, k)
    
    # Evaluate ads using OpenAI API
    result_jsonl_path = "openai_batch_results.jsonl"
    evaluate_ads_batch(sampled_df, rubric, result_jsonl_path)
    
    # Merge results
    sampled_with_evals = merge_results(sampled_df, result_jsonl_path)
    sampled_with_evals.to_parquet("sampled_evaluations_openai.parquet", index=False)
    print("Done. Saved sampled evaluations to sampled_evaluations_openai.parquet")

if __name__ == "__main__":
    main()
