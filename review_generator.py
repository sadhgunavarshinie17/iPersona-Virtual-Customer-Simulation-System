import concurrent.futures
from langchain_openai import ChatOpenAI
import pandas as pd
import re
from functools import lru_cache
from tabulate import tabulate
from tqdm import tqdm
from config import (
    OPENAI_API_KEY, OPENAI_MODEL, OPENAI_TEMPERATURE,
    RATING_METRICS, DEFAULT_RATING, DEFAULT_NUM_REVIEWS_PER_CLUSTER, MAX_THREADS
)

@lru_cache(maxsize=None)
def get_llm(model=OPENAI_MODEL, temperature=OPENAI_TEMPERATURE):
    return ChatOpenAI(
        model=model,
        temperature=temperature,
        api_key=OPENAI_API_KEY # type: ignore
    )

def extract_ratings_vectorized(review_series, metrics, default=3):
    """Extract ratings for each metric from a pandas Series of review text using regex (vectorized)."""
    ratings = {}
    for metric in metrics:
        # Regex: look for 'Metric: <digit>'
        ratings[metric] = (
            review_series.str.extract(fr"{metric}:\s*(\d)", expand=False)
            .astype(float)
            .fillna(default)
            .astype(int)
        )
    return pd.DataFrame(ratings)

def generate_cluster_based_reviews(cluster_characteristics, product_data, num_reviews_per_cluster=DEFAULT_NUM_REVIEWS_PER_CLUSTER):
    """
    Generate AI reviews based on the actual cluster characteristics from customer data.
    Token efficiency: Truncate product table and product details.
    Progress bar: Show tqdm progress for review generation.
    """
    print("Step 1: Initializing LLM with OpenAI")
    llm = get_llm()

    cluster_reviews = {}
    # Truncate comparison table for performance (avoid token overflow)
    comparison_table = product_data.head(10).to_string()

    print("Step 2: Generating reviews based on actual cluster characteristics")

    def process_product_cluster(product, cluster_id, cluster_info):
        product_name = product['Model']
        # Truncate product details for token efficiency
        product_details = product.to_string()[:500]
        system_message = (
            f"You are a customer from a specific demographic cluster reviewing the Apple product: {product_name}\n\n"
            f"Your customer profile (Cluster {cluster_id}):\n"
            f"- Customer Type: {cluster_info['customer_type']}\n"
            f"- Average Age: {cluster_info['avg_age']:.0f} years\n"
            f"- Dominant Gender: {cluster_info['dominant_gender']}\n"
            f"- Occupation: {cluster_info['dominant_occupation']}\n"
            f"- Income Level: {cluster_info['dominant_income']}\n"
            f"- Current Phone Brand Preference: {cluster_info['dominant_brand']}\n"
            f"- Price Range Preference: {cluster_info['price_range']}\n"
            f"- Key Features You Value: {cluster_info['key_features']}\n"
            f"- Satisfaction Focus: {cluster_info['satisfaction_level']}\n\n"
            f"All Available Products for Comparison (showing up to 10):\n{comparison_table}\n\n"
            f"Current Product Details:\n{product_details}\n\n"
            f"Give a customer review that reflects your cluster characteristics. Include your opinion on value, features, and how well it meets your specific needs. Make it sound authentic and personal. Consider how this product compares to other available options.\n\n"
            f"Structure your response as:\n"
            + "\n".join([f"{metric}: [Score /5]" for metric in RATING_METRICS]) +
            "\nReview: [Your detailed review including comparisons with other models]"
        )

        messages = [("system", system_message)]
        try:
            ai_msg = llm.invoke(messages)
            return product_name, cluster_id, ai_msg.content
        except Exception as e:
            print(f"Error generating review for {product_name} from Cluster {cluster_id}: {e}")
            return product_name, cluster_id, f"Error generating review: {str(e)}"

    tasks = []
    for _, product in product_data.iterrows():
        product_name = product['Model']
        cluster_reviews[product_name] = {}
        for cluster_id, cluster_info in cluster_characteristics.items():
            tasks.append((product, cluster_id, cluster_info))

    # Progress bar for review generation
    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_THREADS) as executor:
        futures = [
            executor.submit(process_product_cluster, product, cluster_id, cluster_info)
            for product, cluster_id, cluster_info in tasks
        ]
        for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc="Generating reviews"):
            try:
                product_name, cluster_id, review_content = future.result()
                cluster_reviews[product_name][f"Cluster {cluster_id}"] = {
                    'cluster_info': cluster_characteristics[cluster_id],
                    'review': review_content
                }
            except Exception as e:
                print(f"Error processing task: {e}")

    print(f"Step 3: Generated reviews for {len(cluster_reviews)} products across {len(cluster_characteristics)} clusters")
    return cluster_reviews

def analyze_reviews_with_gpt4(cluster_reviews, cluster_characteristics, custom_prompt=None):
    """
    Analyze the generated reviews using GPT-4 to provide insights and patterns.
    Now supports custom prompts and provides more detailed output.
    """
    print("\n🔍 Analyzing Reviews with GPT-4...")
    llm = get_llm()

    # Prepare detailed review summary
    review_summary = ""
    ratings_data = []

    for product_name, reviews in cluster_reviews.items():
        review_summary += f"\n{'='*50}\nPRODUCT: {product_name}\n{'='*50}\n"
        for cluster_key, review_data in reviews.items():
            cluster_info = review_data['cluster_info']
            review_text = review_data['review']
            review_summary += f"\n{cluster_key} ({cluster_info['customer_type']}):\n"
            review_summary += f"Demographics: {cluster_info['avg_age']:.0f}yr {cluster_info['dominant_gender']}, "
            review_summary += f"{cluster_info['dominant_occupation']}, {cluster_info['dominant_income']} income\n"
            review_summary += f"Review: {review_text}\n"
            ratings_data.append({
                'Product': product_name,
                'Cluster': cluster_key,
                'review': review_text
            })

    # Prepare cluster characteristics summary
    cluster_summary = "\n".join(
        [f"Cluster {k}: {v['customer_type']} - {v['dominant_occupation']}, "
        f"{v['dominant_income']} income, Age {v['avg_age']:.0f}" 
        for k, v in cluster_characteristics.items()]
    )

    # Use custom prompt if provided, otherwise use default
    if custom_prompt:
        analysis_prompt = f"""
        You are an expert market research analyst for Apple products. 
        Provide a detailed analysis based on the following request:
        
        USER QUESTION:
        {custom_prompt}
        
        CONTEXT DATA:
        CLUSTER CHARACTERISTICS:
        {cluster_summary}
        
        CUSTOMER REVIEWS:
        {review_summary[:10000]}  # Increased character limit for custom analysis
        
        Please provide:
        1. Comprehensive answer to the user's question
        2. Supporting evidence from the reviews
        3. Specific recommendations when applicable
        4. Any relevant insights about customer segments
        """
    else:
        analysis_prompt = f"""
        You are an expert market research analyst for Apple products. 
        Provide a detailed analysis of the following customer reviews:
        
        CLUSTER CHARACTERISTICS:
        {cluster_summary}
        
        CUSTOMER REVIEWS:
        {review_summary[:10000]}  # Increased character limit for more complete analysis
        
        Please provide a structured report with these sections:
        1. EXECUTIVE SUMMARY (3-5 key findings)
        2. CLUSTER ANALYSIS (detailed breakdown of each segment)
        3. PRODUCT PERFORMANCE (top products and why they're successful)
        4. OPPORTUNITIES (areas for improvement)
        5. RECOMMENDATIONS (actionable marketing and product suggestions)
        6. QUANTITATIVE INSIGHTS (when available)
        
        Include specific examples from reviews to support your analysis.
        """

    try:
        print("Generating detailed analysis...")
        messages = [
            ("system", "You are a senior Apple product analyst with 15 years experience. "
            "Provide thorough, professional analysis with concrete examples."),
            ("human", analysis_prompt)
        ]

        # Get the AI response
        ai_response = llm.invoke(messages)
        analysis_result = ai_response.content

        # Add quantitative analysis if available
        if ratings_data:
            ratings_df = pd.DataFrame(ratings_data)
            ratings_metrics_df = extract_ratings_vectorized(ratings_df['review'], RATING_METRICS, DEFAULT_RATING)
            ratings_df = pd.concat([ratings_df[['Product', 'Cluster']], ratings_metrics_df], axis=1)
            
            # Enhanced quantitative summary
            quantitative_summary = generate_quantitative_analysis(ratings_df, llm)
            analysis_result += f"\n\n{'='*50}\nQUANTITATIVE ANALYSIS\n{'='*50}\n"
            analysis_result += quantitative_summary
            
            # Add metrics table
            avg_ratings = ratings_df.groupby(['Product', 'Cluster']).mean().reset_index()
            analysis_result += f"\n\nAVERAGE RATINGS BY PRODUCT AND CLUSTER:\n"
            analysis_result += "\n" + avg_ratings.to_string(index=False)

        # Print the full analysis to terminal
        print("\n" + "="*80)
        print("GPT-4 ANALYSIS RESULT")
        print("="*80)
        print(analysis_result)
        print("\nAnalysis complete.")

        return analysis_result

    except Exception as e:
        error_msg = f"Error in GPT-4 analysis: {e}"
        print(error_msg)
        return error_msg
def generate_quantitative_analysis(ratings_df, llm):
    """Generate concise quantitative analysis of ratings data (token-efficient)"""
    try:
        summary_stats = ""
        if not ratings_df.empty:
            available_metrics = [m for m in RATING_METRICS if m in ratings_df.columns]
            if available_metrics:
                # Only show averages, round to 2 decimals, and limit output
                summary_stats = "AVERAGE RATINGS:\n"
                avg_row = ratings_df[available_metrics].mean().round(2)
                summary_stats += ", ".join(f"{m}: {avg_row[m]}" for m in available_metrics) + "\n"
                if 'Product' in ratings_df.columns:
                    product_avg = ratings_df.groupby('Product')[available_metrics].mean().round(2)
                    summary_stats += "\nPRODUCT AVERAGES (top 3 shown):\n"
                    summary_stats += product_avg.head(3).to_string() + "\n"
                if 'Cluster' in ratings_df.columns:
                    cluster_avg = ratings_df.groupby('Cluster')[available_metrics].mean().round(2)
                    summary_stats += "\nCLUSTER AVERAGES:\n"
                    summary_stats += cluster_avg.to_string() + "\n"

        quant_prompt = (
            "Given these average ratings, briefly summarize key differences and business implications:\n"
            f"{summary_stats}"
        )

        messages = [
            ("system", "You are a data analyst. Be concise."),
            ("human", quant_prompt)
        ]

        ai_response = llm.invoke(messages)
        # Only return the first 500 characters of the AI output to minimize output tokens
        return f"{summary_stats}\nINSIGHTS:\n{ai_response.content[:500]}"
    except Exception as e:
        return f"Error in quantitative analysis: {str(e)}"