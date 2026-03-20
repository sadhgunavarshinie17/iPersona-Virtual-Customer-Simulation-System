from review_generator import ChatOpenAI
from config import (
    OPENAI_API_KEY, OPENAI_MODEL, OPENAI_TEMPERATURE,
    RATING_METRICS, DEFAULT_RATING
)

def get_llm():
    """Get configured LLM instance"""
    return ChatOpenAI(
        model=OPENAI_MODEL,
        temperature=OPENAI_TEMPERATURE,
        api_key=OPENAI_API_KEY # type: ignore
    )

def cluster_comparison_analysis(cluster_reviews, cluster_characteristics):
    """Generate detailed cluster comparison analysis"""
    print("\n🔍 Generating Cluster Comparison Analysis...")

    llm = get_llm()

    # Efficient cluster summary
    cluster_summary = "\n".join(
        f"\nCluster {cid}: {info.get('customer_type', 'N/A')}\n"
        f"- Demographics: {info.get('avg_age', 0):.0f}yr {info.get('dominant_gender', 'N/A')}\n"
        f"- Occupation: {info.get('dominant_occupation', 'N/A')}\n"
        f"- Income: {info.get('dominant_income', 'N/A')}\n"
        f"- Brand Preference: {info.get('dominant_brand', 'N/A')}\n"
        f"- Price Range: {info.get('price_range', 'N/A')}\n"
        for cid, info in cluster_characteristics.items()
    )

    prompt = (
        "Compare and contrast the following customer clusters based on their characteristics and review patterns:\n\n"
        f"{cluster_summary}\n\n"
        "Provide:\n"
        "1. Key differences between clusters\n"
        "2. Similarities and overlaps\n"
        "3. Which clusters are most valuable to target\n"
        "4. Cluster-specific marketing strategies\n"
        "5. Product recommendations for each cluster\n"
    )

    try:
        messages = [("system", "You are a customer segmentation expert."), ("human", prompt)]
        response = llm.invoke(messages)
        print("\n" + "=" * 60)
        print("CLUSTER COMPARISON ANALYSIS")
        print("=" * 60)
        print(response.content)
    except Exception as e:
        print(f"Error in cluster comparison: {e}")

def product_performance_analysis(cluster_reviews):
    """Generate detailed product performance analysis"""
    print("\n📱 Generating Product Performance Analysis...")

    llm = get_llm()

    # Extract product ratings efficiently
    product_data = {}
    for product, reviews in cluster_reviews.items():
        product_data[product] = []
        for review_data in reviews.values():
            review_text = review_data.get('review', '')
            ratings = {}
            for metric in RATING_METRICS:
                try:
                    rating = int(review_text.split(f"{metric}: ")[1][0])
                    ratings[metric] = rating
                except Exception:
                    ratings[metric] = DEFAULT_RATING
            product_data[product].append(ratings)

    # Calculate average ratings for each product
    product_metrics = {}
    for product, ratings_list in product_data.items():
        avg_ratings = {}
        for metric in RATING_METRICS:
            avg_ratings[metric] = sum(r[metric] for r in ratings_list) / len(ratings_list)
        product_metrics[product] = avg_ratings

    prompt = (
        "Analyze the performance of Apple products based on customer reviews:\n\n"
        f"{str(product_metrics)}\n\n"
        "Provide:\n"
        "1. Top performing products overall\n"
        "2. Products that excel in specific categories\n"
        "3. Products with consistent vs. polarized ratings\n"
        "4. Value proposition analysis\n"
        "5. Recommendations for product improvements\n"
    )

    try:
        messages = [("system", "You are a product analyst specializing in consumer electronics."), ("human", prompt)]
        response = llm.invoke(messages)
        
        # Return both the structured data and the analysis text
        return {
            "metrics": product_metrics,
            "analysis": response.content
        }
    except Exception as e:
        print(f"Error in product analysis: {e}")
        return None

def market_insights_analysis(analysis_result):
    """Display market insights from the main analysis"""
    print("\n" + "=" * 60)
    print("MARKET INSIGHTS SUMMARY")
    print("=" * 60)
    # Extract market insights section from the main analysis
    sections = analysis_result.split("4. MARKET INSIGHTS:")
    if len(sections) > 1:
        insights_section = sections[1].split("5. STRATEGIC RECOMMENDATIONS:")[0]
        print(insights_section.strip())
    else:
        print("Market insights section not found in the main analysis.")

def custom_analysis(cluster_reviews, cluster_characteristics):
    """Allow user to request custom analysis"""
    print("\n🎯 Custom Analysis Generator")
    print("Ask any specific question about the customer clusters or product reviews.")

    user_question = input("\nEnter your analysis question: ").strip()
    if not user_question:
        print("No question provided.")
        return

    llm = get_llm()

    # Prepare context (truncate for API limits)
    context = f"Customer Clusters: {str(cluster_characteristics)[:1000]}\n\nReviews: {str(cluster_reviews)[:1000]}"

    prompt = (
        "Based on the customer clustering and review data provided, please answer this question:\n\n"
        f"{user_question}\n\n"
        f"Context:\n{context}...\n\n"
        "Provide a detailed, data-driven response. Dont should be  your thought and cluster information, but focus on actionable insights and recommendations.\n"
    )

    try:
        print("\n🤖 Generating custom analysis...")
        messages = [("system", "You are an expert data analyst and market researcher."), ("human", prompt)]
        response = llm.invoke(messages)
        print("\n" + "=" * 60)
        print("CUSTOM ANALYSIS RESULTS")
        print("=" * 60)
        print(response.content)
    except Exception as e:
        print(f"Error in custom analysis: {e}")