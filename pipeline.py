import pandas as pd
from data_preprocessing import load_data, preprocess_data
from clustering import fit_predict
from cluster_analysis import analyze_clusters
from review_generator import generate_cluster_based_reviews, analyze_reviews_with_gpt4
from config import CUSTOMER_CSV, PRODUCT_XLSX

# Initialize global variables (these will be updated by the pipeline functions)
global_df = None
global_cluster_characteristics = None
global_product_data = None
global_cluster_reviews = None
global_analysis_result = None

def run_clustering_pipeline():
    """Run the customer clustering pipeline"""
    global global_df, global_cluster_characteristics
    
    print("\n📊 Step 1: Loading Customer Data...")
    df = load_data(CUSTOMER_CSV)

    print("\n🔍 Step 2: Running Customer Clustering Analysis...")
    clusters = fit_predict(df, preprocess_data)
    df['Cluster'] = clusters

    print("\n📈 Step 3: Analyzing Cluster Characteristics...")
    cluster_characteristics = analyze_clusters(df, clusters)

    # Update global variables
    global_df = df
    global_cluster_characteristics = cluster_characteristics
    
    return df, cluster_characteristics

def run_product_loading_pipeline():
    """Load Apple product data"""
    global global_product_data
    
    print("\n🍎 Step 4: Loading Apple Product Data...")
    try:
        product_data = pd.read_excel(PRODUCT_XLSX)
        if 'Model' in product_data.columns:
            print(f"Loaded {len(product_data)} Apple products")
        global_product_data = product_data
        return product_data
    except Exception as e:
        print(f"❌ Error loading product data: {e}")
        raise

def run_review_generation_pipeline(cluster_characteristics, product_data):
    """Generate cluster-based reviews"""
    global global_cluster_reviews
    
    print("\n🤖 Step 5: Generating AI Reviews Based on Cluster Characteristics...")
    cluster_reviews = generate_cluster_based_reviews(
        cluster_characteristics,
        product_data,
        num_reviews_per_cluster=1
    )
    global_cluster_reviews = cluster_reviews
    return cluster_reviews

# Add at the top with other global variables
global_df = None
global_cluster_characteristics = None
global_product_data = None
global_cluster_reviews = None
global_analysis_result = None

# Modify the analysis pipeline function
def run_analysis_pipeline(cluster_reviews, cluster_characteristics):
    """Run GPT-4 analysis pipeline"""
    global global_analysis_result
    
    print("\n🔍 Step 6: Analyzing Reviews with GPT-4...")
    analysis_result = analyze_reviews_with_gpt4(cluster_reviews, cluster_characteristics)

    # Structure the results for export
    analysis_result = {
        'cluster_characteristics': cluster_characteristics,
        'cluster_reviews': cluster_reviews,
        'analysis': analysis_result  # Use the result from analyze_reviews_with_gpt4
    }
    return analysis_result
def run_analysis_only():
    """Run only the clustering analysis without review generation"""
    global global_df, global_cluster_characteristics
    
    print("=" * 50)
    print("CUSTOMER CLUSTERING ANALYSIS ONLY")
    print("=" * 50)
    try:
        df = load_data(CUSTOMER_CSV)
        clusters = fit_predict(df, preprocess_data)
        df['Cluster'] = clusters
        cluster_characteristics = analyze_clusters(df, clusters)
        
        # Update global variables
        global_df = df
        global_cluster_characteristics = cluster_characteristics
        
        print("\n✅ Clustering analysis completed successfully!")
        return df, cluster_characteristics
    except Exception as e:
        print(f"❌ Error in clustering analysis: {e}")
        return None, None

def run_review_generation_only(cluster_characteristics):
    """Run only the review generation with existing cluster characteristics"""
    global global_cluster_reviews
    
    print("=" * 50)
    print("REVIEW GENERATION ONLY")
    print("=" * 50)
    try:
        product_data = pd.read_excel(PRODUCT_XLSX)
        cluster_reviews = generate_cluster_based_reviews(
            cluster_characteristics,
            product_data,
            num_reviews_per_cluster=1
        )
        global_cluster_reviews = cluster_reviews
        print("\n✅ Review generation completed successfully!")
        return cluster_reviews
    except Exception as e:
        print(f"❌ Error in review generation: {e}")
        return None

def get_global_data_status():
    """Return the status of all global data variables"""
    return {
        'customer_data': global_df is not None,
        'cluster_characteristics': global_cluster_characteristics is not None,
        'product_data': global_product_data is not None,
        'cluster_reviews': global_cluster_reviews is not None,
        'analysis_result': global_analysis_result is not None
    }