import pandas as pd
import numpy as np

def analyze_clusters(df, clusters):
    """Analyze and describe each cluster and store characteristics"""
    df_with_clusters = df.copy()
    df_with_clusters['Cluster'] = clusters
    cluster_characteristics = {}

    # Precompute overall stats for normalization
    overall_age_mean = df_with_clusters['age'].mean()
    overall_age_std = df_with_clusters['age'].std()
    overall_phone_year_mean = df_with_clusters['current_phone_model_year'].mean()
    overall_phone_year_std = df_with_clusters['current_phone_model_year'].std()

    print("Cluster Analysis:")
    print("=" * 50)

    total_customers = len(df_with_clusters)

    # Group once for efficiency
    grouped = df_with_clusters.groupby('Cluster')
    for cluster_id, cluster_data in grouped:
        cluster_size = len(cluster_data)
        print(f"\nCluster {cluster_id} ({cluster_size} customers, {cluster_size/total_customers*100:.1f}%):")
        print("-" * 30)

        # Vectorized stats
        avg_age = cluster_data['age'].mean()
        age_std = cluster_data['age'].std()
        age_zscore = (avg_age - overall_age_mean) / overall_age_std if overall_age_std else 0

        avg_phone_year = cluster_data['current_phone_model_year'].mean()
        phone_year_std = cluster_data['current_phone_model_year'].std()
        phone_year_zscore = (avg_phone_year - overall_phone_year_mean) / overall_phone_year_std if overall_phone_year_std else 0

        # Categorical distributions (use value_counts with normalize=True for efficiency)
        gender_dist = cluster_data['gender'].value_counts(normalize=True)
        occupation_dist = cluster_data['occupation'].value_counts(normalize=True)
        income_dist = cluster_data['income_level'].value_counts(normalize=True)
        brand_dist = cluster_data['current_phone_brand'].value_counts(normalize=True)

        # Most common values (use idxmax only if not empty)
        dominant_gender = gender_dist.idxmax() if not gender_dist.empty else "Mixed"
        dominant_occupation = occupation_dist.idxmax() if not occupation_dist.empty else "Various"
        dominant_income = income_dist.idxmax() if not income_dist.empty else "Mixed"
        dominant_brand = brand_dist.idxmax() if not brand_dist.empty else "Various"

        # Customer type logic (vectorized scoring)
        income_score = sum(('high' in str(x).lower()) - ('low' in str(x).lower()) for x in income_dist.index)
        phone_age_score = np.sign(phone_year_zscore) if abs(phone_year_zscore) > 0.5 else 0

        if income_score > 0 and phone_age_score >= 0:
            profile = {
                'customer_type': "Premium tech-savvy users",
                'price_range': "High-end to Premium",
                'key_features': "Latest technology, premium build quality, advanced features",
                'satisfaction_level': "High expectations for quality and innovation"
            }
        elif income_score < 0 and phone_age_score <= 0:
            profile = {
                'customer_type': "Value-conscious users",
                'price_range': "Budget to Mid-range",
                'key_features': "Reliability, essential features, good value",
                'satisfaction_level': "Price-sensitive, practical needs"
            }
        else:
            profile = {
                'customer_type': "Balanced mainstream users",
                'price_range': "Mid-range",
                'key_features': "Good balance of features and price",
                'satisfaction_level': "Moderate expectations, balanced needs"
            }

        # Store characteristics (avoid unnecessary dict conversions)
        cluster_characteristics[cluster_id] = {
            **profile,
            'avg_age': avg_age,
            'age_std': age_std,
            'age_zscore': age_zscore,
            'dominant_gender': dominant_gender,
            'gender_distribution': gender_dist.to_dict(),
            'dominant_occupation': dominant_occupation,
            'occupation_distribution': occupation_dist.head(3).to_dict(),
            'dominant_income': dominant_income,
            'income_distribution': income_dist.to_dict(),
            'dominant_brand': dominant_brand,
            'brand_distribution': brand_dist.head(3).to_dict(),
            'avg_phone_year': avg_phone_year,
            'phone_year_std': phone_year_std,
            'phone_year_zscore': phone_year_zscore
        }

        # Print concise analysis (use join for efficiency)
        print(f"Age: {avg_age:.1f} ± {age_std:.1f} (z-score: {age_zscore:.2f})")
        print("Gender:", ", ".join(f"{k}: {v*100:.1f}%" for k, v in gender_dist.items()))
        print("Occupation:", ", ".join(f"{k}: {v*100:.1f}%" for k, v in occupation_dist.head(3).items()))
        print("Income:", ", ".join(f"{k}: {v*100:.1f}%" for k, v in income_dist.items()))
        print("Brand:", ", ".join(f"{k}: {v*100:.1f}%" for k, v in brand_dist.head(3).items()))
        print(f"Phone model year: {avg_phone_year:.1f} ± {phone_year_std:.1f} (z-score: {phone_year_zscore:.2f})")
        print(f"Customer type: {profile['customer_type']}")
        print(f"Price preference: {profile['price_range']}")
        print(f"Key features: {profile['key_features']}")

    return cluster_characteristics