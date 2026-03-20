from config import RATING_METRICS, FIGURE_SIZE, COLOR_PALETTE
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import os

def plot_cluster_ratings(cluster_reviews, show_plots=True, save_path=None):
    """Enhanced plots for cluster ratings with config-driven metrics and figure size."""
    products = list(cluster_reviews.keys())
    metrics = RATING_METRICS

    for product in products:
        # Create a safe filename for the product
        safe_product_name = "".join([c for c in product if c.isalnum() or c in (' ', '_')]).rstrip()
        
        ratings_data = []
        cluster_labels = []
        for i, (cluster_key, review_data) in enumerate(cluster_reviews[product].items()):
            review_text = review_data['review']
            cluster_ratings = []
            for metric in metrics:
                try:
                    rating = int(review_text.split(f"{metric}: ")[1][0])
                    cluster_ratings.append(rating)
                except Exception:
                    cluster_ratings.append(3)
            ratings_data.append(cluster_ratings)
            cluster_labels.append(f"Cluster {i}\n({review_data['cluster_info']['customer_type']})")
        ratings_data = np.array(ratings_data)

        fig, axes = plt.subplots(2, 2, figsize=FIGURE_SIZE)
        fig.suptitle(f'Ratings Analysis for {product}', fontsize=18)
        axes = axes.flatten()

        # Plot 1: Radar Chart
        angles = np.linspace(0, 2*np.pi, len(metrics), endpoint=False)
        angles = np.concatenate((angles, [angles[0]]))  # complete the circle
        ax_radar = plt.subplot(2, 2, 1, projection='polar')
        for i, cluster_ratings in enumerate(ratings_data):
            values = np.concatenate((cluster_ratings, [cluster_ratings[0]]))
            ax_radar.plot(angles, values, label=cluster_labels[i])
            ax_radar.fill(angles, values, alpha=0.25)
        ax_radar.set_xticks(angles[:-1])
        ax_radar.set_xticklabels(metrics)
        ax_radar.set_title('Radar Chart of Ratings by Cluster')
        ax_radar.legend(bbox_to_anchor=(1.2, 1))

        # Plot 2: Bar Chart (Average Ratings by Cluster)
        ax = axes[1]
        cluster_means = np.mean(ratings_data, axis=1)
        ax.bar(cluster_labels, cluster_means, color='skyblue')
        ax.set_title('Average Ratings by Cluster')
        ax.set_ylim(0, 5)
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')

        # Plot 3: Heatmap
        ax = axes[2]
        sns.heatmap(ratings_data, annot=True, cmap='YlOrRd', 
                    xticklabels=metrics, yticklabels=cluster_labels, ax=ax)
        ax.set_title('Ratings Heatmap')

        # Plot 4: Box Plot
        df_ratings = pd.DataFrame(ratings_data, columns=metrics)
        ax_box = axes[3]
        df_ratings.boxplot(ax=ax_box)
        ax_box.set_title('Distribution of Ratings')
        ax_box.set_ylim(0, 5)

        plt.tight_layout(rect=(0, 0, 1, 0.97))
        
        # Handle saving/displaying
        if save_path:
            # Only create directory if save_path contains a directory
            dir_name = os.path.dirname(save_path)
            if dir_name:  # Only try to create directory if path contains a directory
                os.makedirs(dir_name, exist_ok=True)
            
            # Save with product-specific filename
            base, ext = os.path.splitext(save_path)
            product_save_path = f"{base}_{safe_product_name}{ext}"
            plt.savefig(product_save_path, bbox_inches='tight', dpi=300)
            print(f"Saved plot for '{product}' to {product_save_path}")
        
        if show_plots:
            plt.show()
        else:
            plt.close()

def print_reviews_to_console(cluster_reviews):
    """Print reviews to console if UI fails"""
    if not cluster_reviews:
        print("No reviews available.")
        return
    
    print("\n" + "=" * 60)
    print("GENERATED CUSTOMER REVIEWS")
    print("=" * 60)
    
    for product_name, reviews in cluster_reviews.items():
        print(f"\n🍎 {product_name}")
        print("-" * 50)
        
        for cluster_key, review_data in reviews.items():
            cluster_info = review_data['cluster_info']
            print(f"\n{cluster_key} ({cluster_info['customer_type']}):")
            print(f"Profile: {cluster_info['avg_age']:.0f}yr {cluster_info['dominant_gender']}, "
                  f"{cluster_info['dominant_occupation']}, {cluster_info['dominant_income']} income")
            print(f"Review: {review_data['review']}")
            print()