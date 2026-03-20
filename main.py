import pandas as pd
from pipeline import (
    run_clustering_pipeline, 
    run_product_loading_pipeline, 
    run_review_generation_pipeline, 
    run_analysis_pipeline,
)
from visualization import plot_cluster_ratings, print_reviews_to_console
from report_generator import save_analysis_report, display_analysis_results
from menu_system import show_analysis_menu

def main_menu():
    df = None
    cluster_characteristics = None
    product_data = None
    cluster_reviews = None
    analysis_result = None

    while True:
        print("\n" + "=" * 60)
        print("MAIN MENU")
        print("=" * 60)
        print("1. Load Data & Run Clustering")
        print("2. Load Product Data")
        print("3. Generate AI Reviews")
        print("4. Run GPT-4 Analysis")
        print("5. Display Results")
        print("6. Advanced Analysis & Export")
        print("7. Exit")
        choice = input("\nSelect an option (1-7): ").strip()

        if choice == "1":
            df, cluster_characteristics = run_clustering_pipeline()
        elif choice == "2":
            product_data = run_product_loading_pipeline()
        elif choice == "3":
            if not cluster_characteristics or product_data is None or product_data.empty:
                print("❗ Please run clustering and load product data first.")
            else:
                cluster_reviews = run_review_generation_pipeline(cluster_characteristics, product_data)
        elif choice == "4":
            if not cluster_reviews or not cluster_characteristics:
                print("❗ Please generate reviews first.")
            else:
                analysis_result = run_analysis_pipeline(cluster_reviews, cluster_characteristics)
        elif choice == "5":
            if not cluster_reviews or not analysis_result:
                print("❗ Please generate reviews and run analysis first.")
            else:
                # Save plots instead of showing them
                plot_cluster_ratings(cluster_reviews, show_plots=False, save_path="cluster_ratings.png")
                print_reviews_to_console(cluster_reviews)
                display_analysis_results(analysis_result)
                save_analysis_report(analysis_result)
                print("✓ Results saved successfully (plots saved as PNG).")
        elif choice == "6":
            if not cluster_reviews or not cluster_characteristics or not analysis_result:
                print("❗ Please complete all previous steps first.")
            else:
                show_analysis_menu(cluster_reviews, cluster_characteristics, analysis_result)
        elif choice == "7":
            print("Exiting. Goodbye!")
            break
        else:
            print("Invalid choice. Please select 1-7.")

if __name__ == "__main__":
    print("Welcome to the Customer Clustering and Review Generation System!")
    main_menu()