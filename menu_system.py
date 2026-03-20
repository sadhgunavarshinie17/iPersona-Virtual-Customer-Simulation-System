from advanced_analysis import (
    cluster_comparison_analysis, 
    product_performance_analysis, 
    market_insights_analysis, 
    custom_analysis
)
from report_generator import export_to_excel

def show_analysis_menu(cluster_reviews, cluster_characteristics, analysis_result):
    """Display and handle the analysis menu options"""
    while True:
        print("\n" + "=" * 50)
        print("ADDITIONAL ANALYSIS OPTIONS:")
        print("1. View Cluster Comparison Analysis")
        print("2. View Product Performance Analysis") 
        print("3. View Market Insights")
        print("4. Generate Custom Analysis")
        print("5. Export Data to Excel")
        print("6. Exit")
        
        choice = input("\nSelect an option (1-6): ").strip()
        
        if choice == '1':
            cluster_comparison_analysis(cluster_reviews, cluster_characteristics)
        elif choice == '2':
            product_performance_analysis(cluster_reviews)
        elif choice == '3':
            market_insights_analysis(analysis_result)
        elif choice == '4':
            custom_analysis(cluster_reviews, cluster_characteristics)
        elif choice == '5':
            export_to_excel(cluster_reviews, analysis_result)
        elif choice == '6':
            break
        else:
            print("Invalid choice. Please select 1-6.")