"""Configuration settings for the customer clustering system"""

# API Configuration
OPENAI_API_KEY = "OPENAI_API_KEY"
OPENAI_MODEL = "gpt-4.1-mini-2025-04-14"
OPENAI_TEMPERATURE = 0.3

# Data & File Paths
CUSTOMER_CSV = "./customer.csv"
PRODUCT_XLSX = "Apple Product details.xlsx"
EXCEL_EXPORT_FILENAME = "customer_analysis_export.xlsx"

# Review & Rating
RATING_METRICS = ["Price", "Performance", "Features", "Battery", "Screen"]
DEFAULT_RATING = 3
DEFAULT_NUM_REVIEWS_PER_CLUSTER = 1

# Visualization
FIGURE_SIZE = (20, 12)
COLOR_PALETTE = "tab10"

# Parallelism
MAX_THREADS = 8