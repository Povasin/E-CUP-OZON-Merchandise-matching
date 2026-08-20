import cProfile
import pstats
import io
import pandas as pd
from src.features import extract_model_features

def main():
    print("Loading data...")
    # Load just a small sample to profile quickly
    matches = pd.read_parquet("assets/matches.parquet").head(10000)
    items = pd.read_parquet("assets/items_human.parquet")
    
    print("Starting profiling...")
    pr = cProfile.Profile()
    pr.enable()
    
    # Run the function
    features = extract_model_features(matches, items)
    
    pr.disable()
    print("Profiling finished. Generating stats...")
    
    s = io.StringIO()
    sortby = pstats.SortKey.CUMULATIVE
    ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
    # Print top 30 expensive functions
    ps.print_stats(30)
    print(s.getvalue())

if __name__ == "__main__":
    main()
