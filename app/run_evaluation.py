import utils as ut
import pandas as pd
from datetime import datetime

test_dataset_filename = "test_dataset3.csv"
s = ut.evaluate_test_set(f"./testsets/{test_dataset_filename}")
dataset = s.dataset.to_pandas()
scores = s.scores.to_pandas()
df = pd.concat([dataset, scores], axis=1)
df.to_csv(f"./testsets/test_results_{datetime.now()}.csv")
print("saved to csv")