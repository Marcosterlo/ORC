import pandas as pd

csvs = []

csvs.append(pd.read_csv("double_data_grid_0-20000.csv"))
csvs.append(pd.read_csv("double_data_grid_20000-20500.csv"))
csvs.append(pd.read_csv("double_data_grid_20500-23000.csv"))
csvs.append(pd.read_csv("double_data_grid_23000-28000.csv"))
csvs.append(pd.read_csv("double_data_grid_28000-30000.csv"))
csvs.append(pd.read_csv("double_data_grid_30000-130000.csv"))
csvs.append(pd.read_csv("double_data_grid_130000-140000.csv"))
csvs.append(pd.read_csv("double_data_grid_140000-end.csv"))
csvs.append(pd.read_csv("double_data_random1.csv"))
csvs.append(pd.read_csv("double_data_random2.csv"))

result = csvs[0]
for i, csv in (enumerate(csvs)):
    if i!=0:
        result = pd.concat([result, csv])

result.to_csv("double_data_final.csv", index=False)