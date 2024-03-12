import pandas as pd

fifa_file_path = "C:\\Users\\Hari Keshav Rajesh\\Desktop\\Computer Projects and resources\\Datasets\\FIFA\\world_cup_comparisons.csv"

fifa_df = pd.read_csv(fifa_file_path)



print(f"Most goals scored in season 2018 by {fifa_df.loc[fifa_df.goals_z.idxmax(),'player']}")

print(f"Most expected goals in season 2018 by {fifa_df.loc[fifa_df.xg_z.idxmax(),'player']}")

print(f"Most passes made in season 2018 by {fifa_df.loc[fifa_df.passes_z.idxmax(),'player']}")


