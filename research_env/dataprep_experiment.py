import azureml.core
from azureml.core import Workspace, Dataset, Run
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib_inline

# Load the workspace from the saved config file
ws = Workspace.from_config()
print(ws)

# Get the experiment run context
run = Run.get_context()

# load the diabetes dataset
dataset = Dataset.get_by_name(ws, name='salary_classification')
df = dataset.to_pandas_dataframe()


# Count the rows and log the result
row_count = (len(df))
run.log('observations', row_count)
print('Analyzing {} rows of data'.format(row_count))

# Count and log the label counts
salary_counts = df['salary'].value_counts()
print(salary_counts)
for k, v in salary_counts.items():
    run.log('Label:' + str(k), v)

## imbalanced dataset figure
fig = plt.figure(figsize=(6,6))
sns.countplot(x = 'salary',data = df)
plt.show()
run.log_image(name='label distribution', plot=fig)

#percentage of imbalanceness
print(f"<= 50k : {round(24720 /32561 * 100 , 2)}")
print(f"> 50k : {round(7841 /32561 * 100 , 2)}")

run.log('% of <= 50k',round(24720 /32561 * 100 , 2))
run.log('% of > 50k', round(7841 /32561 * 100 , 2))


# Log summary statistics for numeric columns
med_columns = ['age','fnlwgt', 'education-num','capital-gain', 'capital-loss', 'hours-per-week']
summary_stats = df[med_columns].describe().to_dict()
for col in summary_stats:
    keys = list(summary_stats[col].keys())
    values = list(summary_stats[col].values())
    for index in range(len(keys)):
        run.log_row(col, stat=keys[index], value = values[index])
        
# Log summary statistics for cat columns
cat_columns = ['workclass','education','marital-status', 'occupation', 'relationship','race', 'sex','country','salary']
cat_summary_stats = df[cat_columns].describe().to_dict()
for col in cat_summary_stats:
    keys = list(cat_summary_stats[col].keys())
    values = list(cat_summary_stats[col].values())
    for index in range(len(keys)):
        run.log_row(col, stat=keys[index], value = values[index])
        
# Complete the run
run.complete()
