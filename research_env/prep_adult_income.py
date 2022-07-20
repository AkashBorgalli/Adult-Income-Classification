# Import libraries
import os
import argparse
import pandas as pd
import seaborn as sns
from azureml.core import Run
from imblearn.combine import SMOTEENN
from collections import Counter


# Get parameters
parser = argparse.ArgumentParser()
parser.add_argument("--input-data", type=str, dest='raw_dataset_id', help='raw dataset')
parser.add_argument('--prepped-data', type=str, dest='prepped_data', default='prepped_data', help='Folder for results')
args = parser.parse_args()
save_folder = args.prepped_data

# Get the experiment run context
run = Run.get_context()

# load the data (passed as an input dataset)
print("Loading Data...")
df = run.input_datasets['Adult-Classification'].to_pandas_dataframe()

# Log raw row count
row_count = (len(diabetes))
run.log('raw_rows', row_count)
print("Logging row count",row_count)

# we will apply log transformation on age and fnlwgt but not oncapital loss, capital gain as log(0) will give n.d
print('Applying Log Transformation to age and fnlwgt columns')
df["age"] = np.log(df["age"])
df["fnlwgt"] = np.log(df["fnlwgt"])


# missing percentage values for:
print('missing percentage values for:')
print(f"workclass : {round(2093 / 32561 , 4) *100}%")
print(f"occupation : {round(1843 / 32561 , 4) *100}%")
print(f"native-country : {round(583 / 32561 , 4) *100}%")

## filling with modes
print('filling missing values by mode on workclass, occupation and country columns')
df["workclass"] = df['workclass'].str.replace('?', 'Private' )
df['occupation'] = df['occupation'].str.replace('?', 'Prof-specialty' )
df['country'] = df['country'].str.replace('?', 'United-States')



# reduced education unique categories
print('Reducing the unique counts from educationan and marital Status')
df["education"].replace(['Preschool', '1st-4th', '5th-6th', '7th-8th', '9th','10th', '11th', '12th'], 'school' ,
                         inplace = True , regex = True)
df["education"].replace(['Assoc-voc', 'Assoc-acdm', 'Prof-school', 'Some-college'], 'higher' , inplace = True , regex = True)

#reduced mariatal-status unique categories
df['marital-status'].replace(['Married-civ-spouse', 'Married-AF-spouse'], 'married' , inplace = True , regex = True)
df['marital-status'].replace(['Divorced', 'Separated','Widowed',
                                                   'Married-spouse-absent'], 'other' , inplace = True , regex = True)
# converting salary columns to binary
print('Converting salary feature to binary')
df["salary"] = df["salary"].replace({'<=50K' : 0 , ">50K" : 1 } , regex = True)

#checking the unique count now
cat_columns = df.select_dtypes(include='object')
cat_columns.columns
## checking the reduced unique count now
for feature in cat_columns.columns:
    print(f" {feature}  :  {len(df[feature].unique())}")
    

# plotting heatmap
print('plotting heat map')
heatmap_fig = plt.figure(figsize=(20,10), dpi = 150)
sns.heatmap(df.corr(), annot = True, cmap = 'viridis')
plt.show()
run.log_image(name='label distribution', plot=heatmap_fig)



## applied label encoding for entire dataset
print('applying label encoding for entire dataset')
from sklearn.preprocessing import  LabelEncoder
df = df.apply(LabelEncoder().fit_transform)


## splitting the data
print('Splitting the data in X and y')
X = df.drop(['salary'], axis =1)
y = df['salary']

# handling imbalanced dataset
SMOTEENN = SMOTEENN(n_jobs=-1)
print('Original dataset shape %s' % Counter(y))
X_res, y_res = SMOTEENN.fit_resample(X, y)
print('After undersample dataset shape %s' % Counter(y_res))
labeldf = pd.DataFrame(y_res,columns=['salary'])

# plotting balanced data graph
balanced_data_fig = plt.figure(figsize=(6,6))
sns.countplot(labeldf['salary'])
##plt.show()
run.log_image(name='Balanced salary distribution', plot=balanced_data_fig)


# Log processed rows
row_count = (len(diabetes))
run.log('processed_rows', row_count)

# Save the prepped data
print("Saving Data...")
os.makedirs(save_folder, exist_ok=True)
save_path = os.path.join(save_folder,'data.csv')
df.to_csv(save_path, index=False, header=True)

# End the run
run.complete()
