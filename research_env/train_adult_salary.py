# Import libraries
from azureml.core import Run, Model
import argparse
import pandas as pd
import numpy as np
import joblib
import os
from sklearn.model_selection import train_test_split
import lightgbm as lgb
from sklearn.metrics import roc_auc_score,f1_score,accuracy_score,precision_score,recall_score
from sklearn.metrics import roc_curve
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, plot_confusion_matrix,ConfusionMatrixDisplay
from sklearn.metrics import precision_score
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import matplotlib_inline
import seaborn as sns
from azureml.interpret import ExplanationClient
from interpret.ext.blackbox import TabularExplainer

labels = ['<=50K', '>50K']
features = ['age', 'workclass', 'fnlwgt', 'education', 'education-num','marital-status', 'occupation', 'relationship', 'race', 'sex',
       'capital-gain', 'capital-loss', 'hours-per-week', 'country']
# Get parameters
parser = argparse.ArgumentParser()
parser.add_argument("--training-data", type=str, dest='training_data', help='training data')
args = parser.parse_args()
training_data = args.training_data
print('training data', training_data)



# Get the experiment run context
run = Run.get_context()

# load the prepared data file in the training folder
print("Loading Data for Training...")
file_path = os.path.join(training_data,'data.csv')
df = pd.read_csv(file_path)

# Separate features and labels
print("Splitting data X and Y...")
X = df.drop(['salary'], axis =1)
y = df['salary']

# Split data into training set and test set
print("Splitting data into X_train and y_train...")
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.3, random_state = 101)

# scaling the data
print("Performing Standard Scalar...")
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Training a lightGBM model
print('Training a LightGBM Classifier model...')
clf = lgb.LGBMClassifier(boosting_type='goss',objective='binary',n_jobs=-1,n_estimators=200)
clf.fit(X_train, y_train)
y_pred=clf.predict(X_test)
print("The ROC-AUC Score obtained on is : " , roc_auc_score(y_test, y_pred))
print("The Macro F1-Score obtained on is : " , f1_score(y_test,  y_pred,average = 'macro'))
print("The F1 scores of each class on are : ",f1_score(y_test,  y_pred,average = None))

# calculate accuracy
acc = np.average(y_pred == y_test)
print('Accuracy:', acc)
run.log('Accuracy', np.float(acc))

# calculate AUC
y_scores = clf.predict_proba(X_test)
auc = roc_auc_score(y_test,y_scores[:,1])
print('AUC: ' + str(auc))
run.log('AUC', np.float(auc))


# calculate precision
precisionscore = precision_score(y_test, y_pred)
print('Precision: ' + str(precisionscore))
run.log('Precision', precisionscore)

# plot ROC curve
fpr, tpr, thresholds = roc_curve(y_test, y_scores[:,1])
fig = plt.figure(figsize=(6, 4))
# Plot the diagonal 50% line
plt.plot([0, 1], [0, 1], 'k--')
# Plot the FPR and TPR achieved by our model
plt.plot(fpr, tpr)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
run.log_image(name = "ROC", plot = fig)
plt.show()

#plot confusion matrix
fig = plt.figure(figsize=(6,6))
cm = confusion_matrix(y_test, y_pred )
plt.title('Heatmap of Confusion Matrix', fontsize = 12)
sns.heatmap(cm, annot = True ,  fmt = "d")
##run.log_confusion_matrix(name = "Heatmap of Confusion Matrix", value = cm)
run.log_image(name='Confusion Matrix LGBM', plot=fig)


# Save the trained model in the outputs folder
print("Saving model...")
os.makedirs('outputs', exist_ok=True)
model_file = os.path.join('outputs', 'adult_income_model.pkl')
joblib.dump(value=clf, filename=model_file)
# Register the model
print('Registering model...')
Model.register(workspace=run.experiment.workspace,
               model_path = model_file,
               model_name = 'adult_income_model',
               tags={'Training context':'Pipeline'},
               properties={'AUC': np.float(auc), 'Accuracy': np.float(acc), 'Precision': precision_score(y_test, y_pred)})

# Get explanation
explainer = TabularExplainer(clf, X_train, features=features, classes=labels)
explanation = explainer.explain_global(X_test)

# Get an Explanation Client and upload the explanation
explain_client = ExplanationClient.from_run(run)
explain_client.upload_model_explanation(explanation, comment='Tabular Explanation')

print('Completed the training')
run.complete()
