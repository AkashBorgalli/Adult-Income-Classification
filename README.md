# Adult Income Machine Learning Classification Project (End to End)


🚩 ***Problem Statement***: \
Building a robust machine learning model that would determine whether census income of an adult would be above 50K or not per year.




## 📜 Data Description:
The dataset provides 14 input variables that are a mixture of categorical, ordinal, and numerical data types. The complete list of variables is as follows:

- Age.
- Workclass.
- Final Weight.
- Education.
- Education Number of Years.
- Marital-status.
- Occupation.
- Relationship.
- Race.
- Sex.
- Capital-gain.
- Capital-loss.
- Hours-per-week.
- Country.
- Salary(Target Feature)
## Architecture Diagram

![](images/Architecture%20Diagram.jpg)
## 📝 Features

- Maintains snapshot of code.
- Logs Important Metrics and images on experiment dashboards as well as print in system logs .
- Shows feature importance.
- Created ML Pipeline (Datastats --> data pre-processing --> Model Training )
- Published ML Pipeline for Scheduling to run every week.
- Predict Real-time from Azure Container Instance .



## ⚙️ Data Preprocessing Techniques used
Steps : 1. Applied Log Transformation over Age and final weight.\
2. Filled missing values with modes of workclass, occupation and country columns.\
3. Reduced no.of unique categories of education, maritial-status.\
4. Converted Salary feature to binary.\
5. Enforced label-encoding on entire dataset.\
6. Applied Smoteenn to handle imbalanceness of database.\
7. Transformed data using StandardScalar.



## ✔️ Deployment
- Used LightGBM Model for Deployment.
- Deployed the model over Azure Container Instance.




## 💻 Tech Stack

**Cloud Platform:**  Azure Machine Learning 

**Language:** Python 3.6.2


## 💡 Screenshots:
- Data Stats over Azure\
!['Data'](screenshots/Experiment%20DataStats.png)




## Author <img src="https://raw.githubusercontent.com/TheDudeThatCode/TheDudeThatCode/master/Assets/Developer.gif" width=35 height=25>

- [@Akash Borgalli](https://www.linkedin.com/in/akashborgalli/)

