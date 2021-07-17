from flask import Flask, request, render_template
from flask_cors import cross_origin
import sklearn
import pickle
import pandas as pd
import numpy as np


app = Flask(__name__)

train_df = pd.read_csv('train.csv')
train_df.rename(columns = {'Date of Joining' : 'Date_of_Joining','Company Type':'Company_Type','WFH Setup Available':'WFH_Setup_Available','Resource Allocation':'Resource_Allocation','Mental Fatigue Score':'Mental_Fatigue_Score'}, inplace = True)
train_df['Joining_day']=pd.to_datetime(train_df.Date_of_Joining,format="%Y-%m-%d").dt.day
train_df['Joining_month']=pd.to_datetime(train_df.Date_of_Joining,format="%Y-%m-%d").dt.month
train_df.drop(['Date_of_Joining'],axis=1,inplace=True)
Gender = train_df[['Gender']]
Gender = pd.get_dummies(Gender)
Company_Type = train_df[['Company_Type']]
Company_Type = pd.get_dummies(Company_Type)
WFH_Setup_Available = train_df[['WFH_Setup_Available']]
WFH_Setup_Available = pd.get_dummies(WFH_Setup_Available)
train_df.drop(['Gender','Company_Type','WFH_Setup_Available'],axis=1,inplace=True)
train_df=pd.concat([train_df,Gender,Company_Type,WFH_Setup_Available],axis=1)
train_df['Resource_Allocation'].fillna(np.mean(train_df['Resource_Allocation']),inplace=True)
train_df['Mental_Fatigue_Score'].fillna(np.mean(train_df['Mental_Fatigue_Score']),inplace=True)
train_df['Burn Rate'].fillna(np.mean(train_df['Burn Rate']),inplace=True)
X = train_df[['Designation', 'Resource_Allocation',
       'Mental_Fatigue_Score', 'Joining_day', 'Joining_month',
       'Gender_Female', 'Gender_Male', 'Company_Type_Product',
       'Company_Type_Service', 'WFH_Setup_Available_No',
       'WFH_Setup_Available_Yes']]
y = train_df['Burn Rate']
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.35, random_state=42)
from sklearn.ensemble import RandomForestRegressor
rfc = RandomForestRegressor(n_estimators = 100,random_state = 0)
rfc.fit(X_train, y_train)
import pickle
file = open('employee_burnout_rf.pkl', 'wb')
pickle.dump(rfc, file)
#model = pickle.load(open("employee_burnout_rf.pkl", "rb"))
#employee_b = pickle.load(model)

@app.route("/")
def home():
    return render_template("home.html")

@app.route("/employee_info")
def employee_info():
    return render_template("employee_info.html")

@app.route("/about")
def about_page():
    return render_template("about.html")

@app.route("/sign")
def signs():
    return render_template("signs.html")

@app.route("/prevention")
def prevent():
    return render_template("prevent.html")

@app.route("/developer")
def developers():
    return render_template("developer.html")

@app.route("/burnout", methods = ["GET", "POST"])
def burnout():
    if request.method == "POST":

        # Date_of_Joining
        Date_of_Joining = request.form["Date of Joining"]
        Joining_day=int(pd.to_datetime(Date_of_Joining,format="%Y-%m-%d").day)
        Joining_month=int(pd.to_datetime(Date_of_Joining,format="%Y-%m-%d").month)

        Gender=request.form['Gender']
        if(Gender=='Male'):
            Gender_Male = 1
            Gender_Female = 0

        else:
            Gender_Male = 0
            Gender_Female = 0

        Company_Type = request.form["Company Type"]
        if(Company_Type == 'Product'):
            Company_Type_Product = 1
            Company_Type_Service = 0

        else:
            Company_Type_Product = 0
            Company_Type_Service = 1

        WFH_Setup_Available = request.form["WFH Setup Available"]
        if(WFH_Setup_Available == 'Yes'):
            WFH_Setup_Available_No = 0
            WFH_Setup_Available_Yes = 1
    
        else:
            WFH_Setup_Available_No = 1
            WFH_Setup_Available_Yes = 0

        Designation = request.form["Designation"]

        Resource_Allocation = request.form["Resource Allocation"]

        Mental_Fatigue_Score = request.form["Mental Fatigue Score"]
        
        prediction=rfc.predict([[ Designation,  Resource_Allocation,
        Mental_Fatigue_Score, Joining_day, Joining_month,
        Gender_Female, Gender_Male, Company_Type_Product,
        Company_Type_Service, WFH_Setup_Available_No,
        WFH_Setup_Available_Yes
        ]])
        #print(prediction)

        output=round(prediction[0],2)
        #print(output)
        return render_template('employee_info.html',prediction_text="Your Burn Rate is {} that is {} %".format(output,output*100))


    return render_template("home.html")

if __name__ == '__main__':
    app.run(debug=True)