import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

MODEL_PATH = 'artifect/model_data.joblib'

model_data = joblib.load(MODEL_PATH)

model = model_data['model']
feature = model_data['features']
scaler = model_data['scaler']
cols_to_scale = model_data['cols_to_scale']

def preparation_of_df(age, income, loan_amount, 
            loan_tenure_months, avg_dpd_per_delinquency, delinquency_ratio,credit_utilization_ratio,
            num_open_accounts,resident_type,loan_purpose,loan_type):
    input_data={
        'number_of_open_accounts':num_open_accounts,
        'credit_utilization_ratio':credit_utilization_ratio,
        'loan_tenure_months':loan_tenure_months,
        'age':age, 
        'loan_to_income':(loan_amount / income) if income > 0 else 0, 
        'delinquency_ratio':delinquency_ratio,
        'avg_dpd_per_delinquency':avg_dpd_per_delinquency, 
        'loan_purpose_Education':1 if loan_purpose == 'Education' else 0,
        'loan_purpose_Home':1 if loan_purpose == 'Home' else 0, 
        'loan_purpose_Personal':1 if loan_purpose == 'Personal' else 0, 
        'loan_type_Unsecured':1 if loan_type == 'Unsecured' else 0,
        'residence_type_Owned':1 if resident_type == 'Owned' else 0, 
        'residence_type_Rented':1 if resident_type == 'Rented' else 0,
        #adding dummy features for cols_to_scale
        'number_of_closed_accounts':1,
        'total_loan_months':1, 
        'delinquent_months':1, 
        'total_dpd':1, 
        'enquiry_count':1,
        'sanction_amount':1, 
        'loan_amount':1,
       'processing_fee':1, 
       'gst':1, 
       'net_disbursement':1,
       'principal_outstanding':1, 
       'bank_balance_at_application':1, 
       'income':1,
       'number_of_dependants':1, 
       'years_at_current_address':1, 
       'zipcode':1
    }

    df = pd.DataFrame([input_data])
    df[cols_to_scale]=scaler.transform(df[cols_to_scale])

    df = df[feature]
    return df


def predict(age, income, loan_amount, 
            loan_tenure_months, avg_dpd_per_delinquency, delinquency_ratio,credit_utilization_ratio,
            num_open_accounts,resident_type,loan_purpose,loan_type):
    
    input_df = preparation_of_df(age, income, loan_amount, 
            loan_tenure_months, avg_dpd_per_delinquency, delinquency_ratio,credit_utilization_ratio,
            num_open_accounts,resident_type,loan_purpose,loan_type)
    
    probability, credit_score, rating = calculate_credit_score(input_df)

    return probability, credit_score, rating

def calculate_credit_score(input_df, base_score=300, scale_length=600):


    default_probability = model.predict_proba(input_df)[0][1]
    non_default_probability = model.predict_proba(input_df)[0][0]

    credit_score = base_score + non_default_probability.flatten() * scale_length

    def get_rating(score):
        if 300 <= score < 500:
            return 'Poor'
        elif 500 <= score < 650:
            return 'Average'
        elif 650 <= score < 750:
            return 'Good'
        elif 750 <= score < 900:
            return 'Excellent'
        else:
            return 'Score is undefined'

    rating = get_rating(credit_score[0])

    return default_probability.flatten()[0], int(credit_score[0]), rating

             

