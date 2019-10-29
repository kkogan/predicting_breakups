import pandas as pd
import numpy as np
from pandas.api.types import CategoricalDtype
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, RandomizedSearchCV 
from sklearn.metrics import confusion_matrix, precision_score, recall_score, precision_recall_curve, f1_score, fbeta_score, make_scorer
from sklearn.ensemble import RandomForestClassifier

import boto3
from dotenv import load_dotenv
import joblib
import os
import util
load_dotenv()
DATA_DIR='../../resources/data'
PRODUCTS,EVENT_MATRIX_IDX,NO_OF_EVENTS,INPUT_PRODUCTS=None,None,None,None

def load_data():
    #https://data.stanford.edu/hcmst
    print("load data")
    df_merged = pd.read_stata(DATA_DIR+'/HCMST_ver_3.04.dta')
    df4 = pd.read_stata(DATA_DIR+'/wave_4_supplement_v1_2.dta')
    df5 = pd.read_stata(DATA_DIR+'/HCMST_wave_5_supplement_ver_1.dta')
    df_merged = pd.merge(df_merged, df4, on='caseid_new', how='left')
    df_merged = pd.merge(df_merged, df5, on='caseid_new', how='left')
    return df_merged

def clean_data(df_dirty):
    print("clean data")

    df = df_dirty.copy()
    #only included partnered at start of study
    df = df[df['qflag'] == 'partnered']
    #turn categoricals into strings
    df[df.loc[:, df.dtypes == 'category'].columns] = df.loc[:, df.dtypes == 'category'].astype(str)
    df.replace({'nan':np.nan}, inplace=True)

    # build up a dataframe for visualization and ML
    df_ml = pd.DataFrame()
    df_ml['caseid_new'] = df['caseid_new']
    #dropnas
    # df =  df[~df['w2345_combo_breakup'].isna()]
    
    # TODO follow up with instructor about whether weights should be utilized
       
    # other 
    
    #
    # age
    #
    df_ml['age_difference'] = df['age_difference']
    df_ml['ppage'] = df['ppage'].astype('int')
    df_ml['partner_age'] = df['q9']
    df_ml['partner_years_younger'] = df_ml['ppage']-df_ml['partner_age']
    
    #
    # sexual orientation
    #
    df_ml['ppgender']= (df['ppgender'] == 'female').astype('int')
    df_ml['lgb'] = (df['glbstatus'] == 'glb').astype('int')
    df_ml['same_sex_couple'] = df['same_sex_couple']#.replace({'same-sex couple':1, 'different sex couple':0})

    #
    # money & lifestyle
    #
    df_ml['hhinc'] = df['hhinc']
    df_ml['pphh_internet'] = (df['ppnet'] == 'yes').astype('int')
    df_ml['ppmsacat_metro'] = (df['ppmsacat'] == 'metro').astype('int')

    #
    # education
    #
    df_ml['respondent_yrsed'] = df['respondent_yrsed']
    df_ml['partner_yrsed'] = df['partner_yrsed']

    #
    # family
    #
    
    df_ml['parental_approval'] = df['parental_approval'].fillna('unknown')#.replace({'approve':1, "don't approve or don't know":0})
    df_ml['respondent_mom_yrsed'] = df['respondent_mom_yrsed']
    df_ml['respondent_mom_yrsed'].fillna(np.mean(df['respondent_mom_yrsed']) , inplace=True)
    df_ml['partner_mom_yrsed'] = df['partner_mom_yrsed']
    df_ml['partner_mom_yrsed'].fillna(np.mean(df['partner_mom_yrsed']) , inplace=True)
    
    df_ml['children_in_hh'] = df['children_in_hh']
    df_ml['pphouseholdsize'] = df['pphouseholdsize']

    # 
    # ethnicity & nationality
    #
    df_ml['us_raised'] = (df['q15a1_compressed'] == 'United States').astype('int')
    # made things a lot worse
    #df_ml['respondent_race'] = df['respondent_race']
    #df_ml['partner_race'] = df['partner_race']
    # what about this?
    df_ml['mixed_race'] = (df['partner_race'] != df['respondent_race']).astype(int)
        
    #
    # religion
    #
#     df_ml['respondent_religion'] = df['papreligion']
#     df_ml['partner_religion'] = df['q7b'].replace({'refused':np.nan})
#     df_ml['respondent_religious'] = (df_ml['respondent_religion'] != 'None').astype('int')
#     df_ml['partner_religious'] = (df_ml['partner_religion'] != 'None').astype('int')
#     df_ml['same_religious_beliefs'] = (df_ml['partner_religion'] == df_ml['respondent_religion']).astype('int')

    #
    # politics
    #
#     df_ml['partner_politics'] = df['q12'].replace({'refused':np.nan})
#     df_ml['respondent_politics'] = df['pppartyid3']
#     df_ml['same_or_ambivalent_politics'] = ((df_ml['partner_politics'] == df_ml['respondent_politics']) | (df_ml['partner_politics'] == 'no preference')).astype('int')
    
    #
    # type of relationship
    #
    df_ml['1_married'] = df['married'].replace({'married':1,'not married':0}) # main survey 1
    df_ml['1_unmarried_sex_partner'] = (df['s2'] == 'yes, i have a sexual partner (boyfriend or girlfriend)').astype('int')
    df_ml['1_unmarried_rom_partner'] = (df['s2'] == 'i have a romantic partner who is not yet a sexual partner').astype('int')
    df_ml['coresident'] = df['coresident'].replace({'Yes':1, 'No':0})

    #
    # qualitative
    #
    df_ml['relationship_quality'] = df['relationship_quality']


    #
    # how they met
    # 
    df_ml['met_through_friends'] = (df['met_through_friends'] == 'meet through friends').astype('int')
    df_ml['met_through_as_neighbors'] = df['met_through_as_neighbors'].replace({"did not meet through or as neighbors":0,"met through or as neighbors":1})
    df_ml['met_through_family'] = df['met_through_family'].replace({"met through family":1,"not met through family":0})
    df_ml['either_internet_adjusted'] = df['either_internet_adjusted']#.replace({"met online":1,"not met online":0})
    df_ml['met_through_as_coworkers'] = df['met_through_as_coworkers']

    # 
    # relationship histories
    #
    # #df_ml['past_marriages'] = need to do some math here Q17A & Q17B

    #
    # targets
    #
    #df_ml['w2_broke_up_y'] = (df['w2_broke_up'] == 'broke up').astype('int')
    #df_ml['w23_broke_up_y'] = df['w2w3_combo_breakup'].replace({'still together, or lost to follow-up, or partner deceased':0, 'broke up':1})  
    #df_ml['w234_broke_up_y'] = df['w234_combo_breakup'].replace({'still together at w4, or some follow-up w/o break-up':0, 'broke up at wave 2, 3, or 4':1})  
    df_ml['w2345_broke_up_y'] = df['w2345_combo_breakup'].replace({'still together at w5 or some follow-up w/o breakup':0, 'broke up at wave 2,3,4, or 5':1})  


    # this will take calculating relationship lenght at each wave of surveying
    df_ml['rel_length_start'] = df['how_long_relationship']
    # this is questionable because I don't think I know exactly when they broke up
    df_ml['rel_length_at_w2'] = df_ml['rel_length_start'] + (df['w2_days_elapsed']/365)
    df_ml['rel_length_at_w2'].fillna(df_ml['rel_length_start'] + np.mean(df['w2_days_elapsed'])/365, inplace=True)
    
    #
    # 
    #
    
    #
    # Drop NA
    # hold off on drop na to do auto viz
    #df_ml = df_ml.dropna()
    
    # drop redundant columns
    return df_ml.drop(['rel_length_start', 'caseid_new'], axis=1)

def train_models_grid(X_train, y_train, scoring='precision'):
    print('train')
    tree = GridSearchCV(estimator =  DecisionTreeClassifier(), 
                        scoring=scoring, 
                        cv=5, 
                        n_jobs=-1, 
                        verbose=2, 
                        param_grid={
                            "class_weight": ['balanced'], #{1 : 2, 0 : 1}
                            "min_samples_split": [8]#[4, 8, 16, 32]
                        }
                       )
    tree.fit(X_train, y_train)

    
    forest = GridSearchCV(estimator = RandomForestClassifier(bootstrap=True), 
                          param_grid={ 
                              'class_weight': ['balanced'], #{1 : 2, 0 : 1}
                              'max_features': ['log2'], 
                              'min_samples_split': [4],#[4, 8, 16, 32], 
                              'n_estimators': [500]
                          },
                          scoring=scoring, 
                          cv=5, 
                          verbose=2, 
                          n_jobs=-1
                         )
    forest.fit(X_train, y_train)

    return {
        'treeGrid': tree,
        'forestGrid':forest,
        'scoring': scoring
    }

# def scaled_models():
#     print("6.Preprocessing categorial data")
#     cus_type_category=processed_data['cus_type'].unique();
#     cus_type = CategoricalDtype(categories=cus_type_category, ordered=True)
#     x_train['cus_type']=x_train['cus_type'].astype(cus_type)
#     x_train=pd.get_dummies(x_train,prefix='cus')

#     print("7.Preprocessing continuous data")
#     cus_point_scaler = MinMaxScaler()
#     x_train["cus_point"]=cus_point_scaler.fit_transform(x_train[["cus_point"]])
    
#     print("9.Training the model")
#     lr = LogisticRegression(max_iter=1000,solver='lbfgs',multi_class='auto')
#     model_lr = lr.fit(x_train,y_train)

#     print("10.Dumping the moodel to s3 bucket")
#     dump_data={ "model":model_lr, 
#                 "event_matrix": event_matrix,
#                 "customer_type": cus_type_category,
#                 "cus_point_scaler":cus_point_scaler }

def prep_data(df):
    print("prep")
    df_check = df.dropna()
    df_check = pd.get_dummies(df_check, drop_first=False)

    return train_test_split(df_check.drop(['w2345_broke_up_y'],axis=1), 
                            df_check['w2345_broke_up_y'], 
                            test_size=.25, 
                            random_state=555)
    


def main():
    print("1.Loading data")
    X_train, X_test, y_train, y_test = prep_data(clean_data(load_data()))
    models = train_models_grid(X_train, y_train, scoring='precision')
    #tree = models['treeGrid']
    model = models['forestGrid'].best_estimator_

    print("10.Dumping the moodel to s3 bucket")
    dump_data={ "model":model }

    util.upload_model_to_s3(dump_data)
    
main()