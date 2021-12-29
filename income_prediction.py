import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

# A class that changes category names according to specified rules
class Country2ContinentConverter(BaseEstimator, TransformerMixin):
    ''' This transformer revises categories according to a dictionary with rules '''
    def __init__(self, country_col='native-country', continent_col='continent', conversion_rules={}):
        self.country_col = country_col
        self.continent_col = continent_col
        self.conversion_rules = conversion_rules
        
    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        X_transformed = X.copy()
        X_transformed[self.continent_col] = ''
        for country in self.conversion_rules:
            X_transformed.loc[X[self.country_col]==country, self.continent_col] = self.conversion_rules[country]
        return X_transformed

# A class that changes category names according to specified rules
class CategoryReviser(BaseEstimator, TransformerMixin):
    ''' This transformer revises categories according to a dictionary with rules '''
    def __init__(self, cat_change_rules={}):
        self.cat_change_rules = cat_change_rules

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        X_transformed = X.copy()
        for feature in self.cat_change_rules:
            for cat in self.cat_change_rules[feature]:
                X_transformed.loc[X[feature]==cat, feature] = self.cat_change_rules[feature][cat]
        return X_transformed

# Here, we regularize the target encodeing by taking the weighted average of the positive rate for the specific category
# and the positive rate for the entire dataset. The argument 'reg_weight' controls the weighting.
class TargetEncoder(BaseEstimator, TransformerMixin):
    ''' This transformer replaces a category with the regularized positive rate for that category '''
    def __init__(self, features, reg_weight=20):
        self.features = features
        self.reg_weight = reg_weight
        self.mapping = {}

    def fit(self, X, y=None):
        self.X_positive_rate = y.mean()
        X_y = X.copy()
        X_y['target'] = y
        for feat in self.features:
            for feat in self.features:
                self.mapping[feat] = {}
                positive_rates = X_y.groupby(feat)['target'].mean()
                value_counts = X_y[feat].value_counts()
                for cat in X_y[feat].unique():
                    n = value_counts.loc[cat]
                    rate = positive_rates.loc[cat]
                    regularized_rate = (rate * n + self.X_positive_rate * self.reg_weight) / (n + self.reg_weight)
                    self.mapping[feat][cat] = regularized_rate
        return self
    
    def transform(self, X, y=None):
        X_transformed = X.copy()
        for feat in self.mapping:
            for cat in X[feat].unique():
                try:
                    # If the category was mapped during fitting, change it. 
                    X_transformed.loc[X[feat]==cat, feat] = self.mapping[feat][cat]
                except: 
                    # If the category is unknown, replace it with the mean positive rate
                    X_transformed.loc[X[feat]==cat, feat] = self.X_positive_rate
                    
        return X_transformed

class CharacterStripper(BaseEstimator, TransformerMixin):
    ''' Strip a charapter from the end of a string'''
    def __init__(self, character_to_strip='.'):
        self.character_to_strip = character_to_strip

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        X_transformed = X
        X_transformed = X_transformed.str.strip(self.character_to_strip)
        return X_transformed

def get_data(columns_to_drop=None):
    ''' Downloan the data, and create train and test sets. '''
    df_1 = pd.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data", 
                   names=['age', 'workclass', 'fnlwgt', 'education', 'education-num', 
                   'marital-status', 'occupation', 'relationship', 'race', 'sex', 
                   'capital-gain', 'capital-loss', 'hours-per-week', 'native-country', 
                   '>=50K'], skipinitialspace=True)
    df_2 = pd.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.test", 
                   names=['age', 'workclass', 'fnlwgt', 'education', 'education-num', 
                   'marital-status', 'occupation', 'relationship', 'race', 'sex', 
                   'capital-gain', 'capital-loss', 'hours-per-week', 'native-country', 
                   '>=50K'], skipinitialspace=True, skiprows=1)
    df_combined = df_1.append(df_2, ignore_index=True, sort=True)
    if columns_to_drop:
        df_combined = df_combined.drop(columns=columns_to_drop)
    X = df_combined.drop(columns=['>=50K'])
    y = df_combined['>=50K']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)
    return X_train, X_test, y_train, y_test

def preprocess_y(y):
    ''' Remove dots from the end and encode the target feature. '''
    character_stripper = CharacterStripper(character_to_strip='.')
    y_stripped = character_stripper.fit_transform(y)
    label_encoder = LabelEncoder()
    y_train_prepared = label_encoder.fit_transform(y_stripped)
    return y_train_prepared

def build_X_pipeline_full(predictor=None):
    ''' Build pipeline with all the features. '''
    num_features = ['age', 'fnlwgt', 'education-num', 'capital-gain', 'capital-loss', 'hours-per-week', 'native-country',]
    cat_features = ['workclass', 'marital-status', 'occupation', 'relationship', 'race', 'sex', 'continent']

    cat_change_rules = {'marital-status':{'Married-AF-spouse':'Married', 'Married-civ-spouse':'Married'}, 
                        'workclass':{'Without-pay':'?', 'Never-worked':'?'},
                        'occupation':{'Armed-Forces':'Prof-specialty'}}
    country_2_contonent_rules = {'United-States':'N-America',
                         'Germany':'Europe',
                         'Mexico':'LatAm',
                         'Scotland':'Europe',
                         'Peru':'LatAm',
                         'Honduras':'LatAm',
                         'Ecuador':'LatAm',
                         'Poland':'Europe',
                         'China':'Asia',
                         'Nicaragua':'LatAm',
                         'India':'Asia',
                         'Philippines':'Asia',
                         'Iran':'Asia',
                         'Japan':'Asia',
                         'Vietnam':'Asia',
                         'Dominican-Republic':'LatAm',
                         'Ireland':'Europe',
                         'Laos':'Asia',
                         'Jamaica':'LatAm',
                         'England':'Europe',
                         'Hong':'Asia',
                         'Puerto-Rico':'LatAm',
                         'Cuba':'LatAm',
                         'Haiti':'LatAm',
                         'Guatemala':'LatAm',
                         'El-Salvador':'LatAm',
                         'Columbia':'LatAm',
                         'Italy':'Europe',
                         'Taiwan':'Asia',
                         'Canada':'N-America',
                         'Portugal':'Europe',
                         'Thailand':'Asia',
                         'Cambodia':'Asia',
                         'France':'Europe',
                         'Greece':'Europe',
                         'Trinadad&Tobago':'LatAm',
                         'Yugoslavia':'Europe',
                         'Hungary':'Europe',
                         'Holand-Netherlands':'Europe',
                        }

    # Create and parametatrize data transformers
    scaler = StandardScaler()
    one_hot_encoder = OneHotEncoder(handle_unknown='ignore')
    category_reviser = CategoryReviser(cat_change_rules=cat_change_rules)
    country_2_continent = Country2ContinentConverter(country_col='native-country', 
                                                     continent_col='continent', 
                                                     conversion_rules=country_2_contonent_rules)
    target_encoder = TargetEncoder(['native-country'])

    column_transformer = ColumnTransformer([('Scaler', scaler, num_features), 
                                            ('One Hot Encoder', one_hot_encoder, cat_features)])

    # Define the data transformation pipeline
    X_pipeline = Pipeline([('CategoryReviser', category_reviser), 
                           ('Country2Continent', country_2_continent), 
                           ('TargetEncoder', target_encoder), 
                           ('ColumnTransformer', column_transformer),
                           ('predictor', predictor)])
    
    if predictor == None:
        X_pipeline.steps.pop()
    
    return X_pipeline

def build_X_pipeline_partial(predictor=None):
    ''' Build pipeline without ethically sensitive features. '''
    num_features = ['age', 'fnlwgt', 'education-num', 'capital-gain', 'capital-loss', 'hours-per-week']
    cat_features = ['workclass', 'occupation']

    cat_change_rules = {'workclass':{'Without-pay':'?', 'Never-worked':'?'},
                        'occupation':{'Armed-Forces':'Prof-specialty'}}

    # Create and parametatrize data transformers
    scaler = StandardScaler()
    one_hot_encoder = OneHotEncoder(handle_unknown='ignore')
    category_reviser = CategoryReviser(cat_change_rules=cat_change_rules)

    column_transformer = ColumnTransformer([('Scaler', scaler, num_features), 
                                            ('One Hot Encoder', one_hot_encoder, cat_features)])

    # Define the data transformation pipeline
    X_pipeline = Pipeline([('CategoryReviser', category_reviser), 
                           ('ColumnTransformer', column_transformer),
                           ('predictor', predictor)])
    
    if predictor == None:
        X_pipeline.steps.pop()
    
    return X_pipeline

def mean_value_score(y_true, y_pred, tp_profit=60, tn_profit=20):
    ''' Custom performance metric that calculate the mean monetary profit and loss implications (per row) of our  
        predictions, relative to the alternative of targeting all potential customers. 
    '''
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    total_profit = tp_profit * (tp - fn) + tn_profit * (tn - fp)
    mean_profit = total_profit / len(y_true)
    return mean_profit    