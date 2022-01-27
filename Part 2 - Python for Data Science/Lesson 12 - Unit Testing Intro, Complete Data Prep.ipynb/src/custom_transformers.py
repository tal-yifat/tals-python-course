import pandas as pd
import numpy as np
import warnings

from sklearn.base import BaseEstimator, TransformerMixin

class CharacterStripper(BaseEstimator, TransformerMixin):
    def __init__(self, character_to_strip='.'):
        self.character_to_strip = character_to_strip

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        X_transformed = X
        X_transformed = X_transformed.str.strip(self.character_to_strip)
        return X_transformed
        
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