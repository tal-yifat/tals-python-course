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
        