import pandas as pd
import numpy as np
import pytest
import sys

sys.path.append('C:/Users/tyifat/Workspace/python-for-dna/Season 2/lesson 12/src')

from custom_transformers import CharacterStripper, Country2ContinentConverter

class TestCharacterStripper():
    def test_on_normal_input(self):
        data = pd.Series(['a.', 'b', 'c...', 'd'])
        char_stripper = CharacterStripper(character_to_strip='.')
        expected = pd.Series(['a', 'b', 'c', 'd'])
        actual = char_stripper.fit_transform(data)
        pd.testing.assert_series_equal(expected, actual)


class TestCountry2ContinentConverter():
    def test_on_normal_data_1(self):
        data = pd.DataFrame({'number':[1, 2, 3, 4, 5],
                            'country':['U.S.', 'China', 'India', 'U.S.', 'Mexico']})
        conversion_rules = {'U.S.':'America',
                           'Mexico':'America',
                           'China':'Asia',
                           'India':'Asia'}
        country_2_continent = Country2ContinentConverter(country_col='country', 
                                                     continent_col='continent', 
                                                     conversion_rules=conversion_rules)
        expected = pd.DataFrame({'number':[1, 2, 3, 4, 5],
                            'country':['U.S.', 'China', 'India', 'U.S.', 'Mexico'],
                             'continent':['America', 'Asia', 'Asia', 'America', 'America']})
        actual = country_2_continent.fit_transform(data)
        pd.testing.assert_frame_equal(expected, actual)

    def test_on_normal_data_2(self):
        pass
    
    def test_on_missing_value(self):
        data = pd.DataFrame({'number':[1, 2, 3, 4, 5],
                            'country':['U.S.', np.nan, 'India', 'U.S.', 'Mexico']})
        conversion_rules = {'U.S.':'America',
                           'Mexico':'America',
                           'China':'Asia',
                           'India':'Asia'}
        country_2_continent = Country2ContinentConverter(country_col='country', 
                                                     continent_col='continent', 
                                                     conversion_rules=conversion_rules)
        expected = pd.DataFrame({'number':[1, 2, 3, 4, 5],
                            'country':['U.S.', np.nan, 'India', 'U.S.', 'Mexico'],
                             'continent':['America', '', 'Asia', 'America', 'America']})
        actual = country_2_continent.fit_transform(data)
        pd.testing.assert_frame_equal(expected, actual)
    
    def test_on_unknown_value(self):
        data = pd.DataFrame({'number':[1, 2, 3, 4, 5],
                            'country':['U.S.', 'Argentina', 'India', 'U.S.', 'Mexico']})
        conversion_rules = {'U.S.':'America',
                           'Mexico':'America',
                           'China':'Asia',
                           'India':'Asia'}
        country_2_continent = Country2ContinentConverter(country_col='country', 
                                                     continent_col='continent', 
                                                     conversion_rules=conversion_rules)
        expected = pd.DataFrame({'number':[1, 2, 3, 4, 5],
                            'country':['U.S.', 'Argentina', 'India', 'U.S.', 'Mexico'],
                             'continent':['America', '', 'Asia', 'America', 'America']})
        actual = country_2_continent.fit_transform(data)
        pd.testing.assert_frame_equal(expected, actual)
    
    def test_on_missing_column(self):
        pass

