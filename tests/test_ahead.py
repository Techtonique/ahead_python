#!/usr/bin/env python

"""Tests for `ahead` package."""

# python -m unittest

import unittest
import pandas as pd
from ahead import EAT, DynamicRegressor, Ridge2Regressor, VAR


# Univariate dataset 
# Data frame containing the time series 
dataset_uni = {
    'date' : ['2020-01-01', '2020-02-01', '2020-03-01', '2020-04-01', '2020-05-01'],
    'value' : [34, 30, 35.6, 33.3, 38.1],    
}
df_uni = pd.DataFrame(dataset_uni).set_index('date')

# Multivariate dataset 
# Data frame containing the time series 
dataset_multi = {
 'date' : ['2001-01-01', '2002-01-01', '2003-01-01', '2004-01-01', '2005-01-01'],
 'series1' : [34, 30, 35.6, 33.3, 38.1],    
 'series2' : [4, 5.5, 5.6, 6.3, 5.1],
 'series3' : [100, 100.5, 100.6, 100.2, 100.1]}
df_multi = pd.DataFrame(dataset_multi).set_index('date')


# Forecasting horizon
h = 5


e1 = EAT(h = h, weights = [0.5, 0.4, 0.1], date_formatting = "ms")
e2 = EAT(h = h, type_pi="T", date_formatting = "original")        
e1.forecast(df_uni)
e2.forecast(df_uni)

print("EAT ---------- \n")
print(e1.averages)
print("\n")
print(e1.ranges)
print("\n")
print(e2.averages)
print("\n")
print(e2.ranges)
print("\n")


e3 = DynamicRegressor(h = h, date_formatting = "ms")
e4 = DynamicRegressor(h = h, date_formatting = "original")
e3.forecast(df_uni)
e4.forecast(df_uni)

print("DynamicRegressor ---------- \n")
print(e3.averages)
print("\n")
print(e3.ranges)
print("\n")
print(e4.averages)
print("\n")
print(e4.ranges)
print("\n")

e5 = Ridge2Regressor(h = h, date_formatting = "ms")
e6 = Ridge2Regressor(h = h, date_formatting = "original")
e5.forecast(df_multi)
e6.forecast(df_multi)
print("Ridge2Regressor ---------- \n")
print(e5.averages)
print("\n")
print(e5.ranges)
print("\n")
print(e6.averages)
print("\n")
print(e6.ranges)
print("\n")

e7 = VAR(h = h, date_formatting = "ms", type_VAR="none")
e8 = VAR(h = h, date_formatting = "original", type_VAR="none")
e7.forecast(df_multi)
e8.forecast(df_multi)
print("VAR ---------- \n")
print(e7.averages)
print("\n")
print(e7.ranges)
print("\n")
print(e8.averages)
print("\n")
print(e8.ranges)
print("\n")


class TestAhead(unittest.TestCase):
    """Tests for `ahead` package."""

    def setUp(self):
        """Set up test fixtures, if any."""

    def tearDown(self):
        """Tear down test fixtures, if any."""

    def test_eat(self):            
        self.assertAlmostEqual(e1.averages[0][1], 34.28750002378786)
        self.assertAlmostEqual(e1.ranges[0][1], 28.95860083805203)
        self.assertAlmostEqual(e2.averages[0][1], 37.833333650504755)
        self.assertAlmostEqual(e2.ranges[0][1], 35.39276803598083)
        self.assertEqual(e1.averages[0][0], 1590962400000)
        self.assertEqual(e2.averages[0][0], '2020-06-01')

    def test_dynrm(self):        
        self.assertAlmostEqual(e3.averages[0][1], 36.79847159323566)
        self.assertAlmostEqual(e3.ranges[0][1], 29.66346091111926)
        self.assertAlmostEqual(e4.averages[0][1], 36.79847159323566)
        self.assertAlmostEqual(e4.ranges[0][1], 29.66346091111926)
        self.assertEqual(e3.averages[0][0], 1590962400000)        
        self.assertEqual(e4.averages[0][0], '2020-06-01')        
    
    def test_ridge2(self):
        self.assertAlmostEqual(e5.averages[0][0][1], 33.99538584327151)
        self.assertAlmostEqual(e6.averages[0][0][1], 33.99538584327151)
        self.assertAlmostEqual(e5.averages[0][0][0], 1136070000000)
        self.assertAlmostEqual(e6.averages[0][0][0], '2006-01-01')

    def test_var(self):            
        self.assertAlmostEqual(e7.averages[0][0][1], 31.44666737744406)
        self.assertAlmostEqual(e8.averages[0][0][1], 31.44666737744406)
        self.assertAlmostEqual(e7.averages[0][0][0], 1136070000000)
        self.assertAlmostEqual(e8.averages[0][0][0], '2006-01-01')

