import warnings
warnings.filterwarnings('ignore')
import pandas as pd
import numpy as np


class FeatureExtractor:

    def __init__(self, df:pd.DataFrame):
        self.df = df.copy()

    def _create_time_features(self):
        self.df['date'] = pd.to_datetime(self.df['date'])
        self.df['year'] = self.df['date'].dt.year
        self.df['month'] = self.df['date'].dt.month
        self.df['period'] = self.df['year'].astype(str) + '-' + self.df['month'].astype(str).str.zfill(2)
        self.df['month_sin'] = np.sin(2 * np.pi * self.df['month'] / 12)
        self.df['month_cos'] = np.cos(2 * np.pi * self.df['month'] / 12)
        self.df['is_high_season'] = self.df['month'].isin([10, 11, 12]).astype(int)
        self.df['is_low_season'] = self.df['month'].isin([3, 4, 5]).astype(int)
        return self.df


    def _aggregate(self):
        self.df = self.df.groupby(['period', 'item_id', 'shop_id']).agg({
            'item_price':'mean',
            'item_cnt_day':'sum',
            'item_name': 'first',
            'item_category_id': 'first',
            'item_category_name': 'first',
            'shop_name':'first',
            'city':'first',
            'year':'first',
            'month':'first',
            'month_sin': 'first',
            'month_cos': 'first',
            'is_high_season': 'first',
            'is_low_season': 'first'
        }).reset_index().rename(columns={'item_cnt_day':'item_cnt_month'})
        self.df['monthly_revenue'] = self.df.loc[:,'item_price'] * self.df.loc[:,'item_cnt_month']
        self.df.set_index('period', inplace=True)
        self.df.sort_index(inplace=True)
        return self.df
    

    def _log_transform(self):
        self.df['item_price_log'] = np.log1p(self.df.loc[:, 'item_price'])
        self.df['monthly_revenue_log'] = np.log1p(self.df.loc[:, 'monthly_revenue'])
        self.df['item_cnt_month_log'] = np.log1p(self.df.loc[:, 'item_cnt_month'])
        return self.df
    
    def _generalize_category(self):
        gen_cats = self.df['item_category_name'].str.split('-').str[0]
        self.df['gen_cat'] = gen_cats 
        return self.df

    def _qcut_expansion(self):
        self.df['price_level'] = pd.qcut(x=self.df['item_price'], q=4, labels=['cheap', 'medium', 'expensive', 'extraordinary'])

        category_sales = self.df.groupby('gen_cat')['item_cnt_month'].sum()
        category_popularity_map = pd.qcut(category_sales, 
                                  q=4, 
                                  labels=['unpopular', 'semi-popular', 'popular', 'very_popular'])
        self.df['category_popularity'] = self.df['gen_cat'].map(category_popularity_map)

        shop_sales = self.df.groupby('shop_name')['item_cnt_month'].sum()
        shop_popularity_map = pd.qcut(shop_sales, 
                                  q=4, 
                                  labels=['unpopular', 'semi-popular', 'popular', 'very_popular'])
        self.df['shop_popularity'] = self.df['shop_name'].map(shop_popularity_map)

        city_sales = self.df.groupby('city')['item_cnt_month'].sum()
        city_popularity_map = pd.qcut(city_sales, 
                                  q=4, 
                                  labels=['unpopular', 'semi-popular', 'popular', 'very_popular'])
        self.df['city_popularity'] = self.df['city'].map(city_popularity_map)

        return self.df
    

    def _create_lags(self):
        for i in [1, 2, 3]:
            self.df[f'sales_lag_{i}'] = self.df['item_cnt_month'].shift(i).fillna(0)

        self.df['sales_rolling_mean_3'] = self.df['item_cnt_month'].rolling(3).mean().fillna(0)
        self.df['sales_rolling_mean_6'] = self.df['item_cnt_month'].rolling(6).mean().fillna(0)

        self.df['sales_lag_12'] = self.df['item_cnt_month'].shift(12).fillna(0)
        return self.df
    
    
    def _drop_reduntant(self):
        self.df = self.df.drop(['item_id', 'shop_id', 'item_category_id', 'item_category_name',
                                 'month', 'item_cnt_month', 'monthly_revenue', 'item_price'], axis=1)
        return self.df
    

    def features_extract(self):
        self.df = self._create_time_features()
        self.df = self._aggregate()
        self.df = self._log_transform()
        self.df = self._generalize_category()
        self.df = self._qcut_expansion()
        self.df = self._create_lags()
        self.df = self._drop_reduntant()
        return self.df
    