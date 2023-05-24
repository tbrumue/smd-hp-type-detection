import pandas as pd 
import numpy as np
from scipy.signal import argrelextrema
import warnings
from multiprocessing import Pool, cpu_count
from functools import partial


# ------------ Helper Methods ------------

def create_time_horizons_dataframe(df:pd.DataFrame, key_timestamp:str, time_horizons:list=['yearly', 'monthly', 'weekly', 'daily']): 
    '''
        Multiple functions of the pipeline (such as the FeatureGenerator or the calculations of operating hours) use the idea of time horizons. 
        Hence, you hand over a data frame with all available time stamps, but you want to calculate something for a given time horizon, e.g. per day, week, month or year. 
        Therefore, this method creates a data frame as empty shell where each row corresponds to one unique entry as defined by the desired time horizons. 
        This empty shell can then be used in other methods which allows for parallel execution. 
        Args: 
            df: the data frame (probably containing smart meter data) to be used as basis 
            key_timestamp: string defining the timestamp column in df
            time_horizons: list of strings to define in which time_horizons the calculations should be done - can be combination of the following: yearly, monthly, weekly, daily, flexible (i.e. over whole data frame)
    '''

    time_horizons = list(set(time_horizons))
    for time_horizon in time_horizons: 
        assert isinstance(time_horizon, str), 'create_time_horizons_dataframe(): Type of parameter time_horizons needs to be a list of strings, but given type of value in list was {}.'.format(type(time_horizon))
        assert time_horizon in ['daily', 'weekly', 'monthly', 'yearly', 'flexible'], 'create_time_horizons_dataframe(): Values in parameter time_horizons can only be one of the following strings: daily, weekly, monthly, yearly, flexible. But given was: {}'.format(time_horizon)

    # create a preprocessed copy of the data frame with additional columns 
    df = preprocess_data_frame_with_timestamp(df, key_timestamp)

    # data frame defining the empty shell 
    df_final = pd.DataFrame()

    for time_horizon in time_horizons: 
        
        if time_horizon == 'flexible': # means take the whole data frame as it is 
            # data frame to store the information of one time_horizon in 
            df_temp = pd.DataFrame()
            df_temp["time_horizon"]  = pd.Series(time_horizon)
            df_temp['dayOfWeek'] = np.nan
            df_temp['week'] = np.nan
            df_temp['month'] = np.nan
            df_temp['year'] = np.nan
            df_temp["time_start"]  = pd.Series(df[key_timestamp].min())
            df_temp["time_end"] = pd.Series(df[key_timestamp].max())

            # do the concatenation
            df_final = pd.concat([df_final, df_temp], axis=0)
        
        else: 
            # maps time horizon to column name in preprocesses data frame 
            col_mapping = {
                'daily' : 'dayOfWeek',
                'weekly' : 'week', 
                'monthly' : 'month',
                'yearly' : 'year'
            }

            # maps time horizon to column names that are needed for unique combinations to create parameters 
            uniques_mapping = {
                'daily' : ['year', 'month', 'week', 'dayOfWeek'],
                'weekly' : ['year', 'month', 'week'], 
                'monthly' : ['year', 'month'],
                'yearly' : ['year']
                }

            col_identifier = col_mapping[time_horizon]
            uniques_list = uniques_mapping[time_horizon] 

            df_unique = df.groupby(uniques_list)[col_identifier].unique().to_frame().rename(columns={col_identifier: 'count'})
            df_unique.reset_index(inplace=True)
            df_unique.drop('count', axis=1, inplace=True)

            # now look through unique combinations of days, weeks, months and years and calculate the features correspondingly 
            for idx, row in df_unique.iterrows(): 
                if time_horizon == 'daily': 
                    df_extracted = df[(df['year'] == row['year']) & (df['month'] == row['month']) & (df['week'] == row['week']) & (df['dayOfWeek'] == row['dayOfWeek'])]
                if time_horizon == 'weekly': 
                    df_extracted = df[(df['year'] == row['year']) & (df['month'] == row['month']) & (df['week'] == row['week'])]
                if time_horizon == 'monthly': 
                    df_extracted = df[(df['year'] == row['year']) & (df['month'] == row['month'])]
                if time_horizon == 'yearly': 
                    df_extracted = df[(df['year'] == row['year'])]

                # data frame to store the information of one time_horizon in 
                df_temp = pd.DataFrame()

                # create meta data columns
                df_temp["time_horizon"]  = pd.Series(time_horizon)
                df_temp['dayOfWeek'] = pd.Series(row['dayOfWeek']) if 'dayOfWeek' in uniques_list else np.nan
                df_temp['week'] = pd.Series(row['week']) if 'week' in uniques_list else np.nan
                df_temp['month'] = pd.Series(row['month']) if 'month' in uniques_list else np.nan
                df_temp['year'] = pd.Series(row['year'])
                df_temp['time_start']  = pd.Series(df_extracted[key_timestamp].min())
                df_temp['time_end'] = pd.Series(df_extracted[key_timestamp].max())

                # do the concatenation
                df_final = pd.concat([df_final, df_temp], axis=0)
        
    df_final['time_start'] = pd.to_datetime(df_final['time_start'])
    df_final['time_end'] = pd.to_datetime(df_final['time_end'])
    df_final.reset_index(drop=True, inplace=True)
    return df_final


def preprocess_data_frame_with_timestamp(df, key_timestamp): 
    '''
        Preprocesses a data frame (e.g. for smart meter data) with timestamps in following steps:
            1. copies data frame
            2. transforms time stamp column to be of the right pandas type 
            3. calculates additional columns  
        Args: 
            df: data frame to be processed (needs to include columns that is given by the parameter key_timestamp)
            key_timestamp: string identifier of the timestamp column in the given data frame
        Returns: 
            processed copy of the data frame with additional columns
    '''
    
    assert isinstance(df, pd.DataFrame), 'Cannot preprocess given data frame. Parameter df must be of type pd.DataFrame, but given was: {}'.format(type(df))
    assert isinstance(key_timestamp, str), 'Cannot preprocess given data frame. Parameter key_timestamp must be of type string, but given was: {}'.format(type(key_timestamp))
    assert key_timestamp in df.columns.values, 'Cannot preprocess given data frame because parameter key_timestamp was assigned to be {}, but is not a column of the data frame'.format(key_timestamp)
    
    # TODO: add sunset / sunrise times? 

    df = df.copy()
    df[key_timestamp] = pd.to_datetime(df[key_timestamp])
    df = df.sort_values(by=key_timestamp)
    df["date"] = df[key_timestamp].dt.date
    df["time"] = df[key_timestamp].dt.time
    df["year"] = df[key_timestamp].dt.year
    df["month"] = df[key_timestamp].dt.month
    df["week"] = df[key_timestamp].dt.isocalendar().week
    df["dayOfWeek"] = df[key_timestamp].dt.dayofweek
    df["hour"] = df[key_timestamp].dt.hour
    df["minute"] = df[key_timestamp].dt.minute

    # mapping months to seasons: 1- winter, 2-spring, 3-summer, 4-autumn 
    seasons = {1: 1, 2: 1, 3: 2, 4: 2, 5:2, 6:3, 7:3, 8:3, 9:4, 10:4, 11:4, 12:1} # maps months to seasons: 
    df['season'] = df['month'].map(seasons, na_action=None)
    df['winter'] = np.where(df['season'] == 1, True, False)
    df['spring'] = np.where(df['season'] == 2, True, False)
    df['summer'] = np.where(df['season'] == 3, True, False)
    df['autumn'] = np.where(df['season'] == 4, True, False)
    df['transitionperiod'] = np.where((df['season'] == 2) | (df['season'] == 4), True, False)

    # mapping of high tarriff and low tarriff (EKZ times) - HT = True; NT = False
    df["HT"] = np.where(((df['dayOfWeek'] <= 4) & ((df['hour'] >= 7) & (df['hour'] <20))) | ((df['dayOfWeek'] == 5) & ((df['hour'] >= 7) & (df['hour'] <13))), True, False)
    df['NT'] = np.where(df['HT'], False, True)

    # mapping if times of the day are met
    df['weekday'] = np.where(df['dayOfWeek'] <=4, True, False)
    df['weekend'] = np.where(df['dayOfWeek'] >=5, True, False)
    df['morning'] = np.where((df['hour'] >= 6) & (df['hour'] < 10), True, False)
    df['noon'] = np.where(((df['hour'] >= 10) & (df['hour'] < 14)), True, False)
    df['afternoon'] = np.where((df['hour'] >= 14) & (df['hour'] < 18), True, False)
    df['evening'] = np.where((df['hour'] >= 18) & (df['hour'] < 23), True, False)
    df['day'] = np.where((df['hour'] < 23) & (df['hour'] >= 6), True, False)
    df['night'] = np.where((df['hour'] >= 23) | (df['hour'] < 6), True, False)
    return df

def check_data_frame_preprocessed(df): 
    '''
        Checks if columns that are created during preprocessing of data frame already exist.
        Args: 
            df: data frame to be processed (needs to include columns that were defines as key_timestamp and key_consumption)
        Returns: 
            True if data frame was already preprocessed
    '''
    assert isinstance(df, pd.DataFrame), 'check_data_frame_preprocessed(): Parameter df must be of type pd.DataFrame.'
    check_cols = ['date', 'time', 'year', 'month', 'week', 'dayOfWeek', 'hour', 'minute', 'HT', 'NT', 'season', 'winter', 'spring', 'summer', 'autumn', 'transitionperiod', 'weekday', 'weekend', 'morning', 'noon', 'afternoon', 'evening', 'day', 'night']
    for c in check_cols: 
        if c not in df.columns.values: 
            return False
    return True

def remove_preprocessing_columns(df): 
    '''
        Removes columns that were created after having called preprocess_data_frame_with_timestamp().
        Args: 
            df: data frame to be processed 
        Returns: 
            copy of data frame with removed columns
    '''
    assert isinstance(df, pd.DataFrame), 'remove_preprocessing_columns(): Parameter df must be of type pd.DataFrame.'
    df = df.copy()
    potential_columns = ['date', 'time', 'year', 'month', 'week', 'dayOfWeek', 'hour', 'minute', 'season', 'winter', 'spring', 'summer', 'autumn', 'transitionperiod', 'HT', 'NT', 'weekday', 'weekend', 'morning', 'noon', 'afternoon', 'evening', 'day', 'night']
    colums_to_delete = []
    existing_columns = df.columns.values
    for e in potential_columns: 
        if e in existing_columns: 
            colums_to_delete.append(e)
    return df.drop(columns=colums_to_delete) 


# ------------ Feature Generation Class ------------

class FeatureGeneration():

    def __init__(self, key_timestamp:str, key_consumption:str):
        '''
            Provides a class for generating features from the data.
            Args: 
                key_timestamp: keyword for timestamp 
                key_consumption: keyword for consumption column (in kWh)
                handler: object of class Handler (for logging) - if None: class creates just normal print-outs for logging
                basepath: optional string to folder for saving features when not handing over a handler object 
                suppress_logs: Boolean to set to True when print outs should be suppressed - NOTE: this will also affect the logs. No logs will be made!
        '''
        self.key_timestamp = key_timestamp
        self.key_consumption = key_consumption

    def __check_key_columns__(self, df:pd.DataFrame): 
        '''
            Checks if assigned key_timestamps and key_consumption are columns in a given data frame
            Args: 
                df: data frame to be checked
            Returns: 
                True if keys exist 
        '''
        all_good = True 
        if self.key_consumption not in df.columns.values: 
            print('FeatureGenerator: Cannot process given data frame because parameter key_consumption was assigned to be {}, but is not a column of the data frame'.format(self.key_consumption))
            all_good = False
        if self.key_timestamp not in df.columns.values: 
            print('FeatureGenerator: Cannot process given data frame because parameter key_timestamp was assigned to be {}, but is not a column of the data frame'.format(self.key_timestamp))
            all_good = False 
        return all_good

    def __preprocess_data_frame__(self, df:pd.DataFrame): 
        '''
            Preprocesses a data frame in following steps: 
                1. copies data frame
                2. transforms time stamp column to be of the right pandas type 
                3. calculates new columns  
            Args: 
                df: data frame to be processed (needs to include columns that were defined as key_timestamp and key_consumption)
            Returns: 
                processed copy of the data frame 
        '''
        assert self.__check_key_columns__(df), 'FeatureGenerator: Data Frame does not satisfy the consumption and timestamp column assignments.'
        df = preprocess_data_frame_with_timestamp(df, self.key_timestamp)
        return df

    def __check_data_frame_preprocessed__(self, df): 
        '''
            Checks if columns that are create during preprocessing of data frame already exist, i.e. prevent preprocessing multiple times.
            Args: 
                df: data frame to be processed (needs to include columns that were defines as key_timestamp and key_consumption)
            Returns: 
                True if data frame was already preprocessed
        '''
        return check_data_frame_preprocessed(df)


    def __calculate_features__(self, row, df:pd.DataFrame): 
        '''
            Calculates the features of one metering_code for given time frame of smart meter data in any-min-granularity.
            NOTE: this method assumes that df contains only consumption values of a single metering code and does not double check for it.
                Partially according to: 
                - https://link.springer.com/article/10.1007%2Fs00450-014-0294-4 - "Feature extraction and filtering for household classification based on smart electricity meter data"
                - https://link.springer.com/article/10.1007/s12525-018-0290-9 - "Enhancing energy efficiency in the residential sector with smart meter data analytics" 
                
            NOTE: there are abbreviations in the variables: c = consumption, r = ratios, s = statistical, t=temporal, w=weather correlation

            Args: 
                row: an element from the iterrows() method of a data frame that defines the time horizons - as calculated by create_time_horizons_dataframe()
                df: data frame to be processed (needs to include columns that were defines as key_timestamp)
            Returns: 
                data frame with filled features as columns and just a single row
        '''

        # TODO: deal with empty slice stuff (especially for features t_above_x_kw), but for now just ignore warnings
        warnings.filterwarnings("ignore")

        if not self.__check_data_frame_preprocessed__(df): 
            df = self.__preprocess_data_frame__(df)

        # now filter the data frame for the right time range given by the row and set all zero consumptions to NAN
        df = df[(df[self.key_timestamp] >= row['time_start']) & (df[self.key_timestamp] <= row['time_end'])]
        df.loc[df[self.key_consumption] == 0, self.key_consumption] = np.nan

        # initialization of the features data frame 
        feats = row.to_frame().T
        feats.reset_index(drop=True, inplace=True)

        feats = self.__calculate_statistical_features__(df, feats, row['time_horizon'])
        feats = self.__calculate_histogram_features__(df, feats)

        return feats

        
    def __calculate_statistical_features__(self, df:pd.DataFrame, feats:pd.DataFrame, time_horizon:str):   
        '''
            Calculates statistical features. 
            Args: 
                df: data frame containing smart meter data 
                feats: data frame to create new columns in and to fill values with
                time_horizon: string to define the time horizon of the feature calculation - can only be: daily, weekly, monthly, or yearly
        '''
        assert time_horizon in ['daily', 'weekly', 'monthly', 'yearly'], 'FeatureGenerator.__calculate_statistical_features__(): Parameter time_horizon must be a string with one of the following values: daily, weekly, monthly, yearly.'

        if not self.__check_data_frame_preprocessed__(df): 
            df = self.__preprocess_data_frame__(df)

        # meta data about percentage of values that are non zero 
        feats['share_non_zero_vals'] = pd.Series(len(df[df[self.key_consumption] > 0]) / len(df)) 

        # maxima
        feats['s_max'] = pd.Series(df[self.key_consumption].max())
        feats['s_winter_max'] = pd.Series(df[df['winter']][self.key_consumption].max()) if time_horizon in ['yearly', 'flexible'] else np.nan
        feats['s_winter_day_max'] = pd.Series(df[df['winter'] & df['day']][self.key_consumption].max()) if time_horizon in ['yearly', 'flexible'] else np.nan
        feats['s_winter_night_max'] = pd.Series(df[df['winter'] & df['night']][self.key_consumption].max()) if time_horizon in ['yearly', 'flexible'] else np.nan

        # minima
        feats['s_min'] = pd.Series(df[self.key_consumption].min())
        feats['s_winter_min'] = pd.Series(df[df['winter']][self.key_consumption].min()) if time_horizon in ['yearly', 'flexible'] else np.nan
        feats['s_winter_day_min'] = pd.Series(df[df['winter'] & df['day']][self.key_consumption].min()) if time_horizon in ['yearly', 'flexible'] else np.nan
        feats['s_winter_night_min'] = pd.Series(df[df['winter'] & df['night']][self.key_consumption].min()) if time_horizon in ['yearly', 'flexible'] else np.nan

        # median 
        feats['s_median'] = pd.Series(df[self.key_consumption].median()) 
        feats['s_winter_median'] = pd.Series(df[df['winter']][self.key_consumption].median()) if time_horizon in ['yearly', 'flexible'] else np.nan
        feats['s_winter_day_median'] = pd.Series(df[df['winter'] & df['day']][self.key_consumption].median()) if time_horizon in ['yearly', 'flexible'] else np.nan
        feats['s_winter_night_median'] = pd.Series(df[df['winter'] & df['night']][self.key_consumption].median()) if time_horizon in ['yearly', 'flexible'] else np.nan

        # mode
        feats['s_mode'] = pd.Series(df[self.key_consumption].mode())
        feats['s_winter_mode'] = pd.Series(df[df['winter']][self.key_consumption].mode()) if time_horizon in ['yearly', 'flexible'] else np.nan
        feats['s_winter_day_mode'] = pd.Series(df[df['winter'] & df['day']][self.key_consumption].mode()) if time_horizon in ['yearly', 'flexible'] else np.nan
        feats['s_winter_night_mode'] = pd.Series(df[df['winter'] & df['night']][self.key_consumption].mode()) if time_horizon in ['yearly', 'flexible'] else np.nan

        # std
        feats['s_std'] = pd.Series(df[self.key_consumption].std())
        feats['s_winter_std'] = pd.Series(df[df['winter']][self.key_consumption].std()) if time_horizon in ['yearly', 'flexible'] else np.nan
        feats['s_winter_day_std'] = pd.Series(df[df['winter'] & df['day']][self.key_consumption].std()) if time_horizon in ['yearly', 'flexible'] else np.nan
        feats['s_winter_night_std'] = pd.Series(df[df['winter'] & df['night']][self.key_consumption].std()) if time_horizon in ['yearly', 'flexible'] else np.nan

        # skew
        feats['s_skew'] = pd.Series(df[self.key_consumption].skew())
        feats['s_winter_skew'] = pd.Series(df[df['winter']][self.key_consumption].skew()) if time_horizon in ['yearly', 'flexible'] else np.nan
        feats['s_winter_day_skew'] = pd.Series(df[df['winter'] & df['day']][self.key_consumption].skew()) if time_horizon in ['yearly', 'flexible'] else np.nan
        feats['s_winter_night_skew'] = pd.Series(df[df['winter'] & df['night']][self.key_consumption].skew()) if time_horizon in ['yearly', 'flexible'] else np.nan

        # kurtosis
        feats['s_kurtosis'] = pd.Series(df[self.key_consumption].kurtosis())
        feats['s_winter_kurtosis'] = pd.Series(df[df['winter']][self.key_consumption].kurtosis()) if time_horizon in ['yearly', 'flexible'] else np.nan
        feats['s_winter_day_kurtosis'] = pd.Series(df[df['winter'] & df['day']][self.key_consumption].kurtosis()) if time_horizon in ['yearly', 'flexible'] else np.nan
        feats['s_winter_night_kurtosis'] = pd.Series(df[df['winter'] & df['night']][self.key_consumption].kurtosis()) if time_horizon in ['yearly', 'flexible'] else np.nan

        # variances 
        feats['s_var'] = pd.Series(df[self.key_consumption].var())
        feats['s_winter_var'] = pd.Series(df[df['winter']][self.key_consumption].var()) if time_horizon in ['yearly', 'flexible'] else np.nan
        feats['s_winter_day_var'] = pd.Series(df[df['winter'] & df['day']][self.key_consumption].var()) if time_horizon in ['yearly', 'flexible'] else np.nan
        feats['s_winter_night_var'] = pd.Series(df[df['winter'] & df['night']][self.key_consumption].var()) if time_horizon in ['yearly', 'flexible'] else np.nan

        # using argrelextrema func from scipy. order specifies how many values are considered before and after
        feats['s_num_peaks'] = pd.Series(len(argrelextrema(df[self.key_consumption].values, np.greater_equal, order=1)[0]))
    
        # diff values
        feats['s_diff_mean'] = pd.Series(df[self.key_consumption].diff().abs().mean())
        feats['s_diff_sum'] = pd.Series(df[self.key_consumption].diff().abs().sum())

        # ratios between statistical features
        c_all_mean = pd.Series(df[self.key_consumption].mean())
        feats['r_mean_max'] = pd.Series(c_all_mean / feats['s_max']) if feats['s_max'].iloc[0] > 0 else np.nan
        feats['r_mean_min'] = pd.Series(c_all_mean / feats['s_min']) if feats['s_min'].iloc[0] > 0 else np.nan
        feats['r_mean_max_no_min'] = pd.Series((c_all_mean - feats["s_min"]) / (feats["s_max"] - feats["s_min"])) if (feats["s_max"] - feats["s_min"]).iloc[0] > 0 else np.nan

        # number of measurements exceeing the given threshold
        feats['t_num_above_0.125kWh'] = pd.Series(len(df[df[self.key_consumption] > 0.125]))
        feats['t_num_above_0.25kWh'] = pd.Series(len(df[df[self.key_consumption] > 0.25]))
        feats['t_num_above_0.5kWh'] = pd.Series(len(df[df[self.key_consumption] > 0.5]))
        feats['t_num_above_mean'] = pd.Series(len(df[df[self.key_consumption] > df[self.key_consumption].mean()]))
        feats['t_num_above_min'] = pd.Series(len(df[df[self.key_consumption] > df[self.key_consumption].min()]))
        feats['t_num_is_max'] = pd.Series(len(df[df[self.key_consumption] == df[self.key_consumption].max()]))

        return feats

    def __calculate_histogram_features__(self, df:pd.DataFrame, feats:pd.DataFrame):  
        '''
            Calculates histogram features. 
            # NOTE: proabably you want to use these features in combination with statistical features not not just alone! 
            # Therefore, calculating other features also makes sense.
            Args: 
                df: data frame containing smart meter data 
                feats: data frame to create new columns in and to fill values with
        '''

        assert isinstance(df, pd.DataFrame), 'FeatureGenerator.__calculate_histogram_features__(): Parameter df must be of type pd.DataFrame, but given was: {}'.format(type(df))
        assert isinstance(feats, pd.DataFrame), 'FeatureGenerator.__calculate_histogram_features__(): Parameter feats must be of type pd.DataFrame, but given was: {}'.format(type(feats))

        if not self.__check_data_frame_preprocessed__(df): 
            df = self.__preprocess_data_frame__(df)

        # this time we need to copy the data frame because we add new columns
        df = df.copy(deep=True)

        # handling that the data frame can only have one single value the whole time (e.g. nan or zero)
        if len(df[self.key_consumption].unique()) == 1: 
            feature_names = ['h_sum_norm_diff_>0.1', 'h_sum_norm_diff_>0.2', 'h_sum_norm_diff_>0.3', 'h_sum_norm_diff_>0.4', 'h_sum_norm_diff_>0.5', 'h_share_diff_>0.1', 'h_share_diff_>0.2', 'h_share_diff_>0.3', 'h_share_diff_>0.4', 'h_share_diff_>0.5',
            'h_share_at_max_bin', 'h_share_at_max_plus2_bin', 'h_share_at_max_plus4_bin', 'h_share_at_max_plus6_bin', 'h_slope_max_to_1', 'h_slope_1_to_2', 'h_slope_2_to_3', 'h_slope_3_to_4', 'h_slope_mean']
            for f in feature_names: 
                feats[f] = pd.Series(np.nan)
        
        else: 
            # calculating absolute diff
            # e.g. power-regulated HP should have way smaller differences
            diff_values = df[self.key_consumption].diff().abs()
            # exclue mini differences which are probably rather from sensor inaccuracy
            diff_values = diff_values[diff_values > 0.05] # TODO: make this an external parameter?

            # handle absolute differences
            feats["h_sum_norm_diff_>0.1"] = pd.Series((diff_values > 0.1).sum() / diff_values.size) if diff_values.size > 0 else np.nan
            feats["h_sum_norm_diff_>0.2"] = pd.Series((diff_values > 0.2).sum() / diff_values.size) if diff_values.size > 0 else np.nan
            feats["h_sum_norm_diff_>0.3"] = pd.Series((diff_values > 0.3).sum() / diff_values.size) if diff_values.size > 0 else np.nan
            feats["h_sum_norm_diff_>0.4"] = pd.Series((diff_values > 0.4).sum() / diff_values.size) if diff_values.size > 0 else np.nan
            feats["h_sum_norm_diff_>0.5"] = pd.Series((diff_values > 0.5).sum() / diff_values.size) if diff_values.size > 0 else np.nan


            feats["h_share_diff_>0.1"] = pd.Series((diff_values > 0.1).sum() / diff_values.sum()) if diff_values.sum() > 0 else np.nan
            feats["h_share_diff_>0.2"] = pd.Series((diff_values > 0.2).sum() / diff_values.sum()) if diff_values.sum() > 0 else np.nan
            feats["h_share_diff_>0.3"] = pd.Series((diff_values > 0.3).sum() / diff_values.sum()) if diff_values.sum() > 0 else np.nan
            feats["h_share_diff_>0.4"] = pd.Series((diff_values > 0.4).sum() / diff_values.sum()) if diff_values.sum() > 0 else np.nan
            feats["h_share_diff_>0.5"] = pd.Series((diff_values > 0.5).sum() / diff_values.sum()) if diff_values.sum() > 0 else np.nan


            # crate bins on df and create series which contains the relative counts per bin
            bins = 100
            df["bin_nr"], bin_array = pd.cut(df[self.key_consumption], labels=np.linspace(1, bins, num=bins).astype(int), bins=bins, retbins=True) # create bins and add them to df
            df["bin_intervals"] = pd.cut(df[self.key_consumption], bins=bins) #do same as before but save the intervals
            rel_bin_count = df["bin_nr"].value_counts().sort_index() #count relative occurance and sort that bins go from 1 to 100
            rel_bin_count = rel_bin_count/df[self.key_consumption].count() #calculate the relative amount of readings per bin
            
            # drop empty bins and reset index
            rel_bin_count = rel_bin_count[rel_bin_count != 0].reset_index(drop=True)

            # save index of bin with most readings
            if len(rel_bin_count[10:]) > 0: # only look after index 10 because before is unrealistic
                idxmax = rel_bin_count[10:].idxmax()
            elif len(rel_bin_count) > 0: # take all bins into account when 10 are not enough because all empty
                idxmax = rel_bin_count.idxmax()
            else: # all bins are empty
                idxmax = None

            if idxmax is not None: 
                #save the realtive frequency in the biggest bin and the 4 bins around it
                # if the bin is max bin is too close to either side of the histogram, going 4 bins left or right won't work. In this case the frequency is set to zero and will not count towards the addition
                freq_at_max = rel_bin_count.loc[idxmax]
                freq_at_max_add1 = rel_bin_count.loc[idxmax+1] if idxmax+1 < len(rel_bin_count) else 0
                freq_at_max_add2 = rel_bin_count.loc[idxmax+2] if idxmax+2 < len(rel_bin_count) else 0
                freq_at_max_add3 = rel_bin_count.loc[idxmax+3] if idxmax+3 < len(rel_bin_count) else 0
                freq_at_max_add4 = rel_bin_count.loc[idxmax+4] if idxmax+4 < len(rel_bin_count) else 0
                
                freq_at_max_sub1 = rel_bin_count.loc[idxmax-1] if idxmax-1 >= 0 else 0
                freq_at_max_sub2 = rel_bin_count.loc[idxmax-2] if idxmax-2 >= 0 else 0
                freq_at_max_sub3 = rel_bin_count.loc[idxmax-3] if idxmax-3 >= 0 else 0
                freq_at_max_sub4 = rel_bin_count.loc[idxmax-4] if idxmax-4 >= 0 else 0

                # add relative frequency in bins around biggest bin
                feats["h_share_at_max_bin"] = pd.Series(freq_at_max)
                feats["h_share_at_max_plus2_bin"] = pd.Series(freq_at_max + freq_at_max_add1 + freq_at_max_sub1)
                feats["h_share_at_max_plus4_bin"] = pd.Series(freq_at_max + freq_at_max_add1 + freq_at_max_sub1 + freq_at_max_add2 + freq_at_max_sub2)
                feats["h_share_at_max_plus6_bin"] = pd.Series(freq_at_max + freq_at_max_add1 + freq_at_max_sub1 + freq_at_max_add2 + freq_at_max_sub2 + freq_at_max_add3 + freq_at_max_sub3)

                # create feature for peakiness around biggest bin
                feats["h_slope_max_to_1"] = pd.Series(abs(freq_at_max - (freq_at_max_add1 + freq_at_max_sub1) / 2))
                feats["h_slope_1_to_2"] = pd.Series(abs((freq_at_max_add1 + freq_at_max_sub1) / 2 - (freq_at_max_add2 + freq_at_max_sub2) / 2))
                feats["h_slope_2_to_3"] = pd.Series(abs((freq_at_max_add2 + freq_at_max_sub2) / 2 - (freq_at_max_add3 + freq_at_max_sub3) / 2))
                feats["h_slope_3_to_4"] = pd.Series(abs((freq_at_max_add3 + freq_at_max_sub3) / 2 - (freq_at_max_add4 + freq_at_max_sub4) / 2))
                feats["h_slope_mean"] = pd.Series((feats["h_slope_max_to_1"] + feats["h_slope_1_to_2"] + feats["h_slope_2_to_3"] + feats["h_slope_3_to_4"]) / 4)

                # calculate how many bins are above a certain percentage level
                feats["h_num_bins_>1%"] = pd.Series((rel_bin_count > 0.01).sum()/bins)
                feats["h_num_bins_>2%"] = pd.Series((rel_bin_count > 0.02).sum()/bins)
                feats["h_num_bins_>3%"] = pd.Series((rel_bin_count > 0.03).sum()/bins)
                feats["h_num_bins_>4%"] = pd.Series((rel_bin_count > 0.04).sum()/bins)
                feats["h_num_bins_>5%"] = pd.Series((rel_bin_count > 0.05).sum()/bins)
                feats["h_num_bins_>6%"] = pd.Series((rel_bin_count > 0.06).sum()/bins)
                feats["h_num_bins_>7%"] = pd.Series((rel_bin_count > 0.07).sum()/bins)

                # calculate features based on window, around center with std
                center = bin_array[idxmax]
                std = df[self.key_consumption].std()

                df_window = df[(df[self.key_consumption] > (center - std)) & (df[self.key_consumption] < (center + std))]
                
                feats["h_centered_window_std"] = pd.Series(df_window[self.key_consumption].std())
                feats["h_centered_window_skew"] = pd.Series(df_window[self.key_consumption].skew())
                feats["h_centered_window_kurtosis"] = pd.Series(df_window[self.key_consumption].kurtosis())

                # within window, check how many readings are above or below mode (left or right)
                window_mode = df_window[self.key_consumption].mode()
                if len(window_mode) == 0: 
                    feats["h_centered_window_share_left_of_mode"] = np.nan
                    feats["h_centered_window_share_right_of_mode"] = np.nan
                else: 
                    window_mode = window_mode[0] # getting the actual value of the series
                    num_without_mode = (df_window[self.key_consumption] != window_mode).sum()
                    feats["h_centered_window_share_left_of_mode"] = pd.Series((df_window[self.key_consumption] < window_mode).sum() / num_without_mode)
                    feats["h_centered_window_share_right_of_mode"] = pd.Series((df_window[self.key_consumption] > window_mode).sum() / num_without_mode)
            
            else: # only empty bins handling
                cols = ['h_share_at_max_bin', 'h_share_at_max_plus2_bin', 'h_share_at_max_plus4_bin', 'h_share_at_max_plus6_bin', 'h_slope_max_to_1', 'h_slope_1_to_2', 'h_slope_2_to_3', 'h_slope_3_to_4', 'h_slope_mean', 
                'h_num_bins_>1%', 'h_num_bins_>2%', 'h_num_bins_>3%', 'h_num_bins_>4%', 'h_num_bins_>5%', 'h_num_bins_>6%', 'h_num_bins_>7%', 'h_centered_window_std', 'h_centered_window_skew', 'h_centered_window_kurtosis', 'h_centered_window_share_left_of_mode',
                'h_centered_window_share_right_of_mode']
                for c in cols: 
                    feats[c] = pd.Series(np.nan)

            ##########
            # feature for  encoding mean run time per cylcus in reading blocks
            ##########

            df = df.reset_index(drop=True) # reset index to start at 0 because we use it for the calculation below
            # determine the mean run time per cylce (counted as number of readings (e.g. 15min blocks) between NAs)
            df["start"] = np.where(df[self.key_consumption].notna() & df[self.key_consumption].shift(periods=1, axis="index").isna(), "start", "") #add start locations where  value notnull but value before isnull
            df["end"] = np.where(df[self.key_consumption].notna() & df[self.key_consumption].shift(periods=-1, axis="index").isna(), "end", "") #add end locations where  value notnull but value after isnull

            # create a list of indices for the start and the ends
            df_slot_length = pd.DataFrame()
            start_list= pd.Series(df.index[df['start'] == "start"]).tolist()
            end_list = pd.Series(df.index[df['end'] == "end"]).tolist()

            # if we are missing one end because its in the next year, we add the end at the last entry of the df.
            if len(start_list) > len(end_list):
                end_list.append((df[self.key_consumption].size-1))

            # add stuff to a df and calculate the difference
            df_slot_length["start"] = start_list
            df_slot_length["end"] = end_list
            df_slot_length["difference"] = (df_slot_length['end'] - df_slot_length['start']) + 1

            #calculate the mean
            feats["h_mean_readings_per_cycle"] = df_slot_length['difference'].mean()
        
            ##########
            # feature for encoding number of "core" unique values relative to max value
            ##########

            # removing start and end readings
            df[self.key_consumption] = df[self.key_consumption].where(df['start'] != "start")
            df[self.key_consumption] = df[self.key_consumption].where(df['end'] != "end")

            # round for comparability
            df[self.key_consumption] = df[self.key_consumption].round(2)

            # count the unque values
            feats["h_unique_vals_relto_max"] = pd.Series(df[self.key_consumption].value_counts().size / df[self.key_consumption].max()) if df[self.key_consumption].max() > 0 else np.nan
        
        return feats


    def create_features(self, df:pd.DataFrame, time_horizons:list, num_processes=1):
        '''
            Creates features for a single metering code, based on a a given time horizons and categories.
            NOTE: this method assumes that df contains only consumption values of a single metering code and does not assert this assumption.
            Args: 
                df: data frame to be processed (needs to include columns that were defines as key_timestamp)
                time_horizon: list of strings to define which time ranges of features should be calculated - strings within the list can be one of the following: daily, weekly, monthly, yearly, flexible
                num_processes: number of concurrently running jobs, i.e. parameter for parallelization --> None means all available CPUs are used
            Returns: 
                data frame with calculated features for one metering code
        '''
    
        for time_horizon in time_horizons: 
            assert isinstance(time_horizon, str), 'FeatureGenerator.create_features(): Type of parameter time_horizons needs to be a list of strings, but given type of value in list was {}.'.format(type(time_horizon))
            assert time_horizon in ['daily', 'weekly', 'monthly', 'yearly', 'flexible'], 'FeatureGenerator.create_features(): Values in parameter time_horizons can only be one of the following strings: daily, weekly, monthly, yearly, flexible. But given was: {}'.format(time_horizon)
        
        # for the case of doubled entries
        time_horizons = list(set(time_horizons))

        if not self.__check_data_frame_preprocessed__(df): 
            df = self.__preprocess_data_frame__(df)

        # create a data frame for all features
        df_features = create_time_horizons_dataframe(df, self.key_timestamp, time_horizons)
        
        # from here on: handle the routine of calculating the features in a parallelized manner
        
        # define the number of CPUs to be used 
        if num_processes is None: 
            num_processes = min(len(df_features), cpu_count())
        else: 
            assert isinstance(num_processes, int), 'FeatureGenerator.create_features(): Parameter num_processes must be of type int or None.'
        
        # create an instance for parallel processing 
        with Pool(num_processes) as pool: 

            # get the sequence to iterate over 
            rows = [row for idx, row in df_features.iterrows()]

            # get the returned rows as pandas series
            results_list = pool.map(partial(self.__calculate_features__, df=df), rows)

            # concatenate the rows again to one data frame 
            df_features = pd.concat(results_list, axis=0)
        
        # sort elements
        df_features.sort_values(by=['time_horizon', 'time_start', 'time_end'], inplace=True)
        df_features.reset_index(drop=True, inplace=True)
        return df_features


    def set_key_timestamp(self, key_timestamp): 
        '''
            Sets parameter key_time_stamp. 
            Args: 
                key_time_stamp: string to define the key_time_stamp parameter 
        '''
        assert isinstance(key_timestamp, str) or key_timestamp is None, 'Visualizer: Parameter key_timestamp must be a string representation of the column name that encodes the timestamp of data frames to be processed or None.'
        self.key_timestamp = key_timestamp

    def get_key_timestamp(self): 
        '''
            Returns string of current key_timestamp parameter. 
        '''
        return self.key_timestamp

    def set_key_consumption(self, key_consumption): 
        '''
            Sets parameter key_consumption. 
            Args: 
                key_consumption: string to define the key_consumption parameter 
        '''
        assert isinstance(key_consumption, str) or key_consumption is None, 'Visualizer: Parameter key_conumption must be a string representation of the column name that encodes the kWh-consumption of data frames to be processed or None.'
        self.key_consumption = key_consumption
    
    def get_key_consumption(self): 
        '''
            Returns string of current key_consumption parameter. 
        '''
        return self.key_consumption