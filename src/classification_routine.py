import os
import numpy as np 
import pandas as pd 
import pickle
from feature_generation import *

if __name__ == '__main__':

    # -------------------------------------------------------------------------
    # Load smart meter data of a single metering code
    # -------------------------------------------------------------------------

    # load smart meter data (CSV example)
    folder = os.getcwd() + '/../data/'
    metering_code = 'smd_example'
    df = pd.read_csv(folder+metering_code+'.csv')

    # -------------------------------------------------------------------------
    # Parameter definition
    # -------------------------------------------------------------------------

    time_horizon = 'weekly' # do you use weekly, monthly, or yearly data?
    sm_measures_only_hp = False # does the smart meter measure the heat pump separately from other appliances? - NOTE: automate this depending the smart meter data loaded
    model_folder = os.getcwd() + '/../models/' # where is the top level folder that contains the models?
    key_timestamp = 'timestamp' # column identifier of the time stamp in the data frame containing the smart meter data 
    key_consumption = 'kWh' # column identifier of the energy consumption in the data frame containing the smart meter data 
    num_processes = None # parameter for parallelization of feature calculation - e.g., None means all available CPUs are used and 2 means 2 CPUs are used

    # -------------------------------------------------------------------------
    # Select the right model 
    # -------------------------------------------------------------------------

    # choice of model depends on the settings above 
    model_subfolder = model_folder + 'TimeHorizon-{}_SeparateSM-{}/'.format(time_horizon, sm_measures_only_hp)

    # load the classifier object for label predictions
    classifier = None 
    with open(model_subfolder+'classifier.pkl', 'rb') as file:
        classifier = pickle.load(file)

    # load the scaler object for feature normalization
    scaler = None
    with open(model_subfolder+'scaler.pkl', 'rb') as file:
        scaler = pickle.load(file)

    # -------------------------------------------------------------------------
    # Generate features 
    # -------------------------------------------------------------------------

    # create feature features 
    fg = FeatureGeneration(key_timestamp, key_consumption)
    df_feats = fg.create_features(df, [time_horizon], num_processes=None)

    # -------------------------------------------------------------------------
    # Choose the right observations and prepare the features for classification
    # -------------------------------------------------------------------------
    
    # NOTE: only consider observations with 85% completeness (at least this was how model was trained)
    # means that 85% of the readings in a window must be non-zero (zero or NAN considered to mean no activity or missing data)
    required_completeness = 0.85 
    df_feats = df_feats[df_feats['share_non_zero_vals'] >= required_completeness]
    
    # NOTE: the model for weekly and monthly observations was trained with only considering winter months  - for the yearly model all months are used
    # further the weekly and monthly model uses less features 
    if time_horizon == 'weekly': 
        allowed_months = [10, 11, 12, 1, 2]
        df_feats = df_feats[df_feats['month'].isin(allowed_months)]

        # define the feature columns to be used in the same order as the trained model expects it
        include_cols = ['s_min', 's_max', 's_median', 's_mode', 's_std', 's_skew',
        's_kurtosis', 's_var', 's_num_peaks', 's_diff_mean', 's_diff_sum',
        'r_mean_max', 'r_mean_min', 'r_mean_max_no_min',
        't_num_above_0.125kWh', 't_num_above_0.25kWh',
        't_num_above_0.5kWh', 't_num_above_mean', 't_num_above_min',
        't_num_is_max', 'h_sum_norm_diff_>0.1', 'h_sum_norm_diff_>0.2',
        'h_sum_norm_diff_>0.3', 'h_sum_norm_diff_>0.4',
        'h_sum_norm_diff_>0.5', 'h_share_diff_>0.1', 'h_share_diff_>0.2',
        'h_share_diff_>0.3', 'h_share_diff_>0.4', 'h_share_diff_>0.5',
        'h_share_at_max_bin', 'h_share_at_max_plus2_bin',
        'h_share_at_max_plus4_bin', 'h_share_at_max_plus6_bin',
        'h_slope_max_to_1', 'h_slope_1_to_2', 'h_slope_2_to_3',
        'h_slope_3_to_4', 'h_slope_mean', 'h_num_bins_>1%',
        'h_num_bins_>2%', 'h_num_bins_>3%', 'h_num_bins_>4%',
        'h_num_bins_>5%', 'h_num_bins_>6%', 'h_num_bins_>7%',
        'h_centered_window_std', 'h_centered_window_skew',
        'h_centered_window_kurtosis',
        'h_centered_window_share_left_of_mode',
        'h_centered_window_share_right_of_mode',
        'h_mean_readings_per_cycle', 'h_unique_vals_relto_max']

    elif time_horizon == 'monthly': 
        allowed_months = [10, 11, 12, 1, 2]
        df_feats = df_feats[df_feats['month'].isin(allowed_months)]

        # define the feature columns to be used in the same order as the trained model expects it
        include_cols = ['s_min', 's_max', 's_median', 's_mode', 's_std', 's_skew',
        's_kurtosis', 's_var', 's_num_peaks', 's_diff_mean', 's_diff_sum',
        'r_mean_max', 'r_mean_min', 'r_mean_max_no_min',
        't_num_above_0.125kWh', 't_num_above_0.25kWh',
        't_num_above_0.5kWh', 't_num_above_mean', 't_num_above_min',
        't_num_is_max', 'h_sum_norm_diff_>0.1', 'h_sum_norm_diff_>0.2',
        'h_sum_norm_diff_>0.3', 'h_sum_norm_diff_>0.4',
        'h_sum_norm_diff_>0.5', 'h_share_diff_>0.1', 'h_share_diff_>0.2',
        'h_share_diff_>0.3', 'h_share_diff_>0.4', 'h_share_diff_>0.5',
        'h_share_at_max_bin', 'h_share_at_max_plus2_bin',
        'h_share_at_max_plus4_bin', 'h_share_at_max_plus6_bin',
        'h_slope_max_to_1', 'h_slope_1_to_2', 'h_slope_2_to_3',
        'h_slope_3_to_4', 'h_slope_mean', 'h_num_bins_>1%',
        'h_num_bins_>2%', 'h_num_bins_>3%', 'h_num_bins_>4%',
        'h_num_bins_>5%', 'h_num_bins_>6%', 'h_num_bins_>7%',
        'h_centered_window_std', 'h_centered_window_skew',
        'h_centered_window_kurtosis',
        'h_centered_window_share_left_of_mode',
        'h_centered_window_share_right_of_mode',
        'h_mean_readings_per_cycle', 'h_unique_vals_relto_max']

    else: # means yearly data 
        include_cols = ['s_min', 's_max', 's_winter_max', 's_winter_day_max',
        's_winter_night_max', 's_winter_min', 's_winter_day_min',
        's_winter_night_min', 's_median', 's_winter_median',
        's_winter_day_median', 's_winter_night_median', 's_mode',
        's_winter_mode', 's_winter_day_mode', 's_winter_night_mode',
        's_std', 's_winter_std', 's_winter_day_std', 's_winter_night_std',
        's_skew', 's_winter_skew', 's_winter_day_skew',
        's_winter_night_skew', 's_kurtosis', 's_winter_kurtosis',
        's_winter_day_kurtosis', 's_winter_night_kurtosis', 's_var',
        's_winter_var', 's_winter_day_var', 's_winter_night_var',
        's_num_peaks', 's_diff_mean', 's_diff_sum', 'r_mean_max',
        'r_mean_min', 'r_mean_max_no_min', 't_num_above_0.125kWh',
        't_num_above_0.25kWh', 't_num_above_0.5kWh', 't_num_above_mean',
        't_num_above_min', 't_num_is_max', 'h_sum_norm_diff_>0.1',
        'h_sum_norm_diff_>0.2', 'h_sum_norm_diff_>0.3',
        'h_sum_norm_diff_>0.4', 'h_sum_norm_diff_>0.5',
        'h_share_diff_>0.1', 'h_share_diff_>0.2', 'h_share_diff_>0.3',
        'h_share_diff_>0.4', 'h_share_diff_>0.5', 'h_share_at_max_bin',
        'h_share_at_max_plus2_bin', 'h_share_at_max_plus4_bin',
        'h_share_at_max_plus6_bin', 'h_slope_max_to_1', 'h_slope_1_to_2',
        'h_slope_2_to_3', 'h_slope_3_to_4', 'h_slope_mean',
        'h_num_bins_>2%', 'h_num_bins_>3%', 'h_num_bins_>4%',
        'h_num_bins_>5%', 'h_num_bins_>6%', 'h_num_bins_>7%',
        'h_centered_window_std', 'h_centered_window_skew',
        'h_centered_window_kurtosis',
        'h_centered_window_share_left_of_mode',
        'h_centered_window_share_right_of_mode',
        'h_mean_readings_per_cycle', 'h_unique_vals_relto_max']
        
    if len(df_feats) == 0: 
        print('ERROR: INVALID OBSERVATIONS - CLASSIFICATION NOT POSSIBLE!')
    
    else: 
        # select the feature columns to be scaled and only affect these 
        np_features =  df_feats[include_cols].values # transformation to numpy and removing columns that are meta data or unused features 
        np_features = scaler.transform(np_features)
        np_features = np.nan_to_num(np_features)

        # -------------------------------------------------------------------------
        # Perform classification
        # -------------------------------------------------------------------------

        # get the predicted label for each observation 
        np_predictions = classifier.predict(np_features)

        # NOTE: label 0 = modulating HP (variable speed HP), label 1 = non-modulating HP (fixed speed HP)
        label_dict = {0 : 'Variable Speed Heat Pump', 1 : 'Fixed Speed Heat Pump'}

        # get the distribution of the predictions per observation 
        np_uniques, np_counts = np.unique(np_predictions, return_counts=True)
        np_shares = np.round(np_counts / len(np_predictions), decimals=3)

        # get the final prediction 
        majority_idx = np.argmax(np_shares) # get the index of the majority prediction
        predicted_label = np_uniques[majority_idx]
        predicted_class_name = label_dict[predicted_label]
        predicted_probability = np_shares[majority_idx]

        print('Prediction: {}'.format(predicted_class_name))
        print('Probability: {}'.format(predicted_probability))
        print('Number of Observations: {}'.format(len(np_predictions)))
        print('Type of Observations: {}'.format(time_horizon))