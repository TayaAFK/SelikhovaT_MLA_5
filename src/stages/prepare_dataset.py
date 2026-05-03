from sklearn.preprocessing import StandardScaler, OrdinalEncoder, PowerTransformer
import pandas as pd
import yaml
import sys
import os

sys.path.append(os.getcwd())
from src.loggers import get_logger

def load_config(config_path):
    with open(config_path) as conf_file:
        config = yaml.safe_load(conf_file)
    return config

def clear_data(path2data):
    df = pd.read_csv(path2data)
    
    cat_columns = ['Gender', 'Occupation', 'BMI Category', 'Sleep Disorder']
    num_columns = [
        'Age', 'Sleep Duration', 'Quality of Sleep', 
        'Physical Activity Level', 'Stress Level', 'Heart Rate', 'Daily Steps'
    ]
    
    if 'Person ID' in df.columns:
        df = df.drop(columns=['Person ID'])
    df['Sleep Disorder'] = df['Sleep Disorder'].fillna('None')
    df = df[(df['Age'] > 0) & (df['Age'] < 110)]
    df = df[df['Daily Steps'] >= 0]
    df = df.reset_index(drop=True)
    ordinal = OrdinalEncoder()
    df[cat_columns] = ordinal.fit_transform(df[cat_columns])
    
    return df

def scale_frame(frame, target_column):
    df = frame.copy()
    X = df.drop(columns=[target_column])
    y = df[target_column]
    scaler = StandardScaler()
    power_trans = PowerTransformer()
    X_scale = scaler.fit_transform(X.values)
    return X_scale, y, power_trans

def featurize(dframe, config) -> None:
    logger = get_logger('FEATURIZE')
    logger.info('Create features for Sleep Health dataset')
    if 'Blood Pressure' in dframe.columns:
        bp_split = dframe['Blood Pressure'].str.split('/', expand=True).astype(int)
        dframe['Systolic_BP'] = bp_split[0]
        dframe['Diastolic_BP'] = bp_split[1]
        dframe = dframe.drop(columns=['Blood Pressure'])
    dframe['Activity_Efficiency'] = dframe['Daily Steps'] / dframe['Sleep Duration']
    features_path = config['featurize']['features_path']
    dframe.to_csv(features_path, index=False)
    logger.info(f'Final features saved to: {features_path}')

if __name__ == "__main__":
    config = load_config("./src/config.yaml")
    df_cleaned = clear_data(config['data_load']['dataset_csv'])
    featurize(df_cleaned, config)
