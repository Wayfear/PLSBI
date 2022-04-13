from typing import Counter
import numpy as np
import rpy2.robjects as robjects
from rpy2.robjects import pandas2ri
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import StandardScaler
import pandas as pd
from sklearn.cross_decomposition import PLSRegression
import argparse
from scipy import stats
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_predict, KFold
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import warnings
from PLSBI import PLSForBrainImaging
warnings.filterwarnings("ignore")


def main(args):
    output = Path(args.output)
    image_path = args.imaging
    clinical_table_path = args.clinical_file
    correlation_thres = args.correlation_threshold
    id_file = args.id_file

    category = {'INTRUSIVE': [1, 2, 3, 4, 17], 'AVOIDANCE': [5, 6, 7], 'NEGATIVE AFFECT': [
        8, 9, 10, 11], "HYERAROUSAL": [12, 13, 14, 15, 16]}

    clinical_column = ['sid', 'ptsdss1_categorical', 'ptsdss2_categorical', 'ptsdss3_categorical', 'ptsdss4_categorical', 'ptsdss5_categorical', 'ptsdss6_categorical', 'ptsdss7_categorical', 'ptsdss8_categorical',
                       'ptsdss9_categorical', 'ptsdss10_categorical', 'ptsdss11_categorical', 'ptsdss12_categorical', 'ptsdss13_categorical', 'ptsdss14_categorical', 'ptsdss15_categorical', 'ptsdss16_categorical', 'ptsdss17_categorical']

    # load imaging data
    # 'FC', 'subjid', 'module_memberships'
    pandas2ri.activate()

    robjects.r['load'](id_file)
    subids = list(robjects.r['subjid'])

    robjects.r['load'](image_path)
    fc = np.array(robjects.r['FC_impute'])
    module_memberships = list(robjects.r['module_memberships_impute'])

    length = fc.shape[0]
    new_fc = np.transpose(fc, (2, 0, 1))

    subids = [int(i) for i in subids]

    imaging_dic = {}
    for i, f in enumerate(new_fc):

        imaging_dic[subids[i]] = f

    imaging_df = pd.DataFrame(data={'id': subids})

    # load clinical variables
    clinical_file = Path(clinical_table_path)
    pss_merged = pd.read_csv(clinical_file)

    # print("Used clinical variables")
    # for c in clinical_column:
    #     print(f'- {c}')

    # process clinical variables
    data = pd.merge(pss_merged[clinical_column],
                    imaging_df, left_on='sid', right_on='id')
    data = data.replace(r'^\s*$', np.nan, regex=True)
    data = data.replace(r'#NULL!', np.nan, regex=True)

    # convert str to float
    for col in data.columns:
        data[col] = pd.to_numeric(data[col], errors='raise')

    clinical_column = ['sid'] + list(category.keys())

    # calculate the sum score of each pss category
    for k, v in category.items():
        used_col = [f'ptsdss{i}_categorical' for i in v]
        data[k] = data[used_col].sum(axis=1)
    data = data[clinical_column]

    origin_data = data[clinical_column]

    data_table = origin_data.dropna()

    id_list = data_table['sid']

    # obtain each pss score
    X = data_table[clinical_column[1:]].to_numpy()

    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # obtain imaging for each subject
    imaging_y = np.empty((0, length, length))

    for i in id_list:
        imaging_y = np.append(imaging_y, np.expand_dims(
            imaging_dic[i], axis=0), axis=0)

    pls = PLSForBrainImaging()
    pls.fit(imaging_y, X)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', default='result', type=str,
                        help='The folder for storing results.')
    parser.add_argument('--imaging', default='data/Resting_State_FMRI_IMPUTE(2).RData', type=str,
                        help='The file contains imaging data, the format is RData')
    parser.add_argument('--clinical_file', default='data/GTP MRI dataset 2016.csv', type=str,
                        help='The file contains clinical variables, the format is csv')
    parser.add_argument('--id_file', default='data/allPTSD.RData', type=str,
                        help='The file contains subject id , the format is RData')
    parser.add_argument('--correlation_threshold', default=0.28, type=float,
                        help='The threshold used to select correlated edges')

    args = parser.parse_args()

    main(args)
