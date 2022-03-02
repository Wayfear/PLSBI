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
from util import get_repeat_result, optimise_pls_cv, generate_edge_matrix_by_p_value, get_k_folder_result
import warnings

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

    imaging_arr = np.empty((0, int((length*length-length)/2)))

    subids = [int(i) for i in subids]

    imaging_dic = {}
    for i, f in enumerate(new_fc):
        flat = f[np.triu_indices(length, 1)].reshape((1, -1))
        imaging_arr = np.append(imaging_arr, flat, axis=0)
        imaging_dic[subids[i]] = flat

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
    imaging_y = np.empty((0, int((length*length-length)/2)))

    for i in id_list:
        imaging_y = np.append(imaging_y, imaging_dic[i], axis=0)

    y = imaging_y

    # using correlation the select part of edges
    corr = np.zeros((X.shape[1], y.shape[1]))
    for i in range(X.shape[1]):
        for j in range(y.shape[1]):
            a = np.corrcoef(X[:, i], y[:, j])
            corr[i, j] = a[0, 1]

    correlation = np.max(np.abs(corr), axis=0)

    selected_edge = correlation > correlation_thres

    corr = corr[:, selected_edge]

    # count the number of positive and negative edges for analysis
    statics_count = Counter()
    for i in range(corr.shape[1]):
        c = corr[:, i]
        idx = np.argmax(np.abs(c))
        value = c[idx]
        if value > 0:
            statics_count['positve'] += 1
        elif value < 0:
            statics_count['negative'] += 1
    print("Positive and negative edge number: ", statics_count)
    print('\n')
    print("Selected_edge", np.sum(selected_edge))
    print('\n')

    # get selected edges in y and store selected edge index
    y = y[:, selected_edge]

    triu_index = np.triu_indices(length, 1)
    triu_index = [index[selected_edge] for index in triu_index]

    y = scaler.fit_transform(y)
    y[np.isnan(y)] = 0

    # exchange X, y
    tmp = y
    y = X
    X = tmp

    # obtain the optimise comp number
    suggest_com_num = optimise_pls_cv(X, y, 18, output/'suggest_com_num.png')
    print('\n')
    _, x_loading, _ = get_repeat_result(
        X, y, comp=suggest_com_num, times=1000)

    # generate and save edge matrices
    generate_edge_matrix_by_p_value(
        triu_index, x_loading, length, output)

    # print R^2 and y loading, R^2 based on k folders,
    # y loading from the whole dataset
    r_sq = get_k_folder_result(X, y, comp=suggest_com_num)

    print(f'R^2: {r_sq:.4f}\n')

    plr = PLSRegression(n_components=suggest_com_num)

    plr.fit(X, y)

    print('## Y Loading\n')
    header = [f'comp {j+1}' for j in range(suggest_com_num)]

    print(f"|column|{'|'.join(header)}|")
    print(f"|{'|'.join([':-:']*(len(header)+1))}|")
    for i in range(y.shape[1]):
        output = f"|{clinical_column[i+1]}|"
        for j in range(suggest_com_num):
            output += f'{plr.y_loadings_[i, j]:.3f}|'
        print(output)


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
