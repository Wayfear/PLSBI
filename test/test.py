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

warnings.filterwarnings("ignore")


class PLSForBrainImaging:

    def __init__(self, component_range=(1, 10), scale=True,
                 max_iter=500, tol=1e-06,
                 correlation_threshold=0.28, edge_selection_threshold=1.96,
                 output_path='PLS_result/', repeat_time=1000) -> None:
        self.output_path = Path(output_path)
        self.output_path.mkdir(parents=True, exist_ok=True)
        self.repeat_time = repeat_time
        self.component_range = component_range
        self.correlation_threshold = correlation_threshold
        self.scale = scale
        self.edge_selection_threshold = edge_selection_threshold
        self.max_iter = max_iter
        self.tol = tol
        self.x_loading = None
        self.y_loading = None

    def optimise_pls_cv(self, X, y, n_comp, name, plot_components=True):
        '''Run PLS including a variable number of components, up to n_comp,
        and calculate MSE '''

        mse = []
        component = np.arange(*n_comp)

        for i in component:
            pls = PLSRegression(
                n_components=i, max_iter=self.max_iter, tol=self.tol)

            # Cross-validation
            y_cv = cross_val_predict(pls, X, y, cv=5)

            mse.append(r2_score(y, y_cv))

        # Calculate and print the position of minimum in MSE
        msemin = np.argmax(mse)
        print("Suggested number of components: ", msemin+1)

        if plot_components is True:
            with plt.style.context(('ggplot')):
                plt.plot(component, np.array(mse),
                         '-v', color='blue', mfc='blue')
                plt.plot(component[msemin], np.array(mse)
                         [msemin], 'P', ms=10, mfc='red')
                plt.xlabel('Number of PLS components')
                plt.ylabel('R-Squared')
                plt.xlim(left=-1)

            plt.tight_layout()

            plt.savefig(name)

        # Define PLS object with optimal number of components
        pls_opt = PLSRegression(n_components=msemin+1,
                                max_iter=self.max_iter, tol=self.tol)

        # Fir to the entire dataset
        pls_opt.fit(X, y)
        y_c = pls_opt.predict(X)

        # Cross-validation
        y_cv = cross_val_predict(pls_opt, X, y, cv=10)

        # Calculate scores for calibration and cross-validation
        score_c = r2_score(y, y_c)
        score_cv = r2_score(y, y_cv)

        # Calculate mean squared error for calibration and cross validation
        mse_c = mean_squared_error(y, y_c)
        mse_cv = mean_squared_error(y, y_cv)

        print("When using the best component:")
        print('\tR2 calib: %5.3f' % score_c)
        print('\tR2 CV: %5.3f' % score_cv)
        print('\tMSE calib: %5.3f' % mse_c)
        print('\tMSE CV: %5.3f' % mse_cv)

        return msemin+1

    def get_repeat_result(self, X, y, comp=5, times=1000):
        r_sqs = []
        x_loading = []
        y_loading = []
        plr = PLSRegression(n_components=comp,
                            max_iter=self.max_iter, tol=self.tol)

        for i in range(times):
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.8, random_state=i)

            plr.fit(X_train, y_train)
            score = plr.score(X_test, y_test)
            r_sqs.append(score)
            x_loading.append(plr.x_loadings_)
            y_loading.append(plr.y_loadings_)

        x_loading = np.array(x_loading)
        y_loading = np.array(y_loading)
        r_sq = np.mean(np.array(r_sqs))

        return r_sq, x_loading, y_loading

    def get_k_folder_result(self, X, y, comp):
        accs = []

        for i in range(200):
            kf = KFold(n_splits=5, shuffle=True, random_state=i)
            for train_index, test_index in kf.split(X):
                clf = PLSRegression(n_components=comp,
                                    max_iter=self.max_iter, tol=self.tol)
                X_train, X_test = X[train_index], X[test_index]
                y_train, y_test = y[train_index], y[test_index]
                clf.fit(X_train, y_train)
                score = clf.score(X_test, y_test)
                accs.append(score)

        return np.mean(np.array(accs))

    def generate_edge_matrix_by_p_value(self, edge_indexs, x_loading, all_node_num, threshold, abs_first=True):

        data = x_loading

        thou_time_mean_data = np.mean(data, axis=0)

        _, _, comp_num = data.shape

        zs_scores = []

        p_val = []

        print("X Loading")

        for i in range(comp_num):

            if not abs_first:

                zs_score = stats.zscore(
                    np.mean(data[:, :, i], axis=0))

                zs_scores.append(np.abs(zs_score) > threshold)

            else:

                zs_score = stats.zscore(
                    np.mean(np.abs(data[:, :, i]), axis=0))

                zs_scores.append(zs_score > threshold)

            counter = np.sum(np.abs(zs_score) > threshold)

            p_val.append(zs_score)

            print(
                f"Component{i+1}, threshold: {threshold}, edge num: {counter}")

        p_value_selected_edge = np.array(zs_scores)

        p_val = np.array(p_val)

        component, _ = p_value_selected_edge.shape

        for i in range(component):
            edge_matrix = np.zeros((all_node_num, all_node_num))

            index = np.where(p_value_selected_edge[i, :] > 0)[0]
            for j in index:
                edge_matrix[edge_indexs[0][j], edge_indexs[1]
                            [j]] = thou_time_mean_data[j, i]

            np.savetxt(
                self.output_path/f'original_{i}_by_rank.edge', edge_matrix,  fmt='%1.5f')

            print(
                f"The result of Component{i+1} saved in f{self.output_path/f'original_{i}_by_rank.edge'}")
        edges = []
        for i in range(component):
            edges_list = []
            edge_matrix = np.zeros((all_node_num, all_node_num))

            index = np.where(p_value_selected_edge[i, :] > 0)[0]
            for j in index:
                edge_matrix[edge_indexs[0][j], edge_indexs[1][j]] = p_val[i, j]
                edges_list.append(
                    (edge_indexs[0][j], edge_indexs[1][j], p_val[i, j]))
            np.savetxt(
                self.output_path/f'{i}_pvalue.edge', edge_matrix,  fmt='%1.3f')
            edges.append(edges_list)

        return edges

    def fit(self, X, y):

        length = X.shape[1]

        # obtain imaging for each subject
        imaging_arr = np.empty((0, int((length*length-length)/2)))

        for i, f in enumerate(X):
            flat = f[np.triu_indices(length, 1)].reshape((1, -1))
            imaging_arr = np.append(imaging_arr, flat, axis=0)

        X = imaging_arr

        if self.scale:
            scaler = StandardScaler()
            y = scaler.fit_transform(y)
            X = scaler.fit_transform(X)

        X[np.isnan(X)] = 0
        y[np.isnan(y)] = 0

        # using correlation the select part of edges
        corr = np.zeros((X.shape[1], y.shape[1]))
        for i in range(X.shape[1]):
            for j in range(y.shape[1]):
                t = np.corrcoef(X[:, i], y[:, j])
                corr[i, j] = t[0, 1]

        correlation = np.max(np.abs(corr), axis=1)

        selected_edge = correlation > self.correlation_threshold

        # get selected edges in y and store selected edge index
        X = X[:, selected_edge]

        triu_index = np.triu_indices(length, 1)
        triu_index = [index[selected_edge] for index in triu_index]

        suggest_com_num = self.optimise_pls_cv(
            X, y, self.component_range, self.output_path/'suggest_com_num.png')

        _, self.x_loading, _ = self.get_repeat_result(
            X, y, comp=suggest_com_num, times=self.repeat_time)

        print('\n')
        # generate and save edge matrices
        self.generate_edge_matrix_by_p_value(
            triu_index, self.x_loading, length, self.edge_selection_threshold)

        # print R^2 and y loading, R^2 based on k folders,
        # y loading from the whole dataset
        r_sq = self.get_k_folder_result(X, y, comp=suggest_com_num)

        print(f'\nPerformance \nR2: {r_sq:.4f}\n')

        plr = PLSRegression(n_components=suggest_com_num,
                            max_iter=self.max_iter, tol=self.tol)

        plr.fit(X, y)

        self.y_loading = plr.y_loadings_

        print('Y Loading\n')
        header = [f'Component{j+1}' for j in range(suggest_com_num)]
        index = [f'Variable{i+1}' for i in range(y.shape[1])]

        df = pd.DataFrame(plr.y_loadings_, columns=header, index=index)

        print(df)


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
