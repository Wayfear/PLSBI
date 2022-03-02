import numpy as np
from scipy import stats
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_predict, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.cross_decomposition import PLSRegression
import numpy
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score


def get_k_folder_result(X, y, comp):
    accs = []

    for i in range(200):
        kf = KFold(n_splits=5, shuffle=True, random_state=i)
        for train_index, test_index in kf.split(X):
            clf = PLSRegression(n_components=comp)
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]
            clf.fit(X_train, y_train)
            # y_scores = clf.predict(X_test)
            score = clf.score(X_test, y_test)
            accs.append(score)

    return np.mean(np.array(accs))


def df_to_markdown(df, float_format='%.2g'):
    """
    Export a pandas.DataFrame to markdown-formatted text.
    DataFrame should not contain any `|` characters.
    """
    from os import linesep
    return linesep.join([
        'index|'+'|'.join(df.columns),
        4 * '-'+'|'+'|'.join(4 * '-' for i in df.columns),
        df.to_csv(sep='|', header=False,
                  float_format=float_format)
    ]).replace('|', ' | ')


def draw_heatmap(node_modules, data, name):
    plt.clf()
    length = len(node_modules)
    index2name = dict(zip(range(length), node_modules))
    id_index = 0
    idx2regionidx = {}
    regionidx2region = {}
    index_list = ['Auditory', 'VentralAttention', 'Salience', 'ParietoMedial', 'DefaultMode', 'Visual', 'CinguloOpercular', 'FrontoParietal',
                  'Reward', 'VentralAttetion', 'DorsalAttention', 'MedialTemporalLobe',  'SomatomotorDorsal', 'SomatomotorLateral']

    diff = sum(map(lambda name: name == 'unassigned', node_modules))

    for group_name in index_list:
        for index, region_name in index2name.items():
            if region_name == group_name:
                idx2regionidx[index] = id_index
                regionidx2region[id_index] = region_name
                id_index += 1

    counter = {}
    for index in index_list:
        counter[index] = []
    for index, regionid in regionidx2region.items():
        if regionid in counter:
            counter[regionid].append(index)

    y_label_position = {}
    divide_line = []
    for id_index, region_index in counter.items():
        y_label_position[id_index] = (max(region_index)+min(region_index))/2
        divide_line.append(max(region_index))
    divide_line = divide_line[:-1]
    y_label = list(y_label_position.keys())
    y_position = list(y_label_position.values())

    m, n = data.shape
    rearrange_data = np.zeros((m-diff, n-diff))
    for i in range(m):
        for j in range(n):
            if i in idx2regionidx and j in idx2regionidx:
                rearrange_data[idx2regionidx[i], idx2regionidx[j]] = data[i, j]
    length = len(y_label_position)
    # temp_divide_line = [0] + divide_line + [m-diff]
    g = sns.heatmap(rearrange_data, cbar=True, cmap='coolwarm')

    # g = sns.heatmap(rearrange_data,  yticklabels=y_label[::-1],
    #                 xticklabels=y_label, annot=True, cbar=True, cmap='coolwarm')

    # g.set_title(title_names[0], fontdict={'fontsize': 30})
    g.hlines(divide_line, *g.get_xlim())
    g.vlines(divide_line, *g.get_ylim())
    g.set_yticks(y_position)

    g.set_yticklabels(y_label, fontsize=18)
    g.set_xticks(y_position)

    g.set_xticklabels(y_label, fontsize=18)
    plt.tight_layout()
    plt.savefig(f"distribution/heatmap{name}.png")


def draw_bar_figure(X, name, title):
    plt.clf()
    colors = {'INTRUSIVE': 'r', 'AVOIDANCE': 'b',
              'NEGATIVE AFFECT': 'g', "HYERAROUSAL": "purple"}
    cate_list = ['INTRUSIVE']*4+['AVOIDANCE'] * \
        3+['NEGATIVE AFFECT']*4+['HYERAROUSAL']*6
    names = [f'PTSDss{i}' for i in range(1, 18)]
    data = pd.DataFrame(data={"name": names, "class": cate_list, 'value': X})
    c = data['class'].apply(lambda x: colors[x])
    bars = plt.bar(data['name'], data['value'], color=c)
    labels = list(colors.keys())
    handles = [plt.Rectangle((0, 0), 1, 1, color=colors[label])
               for label in labels]
    plt.legend(handles, labels)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(name)


def save(X, y, name):
    header = ",".join([f'X{i}' for i in range(X.shape[1])] +
                      [f'y{i}' for i in range(y.shape[1])])
    with open(name, 'w') as f:
        f.write(f'{header}\n')
        data = np.concatenate((X, y), axis=1)
        numpy.savetxt(f, data, delimiter=",")


def generate_edge_matrix(edge_indexs, weight):
    total_edge_num, component = weight.shape
    selected_edge_num = total_edge_num//20

    for i in range(component):
        edge_matrix = np.zeros((total_edge_num, total_edge_num))
        w = weight[:, i]
        tmp = np.abs(w)
        index = tmp.argsort()[-selected_edge_num:]
        for j in index:
            edge_matrix[edge_indexs[0][j], edge_indexs[1][j]] = w[j] + 0.5
        np.savetxt(f'data/edge_matrix/{i}.edge', edge_matrix,  fmt='%1.4f')


def generate_edge_matrix_by_p_value(edge_indexs, x_loading, all_node_num, output_folder, abs_first=True):

    data = x_loading

    thou_time_mean_data = np.mean(data, axis=0)

    _, _, comp_num = data.shape

    zs_scores = []

    p_val = []

    th1 = 1.96

    for i in range(comp_num):

        if not abs_first:

            zs_score = stats.zscore(
                np.mean(data[:, :, i], axis=0))

            zs_scores.append(np.abs(zs_score) > th1)

        else:

            zs_score = stats.zscore(
                np.mean(np.abs(data[:, :, i]), axis=0))

            zs_scores.append(zs_score > th1)

        counter = np.sum(np.abs(zs_score) > th1)

        p_val.append(zs_score)

        print(f"Component{i}, threshold: {th1}, edge num: {counter}")

    p_value_selected_edge = np.array(zs_scores)

    p_val = np.array(p_val)

    component, total_edge_num = p_value_selected_edge.shape

    edges = []

    for i in range(component):
        edge_matrix = np.zeros((all_node_num, all_node_num))

        index = np.where(p_value_selected_edge[i, :] > 0)[0]
        for j in index:
            edge_matrix[edge_indexs[0][j], edge_indexs[1]
                        [j]] = thou_time_mean_data[j, i]

        np.savetxt(
            output_folder/f'original_{i}_by_rank.edge', edge_matrix,  fmt='%1.5f')

    for i in range(component):
        edges_list = []
        edge_matrix = np.zeros((all_node_num, all_node_num))

        index = np.where(p_value_selected_edge[i, :] > 0)[0]
        for j in index:
            edge_matrix[edge_indexs[0][j], edge_indexs[1][j]] = p_val[i, j]
            edges_list.append(
                (edge_indexs[0][j], edge_indexs[1][j], p_val[i, j]))
        np.savetxt(
            output_folder/f'{i}_pvalue.edge', edge_matrix,  fmt='%1.3f')
        edges.append(edges_list)

    return edges


def get_repeat_result(X, y, comp=5, times=1000):
    r_sqs = []
    x_loading = []
    y_loading = []
    plr = PLSRegression(n_components=comp)

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

    # np.save(output_file, {'acc': acc, 'x_loading': x_loading,
    #                       'y_loading': y_loading}, allow_pickle=True)

    return r_sq, x_loading, y_loading


def get_data_with_filling(X, y, use_scaler=True):
    mask = ~np.isnan(y)
    X, y = X[mask], y[mask]
    col_mean = np.nanmean(X, axis=0)
    inds = np.where(np.isnan(X))
    X[inds] = np.take(col_mean, inds[1])
    if use_scaler:
        scaler = StandardScaler()
        # scaler = MinMaxScaler((0, 3))
        X = scaler.fit_transform(X)
    return X, y


def cal_explained_variance(fitted_pls, X, Y):
    LX = fitted_pls.x_scores_
    LY = fitted_pls.y_scores_
    V = fitted_pls.x_loadings_
    U = fitted_pls.y_loadings_
    Var_X = np.var(X, axis=0)
    Var_Y = np.var(Y, axis=0)
    vars_x = np.zeros((LX.shape[1],))
    vars_y = np.zeros((LY.shape[1],))
    for i in range(LX.shape[1]):
        X_hat = np.dot(LX[:, :i+1], V[:, :i+1].T)
        X_res = X-X_hat
        Var_X_res = np.var(X_res, axis=0)
        Y_hat = np.dot(LY[:, :i+1], U[:, :i+1].T)
        Y_res = Y-Y_hat
        Var_Y_res = np.var(Y_res, axis=0)
        vars_x[i] = (np.sum(Var_X) - np.sum(Var_X_res))/np.sum(Var_X)
        vars_y[i] = (np.sum(Var_Y) - np.sum(Var_Y_res))/np.sum(Var_Y)
    return vars_x, vars_y


def optimise_pls_cv(X, y, n_comp, name, plot_components=True):
    '''Run PLS including a variable number of components, up to n_comp,
       and calculate MSE '''

    mse = []
    component = np.arange(1, n_comp)

    for i in component:
        pls = PLSRegression(n_components=i)

        # Cross-validation
        y_cv = cross_val_predict(pls, X, y, cv=5)

        mse.append(r2_score(y, y_cv))

        comp = 100*(i+1)/n_comp

    # Calculate and print the position of minimum in MSE
    msemin = np.argmax(mse)
    print("Suggested number of components: ", msemin+1)

    if plot_components is True:
        with plt.style.context(('ggplot')):
            plt.plot(component, np.array(mse), '-v', color='blue', mfc='blue')
            plt.plot(component[msemin], np.array(mse)
                     [msemin], 'P', ms=10, mfc='red')
            plt.xlabel('Number of PLS components')
            plt.ylabel('R-Squared')
            # plt.title('PLS')
            plt.xlim(left=-1)

        plt.tight_layout()

        plt.savefig(name)

    # Define PLS object with optimal number of components
    pls_opt = PLSRegression(n_components=msemin+1)

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


def filter_row_by_value(data, threshold):
    filters = np.where(data.applymap(lambda x: x > threshold))[0]
    filters_column = set(filters)
    m, _ = data.shape
    mask = [True]*m
    for i in filters_column:
        mask[i] = False
    return data.iloc[mask]
