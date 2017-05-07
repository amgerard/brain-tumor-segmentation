import numpy as np
# from sklearn.metrics import confusion_matrix

def metrics_report(y_true, y_pred):
    pass

def metrics_report(conf_mat, true_cls_idx):
    '''
    sensitivity = tp/(tp+fn)
    - 
    specificity = tn/(tn+fp)
    '''
    tp = conf_mat[true_cls_idx, true_cls_idx]
    fp = np.sum(conf_mat[:, true_cls_idx]) - tp
    tn = np.sum(conf_mat[np.diag_indices(conf_mat.shape[0])]) - tp
    fn = np.sum(conf_mat) - tp - fp - tn
    return float(tp)/(tp+fn), float(tn)/(tn+fp)

if __name__ == "__main__":
    m = [[1971, 19, 1, 8, 0, 1],
         [16, 1940, 2, 23, 9, 10],
         [8, 3, 1891, 87, 0, 11],
         [2, 25, 159, 1786, 16, 12],
         [0, 24, 4, 8, 1958, 6],
         [11, 12, 29, 11, 11, 1926]]

    m = [[50, 10],[5,100]]

    m = [[6358, 34, 615, 83, 13],
         [147, 3668, 304, 1663, 211],
         [628, 132, 6374, 1092, 216],
         [202, 955, 2017, 4047, 1047],
         [35, 67, 786, 1155, 5651]]

    M = np.array(m)
    print(np.sum(M))
    for i in range(M.shape[0]):
        sens, spec = metrics_report(M, i)
        print('sens', sens)
        print('spec', spec)