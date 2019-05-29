import sys
import numpy as np
from sklearn.impute import SimpleImputer
import pandas as pd
from scipy.stats import zscore,iqr
from sklearn.ensemble import IsolationForest
from sklearn import metrics
from scipy.stats import multivariate_normal
from sklearn.metrics.pairwise import euclidean_distances
import eif as iso
from sklearn.cluster import DBSCAN
from sklearn.covariance import EllipticEnvelope
from sklearn.neighbors import LocalOutlierFactor
import statsmodels.api as sm
from sklearn import svm
from scipy.spatial.distance import cdist
from sklearn.preprocessing import MinMaxScaler
from scipy.stats import norm, kstest
from collections import defaultdict
from collections import Counter
from numpy.random import RandomState
from imblearn.over_sampling import SMOTENC,SMOTE,ADASYN
from sklearn.preprocessing import OneHotEncoder,PowerTransformer
import seaborn as sb
from sklearn.cluster import KMeans


class PreProcessor:
    """ Performs data preprocessing

        The objective of this class is to preprocess the data based on training subset. The
        preprocessing steps focus on constant features removal, missing values treatment and
        outliers removal and imputation.

    """

    def __init__(self, training, unseen, target_var, seed):
        """ Constructor

            It is worth to notice that both training and unseen are nothing more nothing less
            than pointers, i.e., pr.training is DF_train and pr.unseen is DF_unseen yields True.
            If you want them to be copies of respective objects, use .copy() on each parameter.

        """
        self.training = training #.copy() to mantain a copy of the object
        self.unseen = unseen #.copy() to mantain a copy of the object
        self.target_var = target_var
        self.seed = seed
        
        self.missing_value_reporter(self.training)
        self._impute_missing_values()
        
# =============================================================================
#         ###Outlier Analysis
#         #outliers = self._boxplot_outlier_detection()
#         outliers = self.extended_isolation_forest()
#         print(outliers)
#         outlier_percent = len(outliers)/self.training.shape[0]
#         if outlier_percent > 0.03:
#             print(">>>>>>> Warning, outlier percentage above 0.03 ("+ str(outlier_percent) + ")")
#             answer = input("Stop execution?")
#             if answer == 'yes':
#                 raise("Too much outliers")
#         self.training.drop(index = [int(i) for i in outliers])
#         ###Normalization
#         
#         
# =============================================================================


    #DEALING WITH MISSING VALUES
    
    def missing_value_reporter(self,dataframe, threshold = 0.03, report = True):
    
        '''
        Reports on percentages of missing values per columns 
        returning columns above threshold
        '''
        print(dataframe.columns)
        columns = []
        for column in dataframe.columns:
            percentage = round(np.sum(dataframe[column].isna())/dataframe.shape[0],2)
            if percentage > 0 and report:       
                print(column+': ' + ' ' + str(percentage))
            if percentage > threshold:
                columns.append(column)
        total_rows = dataframe[columns].shape[0] - dataframe[columns].dropna().shape[0]
        percent = total_rows/dataframe.shape[0]
        if report:
            print('\n Total Removed Rows (' + str(threshold) + '): ' + str(total_rows) + ' ' + str(round(percent,2)*100) +'%')
        return columns
    
    def _drop_missing_values(self):
        #self.report.append('_drop_missing_values')
        self.training.dropna(inplace=True)
        self.unseen.dropna(inplace=True)

    def convert_numeric_labelling(self,var):
        temp = self.training[var].dropna().copy()
        unique = temp.drop_duplicates()
        var_dict = {}
        for ix,value in enumerate(unique):
            var_dict[value] = ix
        self.training[var] = self.training[var].apply(lambda x: var_dict[x] if x in var_dict.keys() else None)
        return var_dict

    def revert_numeric_labelling(self,var,var_dict):
        cat_dict = {}
        for k,v in var_dict.items():
            cat_dict[v] = k
        self.training[var] = self.training[var].apply(lambda x: cat_dict[x] if x in cat_dict.keys() else None)

    def _impute_missing_values(self):
        #self.report.append('_impute_missing_values')
        for column in self.training:
            data = self.training[column]
            loc, scale = norm.fit(data)
            n = norm(loc = loc, scale = scale)
            _, p_value = kstest(self.training[column].values, n.cdf)
            if p_value > 0.05:
                self._imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
                self.training[column] = self._imputer.fit_transform(self.training[column].values.reshape(-1,1))
                self.unseen[column] = self._imputer.transform(self.unseen[column].values.reshape(-1,1))
            else:
                self._imputer = SimpleImputer(missing_values=np.nan, strategy='median')
                self.training[column] = self._imputer.fit_transform(self.training[column].values.reshape(-1,1))
                self.unseen[column] = self._imputer.transform(self.unseen[column].values.reshape(-1,1))

    # DEALING WITH OUTLIERS
    ### UNIVARIATE OUTLIER DETECTION
    def _filter_df_by_std(self):
        #self.report.append('_filter_df_by_std')
        '''Removes Outliers based on standard deviation'''
        def _filter_ser_by_std(series_, n_stdev=3.0):
            mean_, stdev_ = series_.mean(), series_.std()
            cutoff = stdev_ * n_stdev
            lower_bound, upper_bound = mean_ - cutoff, mean_ + cutoff
            return [True if i < lower_bound or i > upper_bound else False for i in series_]

        training_num = self.training[self.numerical_var].drop(["Response"], axis=1)
        mask = training_num.apply(axis=0, func=_filter_ser_by_std, n_stdev=3.0)
        training_num[mask] = np.NaN
        self.training[training_num.columns] = training_num
        return list(training_num.columns)

    def _z_score_outlier_detection(self, treshold):
        self.report.append('_z_score_outlier_detection')
        ''' Only for numerical data'''
        z_score_outliers = {}
        for var in self.numerical_var:
            if var != 'Response':
                df = pd.Series(zscore(self.training[var]))
                df.index = self.training.index
                for ind in df[np.abs(df) > treshold].index:
                    try:
                        z_score_outliers[ind] = z_score_outliers[ind] + ' ' + var
                    except:
                        z_score_outliers[ind] = var
        for key in z_score_outliers.keys(): z_score_outliers[key] = z_score_outliers[key].split(' ')
        return z_score_outliers

    def _boxplot_outlier_detection(self, percent = 0.03, treshold=1.5, ranking = False):
        #self.report.append('_boxplot_outlier_detection')
        box_plot_outliers = []
        #For each var check ouytliers and their distance
        for var in self.training.drop(self.target_var, axis = 1).columns:
            df = pd.Series(zscore(self.training[var].copy()))
            df.index = self.training.index.copy()
            iqr_ = iqr(df)
            Q1 = np.percentile(df, 25)
            Q3 = np.percentile(df, 75)
            ix1 = df.index[np.where(df.values > (Q3 + treshold * iqr_))[0]]
            indexes = list(zip(ix1, [var] * len(ix1), df.ix[ix1].values - (Q3 + treshold * iqr_)))
            ix2 = df.index[np.where(df.values < (Q1 - treshold * iqr_))[0]]
            indexes2 = list(zip(ix2, [var] * len(ix2), np.abs((df.ix[ix2].values + (Q1 - treshold * iqr_)))))
            indexes = indexes + indexes2
            box_plot_outliers = box_plot_outliers + indexes

        #Sort outliers in descending order by their distance
        box_plot_outliers.sort(key = lambda x: x[2],reverse = True)
        if ranking:
            box_plot_outliers = [(item[0],item[1]) for item in box_plot_outliers]
            outlier_dict = {}
            for item in box_plot_outliers:
                if item[0] in outlier_dict.keys():
                    outlier_dict[item[0]] = list(outlier_dict[item[0]]) + list([item[1]])
                else:
                    outlier_dict[item[0]] = [item[1]]
            return outlier_dict
        #Delete repeating ids, leave the ones with the highest distance
        to_delete = []
        for x in range(len(box_plot_outliers)):
            a = box_plot_outliers[x]
            for y in range(1,len(box_plot_outliers)):
                b = box_plot_outliers[y]
                if a[0] == b[0]:
                    if a[1] > b[1]:
                        to_delete.append(b)
                    else:
                        to_delete.append(a)
        box_plot_outliers = np.array(box_plot_outliers)
        for x in range(len(box_plot_outliers)-1,-1):
            del box_plot_outliers[x]
        n = int(self.training.shape[0] * percent)
        return [x[0] for x in box_plot_outliers[:n]]

    def robust_z_score_method(self, treshold=5):
        self.report.append('robust_z_score_method')
        
        robust_zs_outliers = {}
        df = self.training[self.numerical_var]
        size = len(df)
        for var in self.numerical_var:
            if var != 'Response':
                ds = pd.Series(df[var])
                ds.index = df.index
                median = np.median(ds)
                MAD = np.median([np.abs(ds.iloc[i] - median) for i in range(size)])
                modified_z_scores = pd.Series([0.6745 * (ds.iloc[i] - median) / MAD for i in range(size)])
                modified_z_scores.index = ds.index

                for ind in modified_z_scores[np.abs(modified_z_scores) > treshold].index:
                    try:
                        robust_zs_outliers[ind] = robust_zs_outliers[ind] + ' ' + var
                    except:
                        robust_zs_outliers[ind] = var

        for key in robust_zs_outliers.keys(): robust_zs_outliers[key] = robust_zs_outliers[key].split(' ')
        return robust_zs_outliers

    #### MULTIVARIATE OUTLIER DETECTION
    """ CONTAMINATION: The amount of contamination of the data set, i.e. the proportion of outliers in the data set. 
    Used when fitting to define the threshold on the decision function. """

    def isolation_forest(self, contamination, seed):
        self.report.append('isolation_forest')
        clf = IsolationForest(max_samples=100, contamination=contamination, random_state=seed)
        clf.fit(self.training)
        outliers_isoflorest = clf.predict(self.training)
        outliers_isoflorest = pd.Series(outliers_isoflorest)
        outliers_isoflorest.index = self.training.index
        return np.array(outliers_isoflorest[outliers_isoflorest == -1].index)

    def uni_boxplot_outlier_det(self,series_):
        outliers = []
        Q1 = np.percentile(series_, 25)
        Q3 = np.percentile(series_, 75)
        for record in series_.index:
            if series_[series_.index == record].iloc[0] > (Q3 + 1.5 * iqr(series_)) or series_[series_.index == record].iloc[
                0] < (Q1 - 1.5 * iqr(series_)):
                outliers.append(record)
        return outliers

    def extended_isolation_forest(self):

        if_eif = iso.iForest(self.training.astype('float64').values, ntrees=3, sample_size=40, ExtensionLevel=2)
        anomaly_scores = if_eif.compute_paths(X_in=self.training.values)
        anomaly_scores = pd.Series(anomaly_scores)
        anomaly_scores.index = self.training.index
        return self.uni_boxplot_outlier_det(anomaly_scores)

    # Getting the columns (variables) means
    def mahalanobis_distance_outlier(self):
        self.report.append('mahalanobis_distance_outlier')
        ds = self.training.copy()[self.numerical_var]
        #ds[self.cat_vars] = ds[self.cat_vars].astype(float)
        var_means = ds.mean(axis=0).values.reshape(1, -1)
        # Getting the inverse of the covariance matrix
        try:
            inv_cov_matrix = np.linalg.inv(np.cov(ds.T))
        except:
            m = np.cov(ds.T)
            i = np.eye(m.shape[0], m.shape[1])
            inv_cov_matrix = np.linalg.lstsq(m, i)[0]
        try:
            np.linalg.cholesky(inv_cov_matrix)
            print("all good")
        except:
            print("WARNING, mahalanobis no good..")
            # REVER ISTO...
        MDs = cdist(var_means, ds, 'mahalanobis', VI=inv_cov_matrix)
        MDs = pd.Series([distance for sublist in MDs for distance in sublist], index=ds.index)

        def find_outliers(MDs, percent = 0.03):
            treshold = 3
            std = np.std(MDs)
            k = treshold * std
            m = np.mean(MDs)
            l_outliers = MDs[MDs >= m+k]
            l_outliers = l_outliers - (m+k)
            h_outliers = MDs[MDs <= m-k]
            h_outliers = np.abs(h_outliers + (m-k))
            t_outliers = pd.concat([l_outliers,h_outliers])
            t_outliers.sort_values(ascending= False, inplace=True)
            n = int(MDs.shape[0] * percent)
            return t_outliers.index[:n]
        return find_outliers(MDs)

    def dbscan_outlier_detection(self, minpoints=None, radius=None):
        self.report.append('dbscan_outlier_detection')
        ds = self.training[self.numerical_var]
        outlier_detection = DBSCAN(
            eps=radius,
            metric="euclidean",
            min_samples=minpoints,
            n_jobs=-1)

        clusters = outlier_detection.fit_predict(ds)
        # Seing the number of identified noise (outliers)
        # Identifying the outliers:
        clusters = pd.Series(clusters)
        clusters.index = ds.index
        return clusters[clusters == -1].index

    def elliptic_envelope_out(self, contamination):
        self.report.append('elliptic_envelope_out')
        ds = self.training[self.numerical_var]
        elliptic = EllipticEnvelope(contamination=contamination)
        elliptic.fit(ds)
        results = elliptic.predict(ds)
        outlier_elliptic = pd.Series(results)
        outlier_elliptic.index = ds.index
        return outlier_elliptic[outlier_elliptic == -1].index

    def local_outlier_factor(self, n_neighbors = None, contamination = None):
        self.report.append('local_outlier_factor')
        ds = self.training[self.numerical_var]
        lof = LocalOutlierFactor(n_neighbors=n_neighbors, contamination=contamination)
        outiler_lof = lof.fit_predict(ds)
        outiler_lof = pd.Series(outiler_lof)
        outiler_lof.index = ds.index
        return outiler_lof[outiler_lof == -1].index

    def one_class_svm(self):

        self.report.append('one_class_svm')
        ds = self.training[self.numerical_var]
        oneclasssvm = svm.OneClassSVM()
        oneclasssvm_outliers = oneclasssvm.fit_predict(ds)
        oneclasssvm_outliers = pd.DataFrame(oneclasssvm_outliers, index = ds.index)
        return oneclasssvm_outliers[oneclasssvm_outliers == -1].index

    def cooks_distance_outlier(self, vd):
        self.report.append('cooks_distance_outlier')
        df = self.training
        X = df[vd].astype('float64').values
        Y = df.drop(columns=vd).astype('float64').values
        m = sm.OLS(X, Y).fit()
        infl = m.get_influence()
        sm_fr = infl.summary_frame()
        return self.uni_boxplot_outlier_det(sm_fr['cooks_d'].sort_values(ascending=False))

    def uni_iqr_outlier_smoothing(self,DOAGO_results, ds):
        '''use the info gatheres from the previous function (decide on and get outliers) and smoothes the detected outliers.'''
        self.report.append('uni_iqr_outlier_smoothing')
        novo_ds = ds.copy()
        print(DOAGO_results)
        for key in DOAGO_results.keys():
            print(key)
            for var in DOAGO_results[key]:
                print(var)
                if var != 'multi/unknown':
                    if ds[var][ds.index == key].values > np.mean(ds[var]):
                        novo_ds[var][novo_ds.index == key] = np.percentile(ds[var], 75) + 1.5 * iqr(ds[var])

                    else:
                        novo_ds[var][novo_ds.index == key] = np.percentile(ds[var], 25) - 1.5 * iqr(ds[var])
                else: novo_ds=novo_ds[novo_ds.index!=key]

        return novo_ds

    def outlier_rank(self,smoothing,treshold,*arg):
        '''this function returns two things. First, the number of times each row appears as outlier. The second is the full dictionary with the indexes and the columns where they
        appear as outliers.
        '''
        self.report.append('outlier_rank')
        IDS = []
        ids = []
        keys = []
        for array in arg:
            if isinstance(array, dict):
                for key in array.keys():
                    keys.extend([key for _ in range(len(array[key]))])
            else:
                if (len(np.array(array).shape)) < 2:
                    ids.extend([id_ for id_ in array])
                else:
                    IDS.extend([id_ for sublist in array for id_ in sublist])

        full = IDS + keys + ids
        counts = np.array([full.count(i) for i in full]).reshape(-1,1)
        scaler = MinMaxScaler()
        scaler.fit(counts)
        counts=scaler.transform(counts)
        counts=[item for sublist in counts for item in sublist]
        pass_for_full = [thing for thing in arg]

        def get_full_outlier_dict(full_dict):
            dd = defaultdict(list)
            [print(arg_) for arg_ in full_dict]
            for d in ([arg_ for arg_ in full_dict]):
                if isinstance(d,dict):
                    for key, value in d.items():
                        dd[key].append(value)

                else:
                    for id in d:
                        dd[id]=[['multi/unknown']]
            dd = dict(dd)

            for key in dd.keys():
                # flattning the values of the dicts
                dd[key] = [item for sublist in dd[key] for item in sublist]
                # removing the duplicated variables where a key is outlier
                dd[key] = list(dict.fromkeys(dd[key]))
            return dd

        outlier_info_dict = get_full_outlier_dict(pass_for_full)
        counts=dict(sorted(dict(zip(full, counts)).items(), key=lambda x: x[1], reverse=True))
        outliers=list({key for (key, value) in counts.items() if value > treshold})

        if smoothing:
            self.report.append('uni_iqr_rank_outlier_smoothing')
            outlier_info = dict([(k, v) for (k, v) in outlier_info_dict.items() if k in outliers])
            print(outlier_info)
            self.training=self.uni_iqr_outlier_smoothing(outlier_info,self.training)

        else:
            self.training=self.training[~self.training.index.isin(outliers)]

    def decide_on_and_get_outliers(self,outlier_rank_result, treshold):
        '''pass the ranking here and choose the records that appear more than  treshold times to be our outliers'''
        outliers = list({key for (key, value) in outlier_rank_result[0].items() if value > treshold})
        outlier_info = dict([(k, v) for (k, v) in outlier_rank_result[1].items() if k in outliers])
        return outlier_info

    def power_transformation(self):
        pt=PowerTransformer()
        pt.fit(self.training.select_dtypes(exclude='category'))
        temp = pd.DataFrame(pt.transform(self.training.select_dtypes(exclude='category')))
        temp.index=self.training.select_dtypes(exclude='category').index
        temp.columns=self.training.select_dtypes(exclude='category').columns
        for col in self.training.select_dtypes(include='category'):
            temp[col]=self.training[col]
        self.training=temp
        del temp
        temp=pd.DataFrame(pt.transform(self.unseen.select_dtypes(exclude='category')))
        temp.index = self.unseen.select_dtypes(exclude='category').index
        temp.columns = self.unseen.select_dtypes(exclude='category').columns
        for col in self.unseen.select_dtypes(include='category'):
            temp[col]=self.unseen[col]
        self.unseen=temp
        print(temp)


        def get_k_means_elbow_graph(ds, numerical, min_clust, max_clust):
            km = pd.DataFrame(columns=['num_clusters', 'inertia'])
            for i in range(min_clust, max_clust):
                kmeans = KMeans(n_clusters=i).fit(self.training.select_dtypes(exclude='category'))
                km = km.append({'num_clusters': i, 'inertia': kmeans.inertia_}, ignore_index=True)
            sb.lineplot(x=km['num_clusters'], y=km['inertia'])
            return




    ### NORMALIZATION
    def _normalize(self):
        print("Inside: Normalize", self.training.shape,self.unseen.shape)
        self.report.append('_normalize')
        dummies = list(self.training.select_dtypes(include=["category", "object"]).columns)
        dummies.append('Response')
        print(self.training.columns[~self.training.columns.isin(dummies)])
        print(self.unseen.columns[~self.unseen.columns.isin(dummies)])
        scaler = MinMaxScaler()
        print('normalize', self.training.shape)
        scaler.fit(self.training[self.training.columns[~self.training.columns.isin(dummies)]].values)
        self.training[self.training.columns[~self.training.columns.isin(dummies)]] = pd.DataFrame(
            scaler.transform(self.training[self.training.columns[~self.training.columns.isin(dummies)]].values),
            columns=self.training[self.training.columns[~self.training.columns.isin(dummies)]].columns,
            index=self.training[self.training.columns[~self.training.columns.isin(dummies)]].index)
        print('normalize',self.unseen.shape)
        self.unseen[self.unseen.columns[~self.unseen.columns.isin(dummies)]] = pd.DataFrame(
            scaler.transform(self.unseen[self.unseen.columns[~self.unseen.columns.isin(dummies)]].values),
            columns=self.unseen[self.unseen.columns[~self.unseen.columns.isin(dummies)]].columns,
            index=self.unseen[self.unseen.columns[~self.unseen.columns.isin(dummies)]].index)