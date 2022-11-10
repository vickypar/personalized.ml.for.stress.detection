import os

import numpy as np
import pandas as pd
from scipy.linalg import sqrtm
from sklearn import preprocessing
from sklearn.model_selection import train_test_split

#from definitions import DATA_DIR, SUBJECT_IDS, ROOT_DIR


def load_activity_map():
    map = {}
    map[0] = 'transient'
    map[1] = 'lying'
    map[2] = 'sitting'
    map[3] = 'standing'
    map[4] = 'walking'
    map[5] = 'running'
    map[6] = 'cycling'
    map[7] = 'Nordic_walking'
    map[9] = 'watching_TV'
    map[10] = 'computer_work'
    map[11] = 'car driving'
    map[12] = 'ascending_stairs'
    map[13] = 'descending_stairs'
    map[16] = 'vacuum_cleaning'
    map[17] = 'ironing'
    map[18] = 'folding_laundry'
    map[19] = 'house_cleaning'
    map[20] = 'playing_soccer'
    map[24] = 'rope_jumping'
    return map


def load_IMU():
    def generate_three_IMU(name):
        x = name + '_x'
        y = name + '_y'
        z = name + '_z'
        return [x, y, z]

    def generate_four_IMU(name):
        x = name + '_x'
        y = name + '_y'
        z = name + '_z'
        w = name + '_w'
        return [x, y, z, w]

    def generate_cols_IMU(name):
        # temp
        temp = name + '_temperature'
        output = [temp]
        # acceleration 16
        acceleration16 = name + '_3D_acceleration_16'
        acceleration16 = generate_three_IMU(acceleration16)
        output.extend(acceleration16)
        # acceleration 6
        acceleration6 = name + '_3D_acceleration_6'
        acceleration6 = generate_three_IMU(acceleration6)
        output.extend(acceleration6)
        # gyroscope
        gyroscope = name + '_3D_gyroscope'
        gyroscope = generate_three_IMU(gyroscope)
        output.extend(gyroscope)
        # magnometer
        magnometer = name + '_3D_magnetometer'
        magnometer = generate_three_IMU(magnometer)
        output.extend(magnometer)
        # oreintation
        oreintation = name + '_4D_orientation'
        oreintation = generate_four_IMU(oreintation)
        output.extend(oreintation)
        return output

    output = ['time_stamp', 'activity_id', 'heart_rate']
    hand = 'hand'
    hand = generate_cols_IMU(hand)
    output.extend(hand)
    chest = 'chest'
    chest = generate_cols_IMU(chest)
    output.extend(chest)
    ankle = 'ankle'
    ankle = generate_cols_IMU(ankle)
    output.extend(ankle)
    return output


def load_subjects(subject_id):
    output = pd.DataFrame()
    cols = load_IMU()
    if subject_id:
        id_range = range(subject_id, subject_id + 1)
    else:
        id_range = SUBJECT_IDS

    pamap_filepath = os.path.join(DATA_DIR, 'pamap2', 'protocol', 'subject')
    for i in id_range:
        print("Reading data from subject {}".format(i))
        path = pamap_filepath + str(i) + '.dat'
        subject = pd.read_table(path, header=None, sep='\s+')
        subject.columns = cols
        subject['id'] = i
        output = output.append(subject, ignore_index=True)
    output.reset_index(drop=True, inplace=True)
    return output


def fill_nan(data):
    data = data.interpolate()
    # fill all the NaN values in a column with the mean values of the column
    for colName in data.columns:
        data[colName] = data[colName].fillna(data[colName].mean())
    # activity_mean = data.groupby(['activity_label']).mean().reset_index()
    # Alternative for filling NaN values: Interpolate data
    # data = data.interpolate()
    return data


def fix_column_names(data):
    data.rename(columns={'time_stamp': 'timestamp', 'id': 'user_id', 'activity_id': 'activity_label'}, inplace=True)
    return data


# todo Seperate test and train set per peron, i.e., 80% of a person's data for training and 20% for testing
def create_dataset_column(data):
    """
    Adds a column that specifies which dataset (train vs. test) the row belongs to
    :param data: The dataframe to add a column to
    :return: The dataframe with the new "dataset" column
    """
    X_train, X_test, y_train, _ = train_test_split(data, data.activity_label, test_size=0.2, random_state=42)
    _, X_val, _, _ = train_test_split(X_train, y_train, test_size=0.1, random_state=43)

    train_index = X_train.index
    val_index = X_val.index
    data['dataset'] = 'Test'
    data.loc[train_index, ['dataset']] = 'Train'
    data.loc[val_index, ['dataset']] = 'Val'
    # data.loc[test_index]['dataset'] = 'Test'
    return data


def drop_null_activities(data):
    drop_index = []

    # Getting indexes of activity 0
    drop_index += list(data.index[data['activity_label'] == 0])

    # Keep only activities as documented on file "PerformedActivitiesSummary.pdf"
    drop_index += list(data.index[(data['user_id'] == 1) & (data['activity_label'].isin([10, 20]))])
    drop_index += list(data.index[(data['user_id'] == 2) & (data['activity_label'].isin([9, 10, 11, 18, 19, 20]))])
    drop_index += list(
        data.index[(data['user_id'] == 3) & (data['activity_label'].isin([5, 6, 7, 9, 10, 11, 18, 19, 20, 24]))])
    drop_index += list(
        data.index[(data['user_id'] == 4) & (data['activity_label'].isin([5, 9, 10, 11, 18, 19, 20, 24]))])
    drop_index += list(data.index[(data['user_id'] == 5) & (data['activity_label'].isin([9, 11, 18, 20]))])
    drop_index += list(data.index[(data['user_id'] == 6) & (data['activity_label'].isin([9, 11, 20]))])
    drop_index += list(
        data.index[(data['user_id'] == 7) & (data['activity_label'].isin([9, 10, 11, 18, 19, 20, 24]))])
    drop_index += list(data.index[(data['user_id'] == 8) & (data['activity_label'].isin([9, 11]))])
    drop_index += list(data.index[(data['user_id'] == 9) & (
        data['activity_label'].isin([1, 2, 3, 4, 5, 6, 7, 9, 11, 12, 13, 16, 17]))])

    data = data.drop(drop_index)
    return data


def remove_transient_data(data):
    freq = 100
    data['act_block'] = ((data['activity_label'].shift(1) != data['activity_label']) | (
            data['user_id'].shift(1) != data['user_id'])).astype(int).cumsum()
    drop_index = []
    numblocks = data['act_block'].max()
    for block in range(1, numblocks + 1):
        drop_index += list(data[data['act_block'] == block].head(10 * freq).index)
        drop_index += list(data[data['act_block'] == block].tail(10 * freq).index)
    data = data.drop(drop_index)
    data.drop("act_block", axis=1, inplace=True)
    return data


def drop_attributes(data):
    attributes_to_drop = ['magnetometer', 'temperature', 'orientation']
    for attr in attributes_to_drop:
        data = data.loc[:, ~data.columns.str.contains(attr, case=False)]
    return data


def preprocess_pamap2_data(subject_id):
    data = load_subjects(subject_id)  # Read data for all users
    print("Initial Data Shape: " + str(data.shape))
    data = fix_column_names(data)  # Make label names more understandable
    data = drop_null_activities(data)  # Drops rows that correspond to activities not performed by the users
    print("Data shape after dropping activity 0: " + str(data.shape))
    data = fill_nan(data)  # Fill NaN values
    data = remove_transient_data(data)
    print("Data shape after removing transient data: " + str(data.shape))
    data = drop_attributes(data)
    # data = create_dataset_column(data)  # Assign rows to dataset (train, test, validation) - not implemented for
    # personalized learning yet
    # data.to_csv('./multitask_data.csv')
    return data.reset_index(drop=True)


def center(x):
    mean = np.mean(x, axis=1, keepdims=True)
    centered = x - mean
    return centered, mean


def covariance(x):
    mean = np.mean(x, axis=1, keepdims=True)
    n = np.shape(x)[1] - 1
    m = x - mean

    return (m.dot(m.T)) / n


def ccaComparison(acc_left, acc_right):
    Cov_xx = acc_left @ acc_left.T
    Cov_yy = acc_right @ acc_right.T
    Cov_xy = acc_left @ acc_right.T

    Cov_xx_sqrt = sqrtm(np.linalg.inv(Cov_xx))
    Cov_yy_sqrt = sqrtm(np.linalg.inv(Cov_yy))

    prod = Cov_xx_sqrt @ Cov_xy @ Cov_yy_sqrt

    U, s, V = np.linalg.svd(prod)
    print(s)

    def ccaDistance(s, component=None):
        try:
            distance = 1 - (s[:component].sum() / component)
        except:
            distance = 1 - (s.sum() / s.ravel().shape[0])
        return distance

    distance = ccaDistance(s, None)
    return distance


def ccaComponents(signal):
    signal_x = signal[:, 1:]
    signal_y = signal[:, :-1]

    Cxx = signal_x @ signal_x.T
    Cyy = signal_y @ signal_y.T
    Cxy = signal_x @ signal_y.T
    Cyx = Cxy.T

    """
    Cxx=np.cov(signal_x)
    Cyy=np.cov(signal_y)
    C=np.cov(signal_x,signal_y)
    nr_signals=signal_x.shape[0]
    Cxy=C[nr_signals:,:nr_signals]
    Cyx=Cxy.T
    """

    CxxInv = np.linalg.pinv(Cxx)
    CyyInv = np.linalg.pinv(Cyy)

    Cov_xx_sqrt = sqrtm(np.linalg.inv(Cxx))
    Cov_yy_sqrt = sqrtm(np.linalg.inv(Cyy))

    prod = Cov_xx_sqrt @ Cxy @ Cov_yy_sqrt

    U, s, V = np.linalg.svd(prod)

    RhoX = CxxInv @ Cxy @ CyyInv @ Cyx
    RhoY = CyyInv @ Cyx @ CxxInv @ Cxy

    Sx, Wx = np.linalg.eigh(RhoX)
    Sy, Wy = np.linalg.eigh(RhoY)
    return Wx, Wy, Sx, Sy, U, s, V


def ccaReconstruction(datac, threshold=0.5):
    """
    Reconstruction of the signals using cannonical correlation analysis (CCA method).
    Input:
        datac[nr_channels,dims]: the signals with mean substracted along the each channel
        threshold: the minimum value of the correlation value allowed for the reconstruction.
    Output:
        reconstruction_signal[nr_channels,dims]: the reconstruction signal using solely the most correlated components.
    Running eg. reconstruction_signal=ccaReconstruction(datac,threshold=0.5)
    """
    Wx, Wy, Sx, Sy, U, s, V = ccaComponents(datac)
    assert s.ravel().min() >= 0
    assert s.ravel().max() <= 1
    cca_channels = U.T @ datac
    Wx_inv = np.linalg.pinv(U)

    index = np.argmin(np.abs(s - threshold))
    reconstruction_signal = Wx_inv[:index].T @ cca_channels[:index]
    return reconstruction_signal


# Whiten mixed signals
def whiten(x, epsilon=0, normalize=True, diagonalize=True):
    """
    This function performs an linear transformation of the time series x[dims,nr_channels].
    For the transformation to be successful the time series have to be normalized per channel.
    *) If normalize=True then a the auto-covariance of each channel is scaled to one (a.k.a batch normalization)
    **) If diagonalize=True the cross-covariance values are suppressed down to zero using the rotation matrix V 
    ***) If both diagonalize=normalize=False then nothing happens.
    """
    # print(x.shape)
    print("Whitening data with settings: Normalize={} and Diagonalize={}".format(normalize,diagonalize))

    # Calculate the covariance matrix
    coVarM = covariance(x)  # change from x.T

    if normalize:
        if diagonalize:
            # Single value decoposition
            U, S, V = np.linalg.svd(coVarM)

            # Calculate diagonal matrix of eigenvalues
            d = np.diag(1.0 / np.sqrt(S + epsilon))
            # Calculate whitening matrix
            whiteM = np.dot(U, np.dot(d, U.T))

            # Project onto whitening matrix
            Xw = np.dot(whiteM, x.T)
        else:
            Cov = np.diag(coVarM)
            d = np.diag(Cov)
            whiteM = np.linalg.inv(np.sqrt(d))
            Xw = whiteM @ x  # change from x.T
    elif diagonalize:
        # Single value decoposition
        U, S, V = np.linalg.svd(coVarM)

        whiteM = U  # change from V
        Xw = whiteM @ x.T
    else:
        whiteM = np.diag(np.ones((min(x.shape), 1)).squeeze())
        Xw = x

    return Xw, whiteM  # Xw contains the whitened features


def preprocess_pamap2_data_single_subject(subject_id, normalize=True, diagonalize=True):
    data_single_subject = preprocess_pamap2_data(subject_id)
    data_single_subject_array = data_single_subject.iloc[:, 2:-1].to_numpy()  # Convert feature columns into an ndarray
    datac, meandata = center(data_single_subject_array.T)  # Center signals
    # Here we apply normalization and/or diagonalization based on passed settings
    datacw, whiteM = whiten(datac, normalize=normalize, diagonalize=diagonalize)
    # Comment this out for understandability, but include it for model training
    le = preprocessing.LabelEncoder()  # Encode labels to have continuous numbers, e.g., 1,2,3,4 instead of 1,3,6,10
    data_single_subject['activity_label'] = le.fit_transform(data_single_subject['activity_label'].values)
    return data_single_subject, datacw


def preprocess_pamap2_data_multiple_subjects(normalize=True, diagonalize=True):
    data_all_subjects = preprocess_pamap2_data(None)
    data_all_subjects_array = data_all_subjects.iloc[:, 2:-1].to_numpy()  # Convert feature columns into an ndarray
    datac, meandata = center(data_all_subjects_array.T)  # Center signals
    datacw, whiteM = whiten(datac, normalize=normalize, diagonalize=diagonalize)
    # todo adapt the multiple subjects preprocessing to the new whiten function
    # Comment this out for understandability, but include it for model training
    le = preprocessing.LabelEncoder()  # Encode labels to have continuous numbers, e.g., 1,2,3,4 instead of 1,3,6,10
    data_all_subjects['activity_label'] = le.fit_transform(data_all_subjects['activity_label'].values)
    return data_all_subjects, datacw

def keep_common_labels(df):
    print("Keeping only labels that are common accross all subjects...")
    min_labels = 13
    subject_min = None
    labels = None
    for user in SUBJECT_IDS:
        df_user = df[df.user_id == user]
        df_user_labels = np.unique(df_user.activity_label)
        user_labels = len(df_user_labels)
        if user_labels < min_labels:
            min_labels = user_labels
            labels = df_user_labels
            subject_min = user
    # print("Min {} - Subject {}".format(min_labels, subject_min))
    df_new = df.loc[df['activity_label'].isin(labels)]

    return df_new


def preprocess_pamap2_data_multitask():
    filepath = os.path.join(ROOT_DIR, "data_preprocessing", "personalized_multitask_learning", "multitask_data.csv")
    print(filepath)
    if (os.path.exists(filepath)):
        print("Reading preprocessed data from file...")
        data = pd.read_csv(filepath)
        return data
    data_all_subjects = preprocess_pamap2_data(None)
    # features = data_all_subjects.iloc[:, 2:-1].to_numpy()

    data = create_dataset_column(data_all_subjects)
    data = keep_common_labels(data)
    data.to_csv(filepath)
    return data