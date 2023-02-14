from pandas import read_csv
import numpy as np
from scipy import stats
import pickle


#directory for the data
data_dir = 'WESAD/'

#label sampling frequency
LABEL_SF = 700

#sampling rates for all the E4 sensors
ACC_SF = 32
BVP_SF = 64
EDA_SF = 4
TEMP_SF = 4
#sampling frequencies in a dictionary
SF_dict = {'ACC':ACC_SF, 'BVP':BVP_SF, 'EDA':EDA_SF, 'TEMP':TEMP_SF}

#features from the E4 device
features = ['ACC','BVP','EDA','TEMP']

#relavent state labels
baseline_label = 1
stress_label = 2
meditation_label = 4
invalid_labels = [0, 3, 5, 6, 7] #amusement (labelled as 3) will not be used for this project


def get_subject_data(subject):
    """
        @brief Returns all the watch sensor data for the specified subject
        The amusement label will be considered invalid, and meditation label
        will be combined with the baseline label to create a non-stressed label
        @param: subject (string): The number corresponding to the desired subject
        @return: Returns the sensor data and labels
    """

    #open pkl file for desired subject
    with open(data_dir+'S'+subject+'/S'+subject+'.pkl', 'rb') as file:
        data = pickle.load(file, encoding='latin1')

    #get the labels from the data
    data_labels = data['label']
    data = data['signal']['wrist']

    #create window labels (the window labels are the same for all data)
    window_labels = create_labels(data_labels, LABEL_SF)

    #mask for removing invalid labels
    mask = [x for x in range(len(window_labels)) if window_labels[x] not in invalid_labels]
    valid_labels = window_labels[mask]

    for feat in features:
        #form into windows (and line up with labels)
        data[feat] = create_windows(data[feat],SF_dict[feat])
        #remove invalid labels
        #print(mask)
        #data[feat] = data[feat][mask] [main_list[x] for x in mask]
        data[feat] = [data[feat][x] for x in mask]

    #re-label meditation data as baseline, or non-stressed
    med_mask = [x for x in range(len(valid_labels)) if valid_labels[x] == meditation_label]
    valid_labels[med_mask] = baseline_label
    #shift the labels to be 0 and 1, rather than 1 and 2 for binary crossentropy later
    valid_labels -= 1
    #roughly 3 times as much non-stress as stress data

    #combine the data for the subject into a dictionary
    final_data = dict()
    final_data['data'] = {'ACC' : data['ACC'], 'BVP' : data['BVP'],
            'EDA' : data['EDA'], 'TEMP' : data['TEMP']}
    final_data['labels'] = valid_labels

    return final_data

def get_all_subjects():
    """
        @breif: Gets a dictionary with the combined data for all subjects
        @return: Returns a dictionary with the combined subject data
    """
    data = {'data': {'ACC':np.empty((0,ACC_SF,3)), 'BVP':np.empty((0,BVP_SF,1)),
            'EDA':np.empty((0,EDA_SF,1)), 'TEMP':np.empty((0,TEMP_SF,1))},
                'labels':[]}

    #gather and append all subject data
    for x in range(2,18):
    #subjects 1 and 12 were not included in the published data
        if x != 12:
            temp = get_subject_data(str(x))
            for i in features:
                data['data'][i] = np.append(data['data'][i],temp['data'][i],0)
                data['labels'] = np.append(data['labels'],temp['labels'],0)
    return data


def norm(data):
    """
        @brief Mean normalize the data (subtract mean, divide by std)
        @param: data (list): The data to normalize
        @return: The normalized data
    """
    normalized_data = (data - np.mean(np.mean(data,0),0))/np.std(np.std(data,0),0)
    return normalized_data

def create_labels(all_labels, SF):
    """
        @brief Returns the labels for the desired data in one second windows
                overlapping 50%
        @param: all_labels (list): The full list of labels
        @param: SF (int): The sampling frequency of the labels
        @return: The windowed label list for the data
    """
    labels = []
    for x in range(0, len(all_labels) - SF//2, SF//2):
        labels.append(all_labels[x])
    return np.array(labels)

def create_windows(data, data_SF, window_size=1):
    """
        @brief Divide the data into one second windows with 50% overlap
        The rounded average of the label values will be used as the window label
        @param: data (list): The data to divide
        @param: data_SF (int): The sampling frequency of the data
        @param: window_size (int): The number of seconds in each window
        @return: The windowed data and labels
    """
    data_windows = []
    for x in range(0, len(data) - window_size * (data_SF//2), window_size * (data_SF//2)):
        data_windows.append(data[x : window_size * data_SF + x])

    return data_windows

def save_data(path, data):
    """
        @brief Save the data to the given path
        @param: path (string): The path for the saved data
        @param: data (list, array, tuple, etc): The data to save
    """
    with open(path, 'wb') as file:
        pickle.dump(data, file)

def save_formatted_data(path):
    """
        @brief format and save the data for all subjects
        @param: path (String): Folder to save data to (ending with a /)
    """
    data = {'data': {'ACC':np.empty((0,ACC_SF,3)), 'BVP':np.empty((0,BVP_SF,1)),
            'EDA':np.empty((0,EDA_SF,1)), 'TEMP':np.empty((0,TEMP_SF,1))},
                'labels':[]}
    for x in range(2, 18):
        #subjects 1 and 12 were not included in the published data
        if x != 12:
            cur = get_subject_data(str(x))

            #normalize the data here to avoid normalization with statistical data
            for i in features:
                cur['data'][i] = norm(cur['data'][i])

            #save the data for the individual
            save_data(path+'S'+str(x)+'.pkl', cur)
            #append the subject data to the combined data
            for i in features:
                data['data'][i] = np.append(data['data'][i], cur['data'][i],0)
            data['labels'] = np.append(data['labels'], cur['labels'],0)
    #save combined data
    save_data(path+'All.pkl', data)

def save_statistics(path):
    """
        @breif save statistical analysis of each time step for combined
            and individual subjects
        @param: path (string): Folder to save data to (ending with a /)
    """
    #set up blank dictionary for the data
    data = {'data': {'ACC':np.empty((0,8,3)), 'BVP':np.empty((0,8,1)),
            'EDA':np.empty((0,8,1)), 'TEMP':np.empty((0,8,1))},
                'labels':[]}
    for x in range(2, 18):
        #subjects 1 and 12 were not included in the published data
        if x != 12:
            cur = get_statistics(get_subject_data(str(x)))
            save_data(path+'S'+str(x)+'.pkl', cur)

            for i in features:
                data['data'][i] = np.append(data['data'][i], cur['data'][i],0)
            data['labels'] = np.append(data['labels'], cur['labels'],0)

    save_data(path+'All.pkl', data)

def get_statistics(data):
    """
        @breif Gathers statistical values for the data, these include:
            mean, median, minimum, maximum, standard deviation, skew,
            kurtosis, and interquartile range
        @param data (dictionary): The data to do analysis on
        @return A new array with statistical values for each timestep
    """
    #storage for returning
    temp = {'data':{'ACC':[],'BVP':[],'EDA':[],'TEMP':[]}, 'labels':[]}
    #replace each timestep with statistical analysis
    for x in data['data']:
        #transpose to put each timestep as the first axis
        temp['data'][x] = np.transpose([np.mean(data['data'][x],1),
            np.median(data['data'][x],1), np.amin(data['data'][x],1),
            np.amax(data['data'][x],1), np.std(data['data'][x],1),
            stats.skew(data['data'][x],1), stats.kurtosis(data['data'][x],1),
            stats.iqr(data['data'][x],1)], axes=(1,0,2))
    temp['labels'] = data['labels']

    return temp