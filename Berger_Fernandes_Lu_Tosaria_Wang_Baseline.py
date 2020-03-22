# -*- coding: utf-8 -*-
"""
@authors: Michael Berger, Andre Fernandes, Vivian Lu, Sam Tosaria, and Pauline Wang

Functions for Facial Feature Detection Training

"""

#####################################
# imports
#####################################

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Conv2D, Flatten, MaxPool2D, BatchNormalization
from keras.callbacks import ModelCheckpoint
from keras import optimizers, backend
from keras.models import load_model
import time
import pickle

#####################################
# functions
#####################################


def read_train_and_test(path_train, path_test):
    # Reads in the test and train data.

    df_train = pd.read_csv(path_train)
    df_test = pd.read_csv(path_test)

    return df_train, df_test


def preprocess_pixels(df):
    # Processing training data such that each pixel has its own column and is separated from the labels

    df_processed = df.loc[:, 'Image'].T
    df_processed = df_processed.astype(str)
    df_processed = df_processed.str.split(" ", expand=True)

    return df_processed


def create_labels(df, start, end):
    # Seperates image labels from image data

    labels = df.iloc[:, start:end]
    return labels


def normalize_pixels(df):
    # Normalize image pixel to be between 0 and 1

    df_norm = df.astype(int) / 255.
    return df_norm


def separate_non_missing(norm_x_train, labels, normalize=True, inverse=False, index=None):
    # Normalizes the data, in case normalize is set to true. 
    # Otherwise, removes missing values from labels and returns an nd array.

    if normalize: norm_x_train = normalize_pixels(norm_x_train)
    index_none_missing = np.sum(np.isfinite(labels), axis=1) == labels.shape[1]
    if inverse: index_none_missing = ~index
    norm_x_train = np.asarray(norm_x_train.loc[index_none_missing, :])
    labels_no_na = np.asarray(labels.loc[index_none_missing, :])
    if not inverse:
        print('Correct shape for train features?', norm_x_train.shape)
        print('Correct shape for train labels?', labels_no_na.shape)
    return norm_x_train, labels_no_na, index_none_missing


def split_train_and_test(norm_X_train, labels_no_na, percent, seed):
    # This function creates a validation and training set based on
    # the image data, labels, percentage of split and random seed.

    shuffle = np.random.RandomState(seed=seed).permutation(np.arange(norm_X_train.shape[0]))
    X_train, Y = norm_X_train[shuffle], labels_no_na[shuffle]
    train = int(X_train.shape[0] * percent)
    train_data, train_labels = X_train[:train], Y[:train]
    val_data, val_labels = X_train[train:], Y[train:]
    print("validation set shape ({}%):".format(round((1-percent)*100)), val_data.shape, val_labels.shape)
    print("train set shape ({}%):".format(round((percent)*100)), train_data.shape, train_labels.shape)
    return train_data, train_labels, val_data, val_labels


def reshape_image(val_data, val_labels, train_val=None):
    # Reshapes an image for CNN models.

    # convert the "image" column to 96x96 images for plotting the data later
    val_data = np.stack([item.reshape(96, 96) for item in val_data])[:, :, :, np.newaxis]
    # convert the rest of the data to 2140 lists with 30 key facial keypoint positions
    val_labels = np.vstack(val_labels)
    if train_val == "train": print("train feature shape", val_data.shape, "train labels shape", val_labels.shape)
    if train_val == "val": print("val feature shape", val_data.shape, "val labels shape", val_labels.shape)

    return val_data, val_labels


def plot_label_counts(labels):
	# Function to visualize missing labels

    plt.figure(figsize=(8, 10))
    available_df = pd.DataFrame([labels.columns, pd.Series(labels.describe().loc['count'].values)]).T
    available_df.columns = ['features', 'count']
    sns.barplot(x="count", y="features", data=available_df, orient="h", palette="GnBu_d").set_title(
        "Features and Corresponding Counts of Available Data (Train Set)")


def rmse(y_true, y_pred):
	# root mean squared error
    return backend.sqrt(backend.mean(backend.square(y_pred - y_true)))


def r_square(y_true, y_pred):
	# R^2
    SS_res =  backend.sum(backend.square(y_true - y_pred))
    SS_tot = backend.sum(backend.square(y_true - backend.mean(y_true)))
    return abs(1 - SS_res/(SS_tot + backend.epsilon()))


def model_evaluation(model, baseline: False, epochs):
	# Prints some model training statistics 
	# as well as plots the performance per epoch

    print("Below are the model performance plots after {} epochs:".format(epochs))
    if not baseline: baseline = model
    # plot training curve for R^2 (beware of scale, starts very low negative)
    plt.ylim(0.5, max(model.history['val_r_square'])+.2)
    plt.xlim(0, epochs)
    plt.plot(model.history['r_square'])
    plt.plot(model.history['val_r_square'])
    plt.title('Train and Test R^2')
    plt.ylabel('R^2')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()

    # plot training curve for rmse
    plt.ylim(0, max(model.history['rmse'])+10)
    plt.xlim(0, epochs)
    plt.plot(model.history['rmse'])
    plt.plot(model.history['val_rmse'])
    plt.title('Train and Test RMSE')
    plt.ylabel('RMSE')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()

    print('Validation RMSE:', model.history['val_rmse'][-1])
    print('Performance increase relative to baseline: ' + \
          str(round(-(model.history['val_rmse'][-1] / baseline.history['val_rmse'][-1] - 1) * 100, 2)) + '%')
    clear_keras_graph()


def run_baseline_model(train_data, train_labels, val_data, val_labels, epochs, lr,  baseline=False, seed=123, save=False, path=None, mc_path=None):
	# Baseline model

    np.random.seed(seed)
    base_model = Sequential()  # instantiate a base sequential model
    base_model.add(Dense(30, input_shape=(96 * 96,), activation="relu"))  # CHANGE INPUT_DIM to INPUT_SHAPE
    base_model.add(Dense(30))  # the output has no activation function so it's a regression problem
    print(base_model.summary())
    start_time = time.time()
    # sgd = optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    # Optimizer adam, same as Sam's
    adam = optimizers.Adam(lr=lr, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
    # base_model.compile(optimizer=sgd, loss='mean_squared_error', metrics=['accuracy'])#root mean square error b/c accuracy
    base_model.compile(optimizer=adam, loss='mse', metrics=['mse', r_square, rmse])
    mc = ModelCheckpoint(mc_path, monitor='val_rmse', mode='min', verbose=1, save_best_only=True)

    if save:
        base_model.fit(train_data.reshape(train_labels.shape[0], -1), train_labels,
                       validation_data=[val_data.reshape(val_labels.shape[0], -1), val_labels],
                       shuffle=False, epochs=epochs, batch_size=20, callbacks=[mc])
        base_model.save(path)
    else:
        model = base_model.fit(train_data.reshape(train_labels.shape[0], -1), train_labels,
                                 validation_data=[val_data.reshape(val_labels.shape[0], -1), val_labels],
                                 shuffle=False, epochs=epochs, batch_size=20, callbacks=[mc])
        model_evaluation(model, baseline, epochs)
        base_model = [base_model, model]

    total_time = time.time() - start_time
    print("The model took {} seconds to run".format(round(total_time, 3)))
    return base_model, total_time


def run_cnn_model(train_data, train_labels, val_data, val_labels, epochs, lr,  baseline=False, seed=123, save=False, path=None, mc_path=None):
	# 1st extended layer

    np.random.seed(seed)
    model2 = Sequential()

    model2.add(Conv2D(32, kernel_size=(3, 3), strides=(1, 1),
                      input_shape=(96, 96, 1), activation='relu'))
    model2.add(MaxPool2D(pool_size=(2, 2)))

    model2.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
    model2.add(MaxPool2D(pool_size=(2, 2)))

    model2.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
    model2.add(MaxPool2D(pool_size=(2, 2)))

    model2.add(Flatten())
    model2.add(Dense(100))
    model2.add(Activation('relu'))
    model2.add(Dense(100))
    model2.add(Activation('softmax'))
    model2.add(Dense(30))

    print(model2.summary())
    start_time = time.time()

    adam = optimizers.Adam(lr=lr, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
    model2.compile(loss='mse', optimizer=adam, metrics=[r_square, rmse])
    mc = ModelCheckpoint(mc_path, monitor='val_rmse', mode='min', verbose=1, save_best_only=True)

    if save:
        model2.fit(train_data, train_labels, validation_data=[val_data, val_labels], shuffle=False,
                   epochs=epochs, batch_size=20, callbacks=[mc])
        model2.save(path)
    else:
        ext_model = model2.fit(train_data, train_labels, validation_data=[val_data, val_labels], shuffle=False,
                               epochs=epochs, batch_size=20, callbacks=[mc])
        model_evaluation(ext_model, baseline, epochs)
        model2 = [model2, ext_model]

    total_time = time.time() - start_time
    print("The cnn model took {} seconds to run".format(round(total_time, 3)))
    return model2, total_time


def clear_keras_graph():
	# clears model graph 

    backend.clear_session()
    return


def generate_new_images(train_data, rotation_range=40, width_shift_range=0.2,
                        height_shift_range=0.2, shear_range=0.2, zoom_range=0.2,
                        horizontal_flip=True, fill_mode='nearest', batch_size=1,
                        seed=123):
    
    #########################################
    # code used to run this in kickstarter file
    #
    # import pandas as pd
    # import numpy as np
    # aug_train_data = facial_functions.generate_new_images(train_data, batch_size=10).astype(int)
    # aug_train_data.to_csv(os.path.join(workDir, "data", "augmented_train_data.csv"))
    #########################################

    start_time = time.time()
    from tensorflow.keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
    datagen = ImageDataGenerator(
        rotation_range=rotation_range,
        width_shift_range=width_shift_range,
        height_shift_range=height_shift_range,
        shear_range=shear_range,
        zoom_range=zoom_range,
        horizontal_flip=horizontal_flip,
        fill_mode=fill_mode)
    #new_images = []
    new_images = pd.DataFrame()
    for img in train_data:
        img_reshape = np.reshape((np.array(img * 255).astype(int)), (1, 96, 96, 1))
        i = 0
        for batch in datagen.flow(img_reshape, batch_size=batch_size, seed=seed):
            new_images = new_images.append((pd.Series(np.reshape(batch[0].astype(int), (9216,)))), ignore_index=True)
            #new_images = np.append(new_images, np.reshape(batch[0].astype(int), (9216,)))
            #new_images = np.concatenate(([new_images], [np.reshape(batch[0].astype(int), (9216,))]))
            i += 1
            if i % batch_size == 0:
                break
    total_time = time.time() - start_time
    print("The data augmentation generator took {} seconds to run".format(round(total_time, 3)))
    return new_images


def run_regularized_cnn_model(train_data, train_labels, val_data, val_labels, epochs, lr, dropout=.25, baseline=False,
                              seed=123, batch_size=20, save=False, path=None, mc_path=None):
    # second extended model

    np.random.seed(seed)
    model2 = Sequential()

    model2.add(Conv2D(32, kernel_size=(3, 3), strides=(1, 1),
                      input_shape=(96, 96, 1), activation='relu'))
    model2.add(MaxPool2D(pool_size=(2, 2)))
    model2.add(BatchNormalization())
    model2.add(Dropout(dropout))

    model2.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
    model2.add(MaxPool2D(pool_size=(2, 2)))
    model2.add(BatchNormalization())
    model2.add(Dropout(dropout))

    model2.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
    model2.add(MaxPool2D(pool_size=(2, 2)))
    model2.add(BatchNormalization())
    model2.add(Dropout(dropout))

    model2.add(Flatten())
    model2.add(Dense(128, activation='relu'))
    model2.add(BatchNormalization())
    model2.add(Dropout(dropout))

    model2.add(Dense(64, activation='relu'))
    model2.add(BatchNormalization())
    model2.add(Dropout(dropout))

    model2.add(Dense(30))

    print(model2.summary())
    start_time = time.time()

    adam = optimizers.Adam(lr=lr)
    model2.compile(loss='mse', optimizer=adam, metrics=[r_square, rmse])
    mc = ModelCheckpoint(mc_path, monitor='val_rmse', mode='min', verbose=1, save_best_only=True)

    if save:
        model2.fit(train_data, train_labels, validation_data=[val_data, val_labels], shuffle=True,
                   epochs=epochs, batch_size=batch_size, callbacks=[mc])
        model2.save(path)
    else:
        ext_model = model2.fit(train_data, train_labels, validation_data=[val_data, val_labels], shuffle=True,
                               epochs=epochs, batch_size=batch_size)
        model_evaluation(ext_model, baseline, epochs)
        model2 = [model2, ext_model]

    total_time = time.time() - start_time
    print("The regularized cnn model took {} seconds to run".format(round(total_time, 3)))
    return model2, total_time


def read_and_process_imputed_data(df_train_x, df_train_y):
    # calls function to read in data, normalize it, create labels
    # filter for missing values

    df_train_x, df_train_y = read_train_and_test(df_train_x, df_train_y)

    kaggle_train = normalize_pixels(np.asarray(df_train_x))
    labels = create_labels(df_train_y, 0, 30)

    kaggle_train, labels = filter_missing(kaggle_train, labels)

    return kaggle_train, np.asarray(labels)


def load_imputed_labels_for_na(kaggle_train, kaggle_labels, path_labels, index=None):
    # loading the image data imputed with face net package

    imputed_labels = pd.read_csv(path_labels)
    missing_x, missing_y, _ = separate_non_missing(kaggle_train, kaggle_labels, normalize=True, inverse=True, index=index)

    #shaped_missing_x, shaped_missing_y = reshape_image(missing_x, missing_y)
    missing_features, missing_labels = filter_missing(missing_x, np.asarray(imputed_labels[~index]))
    return missing_features, missing_labels


def append_imputed_data(list_train_features, list_train_labels):
    # appending imputed labels to our existing labels data

    appended_features = np.concatenate(list_train_features, 0)
    appended_labels = np.concatenate(list_train_labels, 0)

    print("new feature shape: ", appended_features.shape)
    print("new label shape: ", appended_labels.shape)

    return appended_features, appended_labels


def filter_missing(df_features, df_labels):
    # filter out missing data

    completeness_index = np.sum(np.isfinite(df_labels), axis=1) == 30

    return df_features[completeness_index], df_labels[completeness_index]


def get_image_and_save(index, data_df):
    # Packages needed 
    from PIL import ImagePIL
    import PIL.ImageOps
    # This takes in the images, reshapes, and spits out the image
    a = np.array(data_df.iloc[index,:].astype(int))
    b = np.interp(a, (a.min(), a.max()), (0, +1)) #rescale 
    mat=np.reshape(b, (96,96))
    img = ImagePIL.fromarray(np.uint8(mat * 255) , 'L')
    return img


def make_points(labels_fr, index):
    # Reduces imputed points to 30 labels

    kaggle_keys = ['left_eyebrow','right_eyebrow','nose_tip','left_eye','right_eye','top_lip','bottom_lip']
    new_row_label={'left_eye_center_x': 0, 'left_eye_center_y': 0, 'right_eye_center_x': 0,
       'right_eye_center_y':0, 'left_eye_inner_corner_x':0,
       'left_eye_inner_corner_y':0, 'left_eye_outer_corner_x':0,
       'left_eye_outer_corner_y':0, 'right_eye_inner_corner_x':0,
       'right_eye_inner_corner_y':0, 'right_eye_outer_corner_x':0,
       'right_eye_outer_corner_y':0, 'left_eyebrow_inner_end_x':0,
       'left_eyebrow_inner_end_y':0, 'left_eyebrow_outer_end_x':0,
       'left_eyebrow_outer_end_y':0, 'right_eyebrow_inner_end_x':0,
       'right_eyebrow_inner_end_y':0, 'right_eyebrow_outer_end_x':0,
       'right_eyebrow_outer_end_y':0, 'nose_tip_x':0, 'nose_tip_y':0,
       'mouth_left_corner_x':0, 'mouth_left_corner_y':0, 'mouth_right_corner_x':0,
       'mouth_right_corner_y':0, 'mouth_center_top_lip_x':0,
       'mouth_center_top_lip_y':0, 'mouth_center_bottom_lip_x':0,
       'mouth_center_bottom_lip_y':0, 'row_index':int(index)}
    for face_landmarks in labels_fr: 
        for facial_feature in kaggle_keys: 
            df_feature = pd.DataFrame(face_landmarks[facial_feature])
            df_feature.columns=['x','y']
            if (facial_feature=='left_eyebrow'): 
                left_inner = df_feature.sort_values(by='x', ascending=True).reset_index(drop=True).loc[0,:]
                left_outer = df_feature.sort_values(by='x', ascending=False).reset_index(drop=True).loc[0,:]
                new_row_label['left_eyebrow_inner_end_y']=left_inner['y']
                new_row_label['left_eyebrow_inner_end_x']=left_inner['x']
                new_row_label['left_eyebrow_outer_end_y']=left_outer['y']
                new_row_label['left_eyebrow_outer_end_x']=left_outer['x']
            elif (facial_feature=='right_eyebrow'): 
                right_inner = df_feature.sort_values(by='x', ascending=True).reset_index(drop=True).loc[0,:]
                right_outer = df_feature.sort_values(by='x', ascending=False).reset_index(drop=True).loc[0,:]
                new_row_label['right_eyebrow_inner_end_y']=right_inner['y']
                new_row_label['right_eyebrow_inner_end_x']=right_inner['x']
                new_row_label['right_eyebrow_outer_end_y']=right_outer['y']
                new_row_label['right_eyebrow_outer_end_x']=right_outer['x']
            elif (facial_feature=='nose_tip'): 
                df_subfeature=df_feature.sort_values(by='y', ascending=False).reset_index(drop=True).loc[0,:]
                new_row_label['nose_tip_x']=df_subfeature['x']
                new_row_label['nose_tip_y']=df_subfeature['y']
            elif (facial_feature=='left_eye'):
                left_inner = df_feature.sort_values(by='x', ascending=True).reset_index(drop=True).loc[0,:]
                left_outer = df_feature.sort_values(by='x', ascending=False).reset_index(drop=True).loc[0,:]
                mu_x= (left_inner['x']+left_outer['x'])/2
                mu_y= (left_inner['y']+left_outer['y'])/2 
                new_row_label['left_eye_center_x']=mu_x
                new_row_label['left_eye_center_y']=mu_y
                new_row_label['left_eye_inner_corner_x']=left_inner['x']
                new_row_label['left_eye_inner_corner_y']=left_inner['y']
                new_row_label['left_eye_outer_corner_x']=left_outer['x']
                new_row_label['left_eye_outer_corner_y']=left_outer['y']
            elif (facial_feature=='right_eye'): 
                right_inner = df_feature.sort_values(by='x', ascending=True).reset_index(drop=True).loc[0,:]
                right_outer = df_feature.sort_values(by='x', ascending=False).reset_index(drop=True).loc[0,:]
                mu_x= (right_inner['x']+right_outer['x'])/2
                mu_y= (right_inner['y']+right_outer['y'])/2 
                new_row_label['right_eye_center_x']=mu_x
                new_row_label['right_eye_center_y']=mu_y
                new_row_label['right_eye_inner_corner_x']=right_inner['x']
                new_row_label['right_eye_inner_corner_y']=right_inner['y']
                new_row_label['right_eye_outer_corner_x']=right_outer['x']
                new_row_label['right_eye_outer_corner_y']=right_outer['y']                
            elif (facial_feature=='top_lip'): 
                df_subfeature = df_feature.sort_values(by='y', ascending=False).reset_index(drop=True)
                toplip_x_mean=np.nanmean([min(df_subfeature['x']), max(df_subfeature['x'])])
                subdf = df_subfeature[(df_subfeature['x']<=(toplip_x_mean+5)) & (df_subfeature['x']>=(toplip_x_mean-5))].sort_values(by='y').reset_index(drop=True).loc[0,:]
                getnosetip = pd.DataFrame(face_landmarks['nose_tip']) 
                getnosetip.columns=['x','y']
                getnosetip_sub = getnosetip.sort_values(by='y', ascending=False).reset_index(drop=True).loc[0,:]
                new_row_label['mouth_center_top_lip_x']=getnosetip_sub['x']
                new_row_label['mouth_center_top_lip_y']=subdf['y']
            else: 
                # this is bottom_lip 
                right = df_feature.sort_values(by='x', ascending=True).reset_index(drop=True).loc[0,:]
                left = df_feature.sort_values(by='x', ascending=False).reset_index(drop=True).loc[0,:]
                df_subfeature = df_feature.sort_values(by='y', ascending=False).reset_index(drop=True)
                bottomlip_x_mean = np.nanmean([min(df_subfeature['x']), max(df_subfeature['x'])])
                subdf = df_subfeature[(df_subfeature['x']<=(bottomlip_x_mean+5)) & (df_subfeature['x']>=(bottomlip_x_mean-5))].sort_values(by='y', ascending=False).reset_index(drop=True).loc[0,:]
                getnosetip = pd.DataFrame(face_landmarks['nose_tip']) 
                getnosetip.columns=['x','y']
                getnosetip_sub = getnosetip.sort_values(by='y', ascending=False).reset_index(drop=True).loc[0,:]
                new_row_label['mouth_left_corner_x']=left['x']
                new_row_label['mouth_left_corner_y']=left['y']
                new_row_label['mouth_right_corner_x']=right['x']
                new_row_label['mouth_right_corner_y']=right['y']
                new_row_label['mouth_center_bottom_lip_x']=getnosetip_sub['x']
                new_row_label['mouth_center_bottom_lip_y']=subdf['y']
    return pd.Series(new_row_label)


########################################################################################################################
    # OPTIONAL: Code to run a classification model before the regression


def creating_balanced_dataset(train, labels):

    # This function creates a dataset with 50% of the observations
    # having no misisng labels and the 50% at least one missing label.
    # It takes the first number of observations for the missing label
    # vector.
    
    subset_vector = labels.notna().sum(axis = 1) != 30
    
    labels_missing = labels.loc[subset_vector,:]
    labels_full = labels.loc[~subset_vector,:]
    
    train_missing = train.loc[subset_vector,:]
    train_full = train.loc[~subset_vector,:]
      
    train = train_full.append(train_missing.iloc[0:train_full.shape[0],:])
    labels = labels_full.append(labels_missing.iloc[0:labels_full.shape[0],:])
    
    return train, labels


def class_reg_labels_with_zero(labels):

    # Replaces missing values in the labels with 0. Run this function on the pandas labels dataframe.

    return labels.fillna(0)


def classification_labels(label_data):
    # This function takes the created labels data after the cross validation split,
    # and allocates 0 or 1, depending on whether a value >0
    # is present. It returns the new dataframe for classification.

    labels_clasf = label_data.copy()
    labels_clasf[labels_clasf > 0] = 1
    
    return np.asarray(labels_clasf).astype(int)


def classification_data_preparation(train, labels):
    # This function performs the transformations on the kaggle_train and
    # labels data in order to create classification train and validation
    # data for the classification CNN.
    # It returns training image and labels data as well as validation image and labels data.

    # We create a balanced dataset with 50% of the images having all facial keypoints and 
    # 50% having at leats one missing keypoint
    train_balanced, labels_balanced = creating_balanced_dataset(train, labels)

    # We need to adjust the labels to code missing with zero
    labels_adj = class_reg_labels_with_zero(labels_balanced)
    norm_x_train, class_labels, blank = separate_non_missing(train_balanced, labels_adj, normalize=True)

    # Finally we split the data in test and validation sets. The combined model is tested on this validation set
    x_train, y_train, x_val, y_val = split_train_and_test(norm_x_train, class_labels, percent=.8, seed=123)
    x_val, y_val = reshape_image(x_val, y_val)

    # We finally prepare the labels to be binary
    class_labels = classification_labels(y_train)
    class_val = classification_labels(y_val)

    return x_train, class_labels, x_val, class_val


def run_classification_model(train_data, train_labels, val_data, val_labels, epochs, lr, dropout=.25,
                            seed=123, batch_size=20, save=False, path=None):
    # This model is a copy of our classification model. It is trained on the x- and
    # y-coordinates per facial keypoint. It returns the fitted model.
    #
    # The changes follow the following idea:
    # https://www.depends-on-the-definition.com/guide-to-multi-label-classification-with-neural-networks/
    #
    # The only difference is the input labels (as list of 0s and 1s, making it a mult-label
    # classification problem), the loss function and the last layer. The last layer has a
    # sigmoid activation function. Hence, it is assumed that each coordinate is independent
    # (even x- and y-coordinates of one label, which makes this a conservative assumption).
    # In addition, the loss function is binary_crossentropy, which penalizes each output node
    # independently. In order to handle the independence, another function will map 0s and 1s
    # depending on whether the classification model predicts a high probability of missing (0)
    # for both x- and y-coordinates. Otherwise, the function will set present (1).
    #
    # The model is used to predict for each keypoint coordinate whether it is present (1)
    # or missing (0).
    
    np.random.seed(seed)
    classf_model = Sequential()

    classf_model.add(Conv2D(32, kernel_size=(3, 3), strides=(1, 1),
                      input_shape=(96, 96, 1), activation='relu'))
    classf_model.add(MaxPool2D(pool_size=(2, 2)))
    classf_model.add(BatchNormalization())
    classf_model.add(Dropout(dropout))

    classf_model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
    classf_model.add(MaxPool2D(pool_size=(2, 2)))
    classf_model.add(BatchNormalization())
    classf_model.add(Dropout(dropout))

    classf_model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
    classf_model.add(MaxPool2D(pool_size=(2, 2)))
    classf_model.add(BatchNormalization())
    classf_model.add(Dropout(dropout))

    classf_model.add(Flatten())
    classf_model.add(Dense(128, activation='relu'))
    classf_model.add(BatchNormalization())
    classf_model.add(Dropout(dropout))

    classf_model.add(Dense(64, activation='relu'))
    classf_model.add(BatchNormalization())
    classf_model.add(Dropout(dropout))
  
    classf_model.add(Dense(30, activation = 'sigmoid'))

    print(classf_model.summary())
    start_time = time.time()

    adam = optimizers.Adam(lr=lr)
    classf_model.compile(loss='binary_crossentropy', optimizer=adam, metrics=['accuracy'])
    train_data, train_labels = reshape_image(train_data, train_labels)
    cl_model_fitted = classf_model.fit(train_data, train_labels, validation_data=[val_data, val_labels], shuffle=False,
                           epochs=epochs, batch_size=batch_size)

    total_time = time.time() - start_time
    print("The classification cnn model took {} seconds to run".format(round(total_time, 3)))

    if save:
        classf_model.fit(train_data, train_labels, validation_data=[val_data, val_labels], shuffle=False,
                           epochs=epochs, batch_size=batch_size)
        classf_model.save(path)
    
    return classf_model


def adjusting_keypoint_prediction(model, val_data, threshold=0.1):
    # The classification model assumes independence. In addition, each coordinate is
    # treated as class. But a facial keypoint, which is missing, has neither x- nor
    # y-coordinates. Hence, in case the model predicts for both coordinates of a label
    # a low probability (below threshold) of the cooridnate being present, we assume
    # that the label is missing and set the coordinates to 0, otherwise we set both to 1.
    
    predicted_matrix = model.predict(val_data)
    
    for obs in range(predicted_matrix.shape[0]):
        for pred_coord in range(0, predicted_matrix.shape[1], 2):
            if predicted_matrix[obs, pred_coord] > threshold and predicted_matrix[obs, pred_coord+1] > threshold:
                predicted_matrix[obs, pred_coord] = 1
                predicted_matrix[obs, pred_coord+1] = 1
            else:
                predicted_matrix[obs, pred_coord] = 0
                predicted_matrix[obs, pred_coord+1] = 0
    
    return predicted_matrix


def rmse_metric_class_reg(y_true, y_pred):
    # Same as previous rmse function, but defined using numpy.

    return np.sqrt(np.mean(np.square(y_pred - y_true)))


def evaluate_clas_reg_rmse(class_model_path, reg_model_path, val_data, val_labels, metric = rmse_metric_class_reg, threshold = 0.1):
    # This function takes the regression and classification model as input, as well as
    # validation data and labels (where the missing keypoints have x- and y-coordinates of 0)
    # and a metric and a threshold for when a facial keypoint should be considered present.
    # It conducts an element-wise multiplication of the predicted classes and regression values.
    # It then evaluates the prediction on the provided validation data. It returns the result

    reg_model = load_model(reg_model_path, custom_objects={'r_square': r_square, 'rmse': rmse})
    regression_prediction = reg_model.predict(val_data)
    class_model = load_model(class_model_path, custom_objects={'r_square': r_square, 'rmse': rmse})
    class_val_matrix = adjusting_keypoint_prediction(class_model, val_data, threshold)
    predicted_labels = np.multiply(class_val_matrix, regression_prediction)
    
    return metric(val_labels, predicted_labels)


def ensemble_predict(class_model, reg_model, val_data, threshold = 0.5):
    # This function takes a classification model and a regression model,
    # adjusts the classification of the keypoints based on a threshold and
    # then creates predictions on an image dataset with both models.
    # The models predictions are then combined via elementwise
    # multiplication.
    #
    # The matrix of predictions for the keypoints is returned with

    val_data = np.stack([item.reshape(96, 96) for item in np.asarray(val_data)])[:, :, :, np.newaxis]
    regression_prediction = reg_model.predict(val_data)
    class_val_matrix = adjusting_keypoint_prediction(class_model, val_data, threshold)
    predicted_labels = np.multiply(class_val_matrix, regression_prediction)

    return predicted_labels


def create_kaggle_prediction(labels, model_path, df_test, in_path_submission_sheet = 'IdLookupTable.csv', out_path = 'kaggle_submission.csv', baseline=False):
    #    This function takes the kaggle test set, the labels from the training and a regression model
    #    to predict on the data.
    #
    #    Based on the supplied path to the submisison kaggle sheet, it loads the sheet, extracts
    #    from the prediction matrix the keypoint coordinates needed to be predicted and saves the
    #    extracted coordinates into the kaggle submission format. It saves the created dataframe into
    #    a csv file at the output_path supplied. It controls for the predicted coordinates being between
    #    [0, 96].

    submission = pd.read_csv(in_path_submission_sheet)
    kaggle_test = preprocess_pixels(df_test)
    kaggle_test = normalize_pixels(kaggle_test)

    model = load_model(model_path, custom_objects={'r_square': r_square, 'rmse': rmse})

    if not(baseline): 
        val_data = np.stack([item.reshape(96, 96) for item in np.asarray(kaggle_test)])[:, :, :, np.newaxis]
    else: 
        val_data = kaggle_test

    regression_prediction = model.predict(val_data)

    keypoints = list(labels.columns)

    result = list()

    for row in range(submission.shape[0]):
      id = submission.iloc[row,1]-1
      keypoint_prediction = regression_prediction[id]
      label = submission.iloc[row,2]
      index = keypoints.index(label)
      prediction_coordinate = min(96, max(0, keypoint_prediction[index]))
      submission.iloc[row, 3] = prediction_coordinate


    submission = submission.drop(['ImageId', 'FeatureName'], axis=1)
    submission.to_csv(out_path, index=False)


########################################################################################################################
    # OPTIONAL: Starter code for running image flips. This function was abandoned and is not finished. It correctly
    # outputs a flipped image, but the column labels need to be flipped in order to be used for training
    # (i.e. "left eye" coordinates become "right eye" coordinates)


def flip_horizontally(df, labels, start=0, pickle_feature=None, pickle_label=None):
    new_images = pd.DataFrame()
    if pickle_feature: new_images = pickle.load(open(pickle_feature, 'rb'))

    new_labels = pd.DataFrame()
    if pickle_label: new_labels = pickle.load(open(pickle_label, 'rb'))
    transformation_vector = [96, 0]*15
    transformation_vector = np.asarray(np.reshape(transformation_vector, (30,)))

    for img in range(start, df.shape[0]):
        new_images = new_images.append((pd.Series(np.reshape(np.fliplr(df[img].reshape(96, 96)), (9216,)))), ignore_index=True)
        tmp_labels = np.reshape(abs(labels[img].copy() - transformation_vector), (30,))
        new_labels = new_labels.append((pd.Series(tmp_labels)),
                                       ignore_index=True)

    return np.asarray(new_images), np.asarray(new_labels)