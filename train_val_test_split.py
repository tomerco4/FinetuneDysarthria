import pandas as pd
import os
import numpy as np
from sklearn.model_selection import train_test_split


TEST_SUBJECT = "M01"

if __name__ == '__main__':

    data_csv = pd.read_csv("archive/Dysarthria and Non Dysarthria/Torgo/processed_data.csv")

    # Check how many samples of each subject to know who to take for test set
    subjects = [wav_path.split("/")[4] for wav_path in data_csv["Wav_path"]]
    subjects_dict = {subject: len([i for i in subjects if i == subject])  for subject in set(subjects)}
    print(subjects_dict)

    # Take TEST_SUBJECT to a new csv
    test_csv = data_csv[np.array(subjects) == TEST_SUBJECT].reset_index(drop=True)
    train_val_csv = data_csv[np.array(subjects) != TEST_SUBJECT].reset_index(drop=True)

    train_csv, val_csv = train_test_split(train_val_csv, test_size=0.14, random_state=4)

    train_csv.to_csv("archive/Dysarthria and Non Dysarthria/Torgo/processed_data_train.csv", index=False)
    val_csv.to_csv("archive/Dysarthria and Non Dysarthria/Torgo/processed_data_val.csv", index=False)
    test_csv.to_csv("archive/Dysarthria and Non Dysarthria/Torgo/processed_data_test.csv", index=False)


