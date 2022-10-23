from operator import mod
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
import numpy as np


class AdultDataset(Dataset):
    def __init__(self, mode, ont_hot=True, root_path="C:\\Develop\\ML_Elements\\dataset\\adult") -> None:
        super().__init__()
        if mode == "train":
            with open(f"{root_path}\\adult.data") as fin:
                lines = fin.read().split("\n")[:-1]
        if mode == "val":
            with open(f"{root_path}\\adult.test") as fin:
                lines = fin.read().split("\n")[:-1]
        self.mode = mode
        self.ont_hot = ont_hot
        self.data = []
        for line in lines:
            self.data.append(line.split(", "))
        self.data, self.label = self.pre_process(self.data)

    def pre_process(self, data):
        # age
        ages = []
        for idx in range(len(data)):
            ages.append(float(data[idx][0]))
        ages = np.array(ages)
        ages = (ages-ages.mean())/ages.std()
        ages = np.expand_dims(ages, axis=-1)

        # workclass
        workclass = []
        workclass_category = ["Private", "Self-emp-not-inc", "Self-emp-inc",
                              "Federal-gov", "Local-gov", "State-gov", "Without-pay", "Never-worked"]
        for idx in range(len(data)):
            if data[idx][1] == "?":  # uniform
                workclass.append(
                    np.ones(len(workclass_category))/len(workclass_category))
            else:  # one-hot
                workclass.append(np.squeeze(np.eye(len(workclass_category))[
                    workclass_category.index(data[idx][1])]))
        workclass = np.stack(workclass)

        # fnlwgt
        fnlwgt = []
        for idx in range(len(data)):
            fnlwgt.append(float(data[idx][2]))
        fnlwgt = np.array(fnlwgt)
        fnlwgt = (fnlwgt-fnlwgt.mean())/fnlwgt.std()
        fnlwgt = np.expand_dims(fnlwgt, axis=-1)

        # education
        education = []
        education_category = ["Bachelors", "Some-college", "11th", "HS-grad", "Prof-school", "Assoc-acdm",
                              "Assoc-voc", "9th", "7th-8th", "12th", "Masters", "1st-4th", "10th", "Doctorate", "5th-6th", "Preschool"]
        for idx in range(len(data)):
            # one-hot
            education.append(np.squeeze(np.eye(len(education_category))[
                education_category.index(data[idx][3])]))
        education = np.stack(education)

        # education-num
        education_num = []
        for idx in range(len(data)):
            education_num.append(float(data[idx][4]))
        education_num = np.array(education_num)
        education_num = (education_num-education_num.mean()) / \
            education_num.std()
        education_num = np.expand_dims(education_num, axis=-1)

        # marital-status
        marital_status = []
        marital_status_category = ["Married-civ-spouse", "Divorced", "Never-married",
                                   "Separated", "Widowed", "Married-spouse-absent", "Married-AF-spouse"]
        for idx in range(len(data)):
            # one-hot
            marital_status.append(np.squeeze(np.eye(len(marital_status_category))[
                marital_status_category.index(data[idx][5])]))
        marital_status = np.stack(marital_status)

        # occupation
        occupation = []
        occupation_category = ["Tech-support", "Craft-repair", "Other-service", "Sales", "Exec-managerial", "Prof-specialty", "Handlers-cleaners",
                               "Machine-op-inspct", "Adm-clerical", "Farming-fishing", "Transport-moving", "Priv-house-serv", "Protective-serv", "Armed-Forces"]
        for idx in range(len(data)):
            if data[idx][6] == "?":  # uniform
                occupation.append(
                    np.ones(len(occupation_category))/len(occupation_category))
            else:  # one-hot
                occupation.append(np.squeeze(np.eye(len(occupation_category))[
                    occupation_category.index(data[idx][6])]))
        occupation = np.stack(occupation)

        # relationship
        relationship = []
        relationship_category = [
            "Wife", "Own-child", "Husband", "Not-in-family", "Other-relative", "Unmarried"]
        for idx in range(len(data)):
            # one-hot
            relationship.append(np.squeeze(np.eye(len(relationship_category))[
                relationship_category.index(data[idx][7])]))
        relationship = np.stack(relationship)

        # race
        race = []
        race_category = ["White", "Asian-Pac-Islander",
                         "Amer-Indian-Eskimo", "Other", "Black"]
        for idx in range(len(data)):
            # one-hot
            race.append(np.squeeze(np.eye(len(race_category))[
                race_category.index(data[idx][8])]))
        race = np.stack(race)

        # sex
        sex = []
        sex_category = ["Female", "Male"]
        for idx in range(len(data)):
            # one-hot
            sex.append(np.squeeze(np.eye(len(sex_category))[
                sex_category.index(data[idx][9])]))
        sex = np.stack(sex)

        # capital-gain
        capital_gain = []
        for idx in range(len(data)):
            capital_gain.append(float(data[idx][10]))
        capital_gain = np.array(capital_gain)
        capital_gain = (capital_gain-capital_gain.mean()) / \
            capital_gain.std()
        capital_gain = np.expand_dims(capital_gain, axis=-1)

        # capital-loss
        capital_loss = []
        for idx in range(len(data)):
            capital_loss.append(float(data[idx][11]))
        capital_loss = np.array(capital_loss)
        capital_loss = (capital_loss-capital_loss.mean()) / \
            capital_loss.std()
        capital_loss = np.expand_dims(capital_loss, axis=-1)

        # hours-per-week
        hours_per_week = []
        for idx in range(len(data)):
            hours_per_week.append(float(data[idx][12]))
        hours_per_week = np.array(hours_per_week)
        hours_per_week = (hours_per_week-hours_per_week.mean()) / \
            hours_per_week.std()
        hours_per_week = np.expand_dims(hours_per_week, axis=-1)

        # native-country
        native_country = []
        native_country_category = ["United-States", "Cambodia", "England", "Puerto-Rico", "Canada", "Germany", "Outlying-US(Guam-USVI-etc)", "India", "Japan", "Greece", "South", "China", "Cuba", "Iran", "Honduras", "Philippines", "Italy", "Poland", "Jamaica", "Vietnam", "Mexico",
                                   "Portugal", "Ireland", "France", "Dominican-Republic", "Laos", "Ecuador", "Taiwan", "Haiti", "Columbia", "Hungary", "Guatemala", "Nicaragua", "Scotland", "Thailand", "Yugoslavia", "El-Salvador", "Trinadad&Tobago", "Peru", "Hong", "Holand-Netherlands"]
        for idx in range(len(data)):
            if data[idx][13] == "?":  # uniform
                native_country.append(
                    np.ones(len(native_country_category))/len(native_country_category))
            else:  # one-hot
                native_country.append(np.squeeze(np.eye(len(native_country_category))[
                    native_country_category.index(data[idx][13])]))
        native_country = np.stack(native_country)

        new_data = np.concatenate([ages, workclass, fnlwgt, education, education_num, marital_status, occupation,
                                   relationship, race, sex, capital_gain, capital_loss, hours_per_week, native_country], axis=-1)

        # label
        label = []
        for idx in range(len(data)):
            if self.mode == "train":
                label_str = data[idx][14]
            if self.mode == "val":
                label_str = data[idx][14][:-1]
            if label_str == ">50K":
                label.append(1)
            else:
                label.append(0)
        label = np.stack(label)
        native_country = np.stack(native_country)

        new_data = new_data.astype(np.float32)
        label = label.astype(np.int32)
        return new_data, label

    def check_missing(self):  # before preprocess
        bucket = [0]*15
        for i in range(len(self.data)):
            data_piece = self.data[i]
            for j in range(len(data_piece)):
                if data_piece[j] == "?":
                    bucket[j] += 1
        print(bucket)
        cnt_2_7 = 0
        for i in range(len(self.data)):
            data_piece = self.data[i]
            if data_piece[1] == "?" and data_piece[6] == "?":
                cnt_2_7 += 1
        print(f"cnt_2_7: {cnt_2_7}")
        cnt_2_14 = 0
        for i in range(len(self.data)):
            data_piece = self.data[i]
            if data_piece[1] == "?" and data_piece[13] == "?":
                cnt_2_14 += 1
        print(f"cnt_2_14: {cnt_2_14}")
        cnt_7_14 = 0
        for i in range(len(self.data)):
            data_piece = self.data[i]
            if data_piece[1] == "?" and data_piece[13] == "?":
                cnt_7_14 += 1
        print(f"cnt_7_14: {cnt_7_14}")
        print(self.__len__())

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index], self.label[index]


class AvilaDataset(Dataset):
    def __init__(self, mode, root_path="C:\\Develop\\ML_Elements\\dataset\\AvilaDataset") -> None:
        super().__init__()
        if mode == "train":
            with open(f"{root_path}\\avila-tr.txt") as fin:
                lines = fin.read().split("\n")[:-1]
        if mode == "val":
            with open(f"{root_path}\\avila-ts.txt") as fin:
                lines = fin.read().split("\n")[:-1]
        self.mode = mode

        self.data = []
        self.label = []
        label_category = ["A", "B", "C", "D", "E",
                          "F", "G", "H", "I", "W", "X", "Y"]
        for line in lines:
            tmp = line.split(",")
            vals = [0]*(len(tmp)-1)
            for idx in range(len(tmp)-1):
                vals[idx] = float(tmp[idx])
            vals = np.array(vals)
            self.data.append(vals)
            self.label.append(label_category.index(tmp[-1]))
        self.data = np.stack(self.data)
        self.label = np.array(self.label)
        self.data = self.pre_process(self.data)

    def pre_process(self, data: np.ndarray):
        # data = data.mean(axis=0)
        data = (data-data.mean(axis=0, keepdims=True)) / \
            data.std(axis=0, keepdims=True)
        return data.astype(np.float32)

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, index):
        data = self.data[index]
        label = self.label[index]
        return data, label


# if __name__ == "__main__":
#     adult_dataset = AdultDataset(mode="train")
#     print((adult_dataset.label == 0).sum())
#     print((adult_dataset.label == 1).sum())
#     adult_dataset = AdultDataset(mode="val")
#     print((adult_dataset.label == 0).sum())
#     print((adult_dataset.label == 1).sum())
#     # adult_dataset.check_missing()
#     # print(len(adult_dataset))

if __name__ == "__main__":
    avila_dataset = AvilaDataset(mode="val")
