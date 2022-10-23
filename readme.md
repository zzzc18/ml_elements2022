# 6.3

[TOC]

## 题目

选择两个UCI数据集，分别用线性核和高斯核训练一个SVM，并与BP神经网络和C4.5决策树进行试验比较。

## 数据集选择

### Adult Dataset

[Adult Data Set](https://archive.ics.uci.edu/ml/datasets/Adult)

Ron Kohavi, "Scaling Up the Accuracy of Naive-Bayes Classifiers: a Decision-Tree Hybrid", Proceedings of the Second International Conference on Knowledge Discovery and Data Mining, 1996

利用数据判断一个人的收入是否超过50K



数据分为14个属性外加一个分类标准，共15个数据条目



训练集一共32561条数据，其中收入小于50K的有24720条，占比75.9%

验证集一共16281条数据，其中收入小于50K的有12435条，占比76.4%



缺失数据分析：所有缺失的数据都用问号字符“？”代替，经过统计发现训练集和测试集只有三类数据存在缺失：

第2类：workclass（单位类型），训练集缺失1836条，占比5.6%，测试集缺失963条，占比5.9%

第7类：occupation（工作类型），训练集缺失1843条，占比5.7%，测试集缺失966条，占比5.9%

第14类：native-country（祖国），训练集缺失583条，占比1.8%，测试集缺失274条，占比1.7%



由于workclass和occupation缺失量非常接近，可能有较大重复，经统计确认发现所有缺失workclass的类型都缺失occupation。



### Avila Dataset

[Avila Dataset](https://archive.ics.uci.edu/ml/datasets/Avila)

Reliable writer identification in medieval manuscripts through page layout features: The 'Avila' Bible case, Engineering Applications of Artificial Intelligence, Volume 72, 2018, pp. 99-110.



## SVM

[Intel® Extension for Scikit-learn](https://intel.github.io/scikit-learn-intelex/) 可以利用并行加速，可以显著提速，且不影响结果

python -m sklearnex .\svm_adult.py



TODO:

数据集分布图

Avila截图，数据分布