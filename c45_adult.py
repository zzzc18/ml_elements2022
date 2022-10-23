from torch.utils.data import DataLoader
from data import AdultDataset, AvilaDataset
from tqdm import tqdm
import sklearn.metrics
from my_c45 import tree

if __name__ == "__main__":
    '''
    C45 for AdultDataset
    '''
    # Val Acc: 0.843744241754192
    # Train Acc: 0.8565768864592611
    model = tree.DecisionTree(criterion="C4.5", max_depth=10)
    # Val Acc: 0.8578097168478594
    # Train Acc: 0.8667731335032708
    # model = tree.DecisionTree(criterion="entropy", max_depth=10)
    # Val Acc: 0.858055402002334
    # Train Acc: 0.8710113325757809
    # model = tree.DecisionTree(criterion="gini", max_depth=10)

    # Val Acc: 0.8570112400958172
    # Train Acc: 0.8676330579527656
    # import sklearn.tree
    # model = sklearn.tree.DecisionTreeClassifier(
    #     criterion="entropy", max_depth=10)
    # Val Acc: 0.857379767827529
    # Train Acc: 0.8703663892386597
    # import sklearn.tree
    # model = sklearn.tree.DecisionTreeClassifier(
    #     criterion="gini", max_depth=10)

    print("Loading dataset...")
    train_dataset = AdultDataset(mode="train")
    val_dataset = AdultDataset(mode="val")

    print("Training...")
    model.fit(train_dataset.data, train_dataset.label)

    print("Validating...")
    pred = model.predict(val_dataset.data)
    acc = sklearn.metrics.accuracy_score(val_dataset.label, pred)
    print(f"Val Acc: {acc}")
    pred = model.predict(train_dataset.data)
    acc = sklearn.metrics.accuracy_score(train_dataset.label, pred)
    print(f"Train Acc: {acc}")
