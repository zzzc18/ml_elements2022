from torch.utils.data import DataLoader
from data import AdultDataset, AvilaDataset
from tqdm import tqdm
import sklearn.metrics
from my_c45 import tree
import sklearn.externals
import graphviz

if __name__ == "__main__":
    '''
    C45 for AvilaDataset
    '''
    # Train Acc: 0.9871524448705656
    # Val Acc: 0.6190476190476191
    # model = tree.DecisionTree(criterion="C4.5", max_depth=30)
    # Train Acc: 1
    # Val Acc: 0.7138066494203316
    # model = tree.DecisionTree(criterion="entropy", max_depth=30)
    # Train Acc: 1
    # Val Acc: 0.6729903228897193
    # model = tree.DecisionTree(criterion="gini", max_depth=30)

    # Val Acc: 0.6727028839704896
    # Train Acc: 1.0
    import sklearn.tree
    model = sklearn.tree.DecisionTreeClassifier(
        criterion="entropy", max_depth=30)
    # Val Acc: 0.682284181278145
    # Train Acc: 1.0
    # import sklearn.tree
    # model = sklearn.tree.DecisionTreeClassifier(
    #     criterion="gini", max_depth=30)

    print("Loading dataset...")
    train_dataset = AvilaDataset(mode="train")
    val_dataset = AvilaDataset(mode="val")

    print("Training...")
    model.fit(train_dataset.data, train_dataset.label)

    print("Validating...")
    pred = model.predict(val_dataset.data)
    acc = sklearn.metrics.accuracy_score(val_dataset.label, pred)
    print(f"Val Acc: {acc}")
    pred = model.predict(train_dataset.data)
    acc = sklearn.metrics.accuracy_score(train_dataset.label, pred)
    print(f"Train Acc: {acc}")

    # model.print_tree()
    # dot_data = sklearn.tree.export_graphviz(model, out_file=None, max_depth=1)
    # dot = graphviz.Source(dot_data)
    # dot.view()
