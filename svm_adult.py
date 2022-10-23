from torch.utils.data import DataLoader
from data import AdultDataset, AvilaDataset
from tqdm import tqdm
import sklearn.svm
import sklearn.metrics


if __name__ == "__main__":
    '''
    SVM for AdultDataset
    '''
    # Train Acc: 0.8660974785786677
    # Val Acc: 0.8599594619495117
    model = sklearn.svm.SVC(kernel="rbf", probability=True)
    # Train Acc: 0.8525843800866066
    # Val Acc: 0.8527731711811314
    # model = sklearn.svm.SVC(kernel="linear", probability=True)

    print("Loading dataset...")
    train_dataset = AdultDataset(mode="train")
    val_dataset = AdultDataset(mode="val")

    print("Training...")
    model.fit(train_dataset.data, train_dataset.label)

    print("Validating...")
    pred = model.predict(train_dataset.data)
    acc = sklearn.metrics.accuracy_score(train_dataset.label, pred)
    print(f"Train Acc: {acc}")
    pred = model.predict(val_dataset.data)
    acc = sklearn.metrics.accuracy_score(val_dataset.label, pred)
    print(f"Val Acc: {acc}")
