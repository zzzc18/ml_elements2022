from torch.utils.data import DataLoader
from data import AdultDataset, AvilaDataset
from tqdm import tqdm
import sklearn.svm
import sklearn.metrics


if __name__ == "__main__":
    '''
    SVM for AvilaDataset
    '''
    # Train Acc: 0.7140939597315437
    # Val Acc: 0.7088243748203507
    # model = sklearn.svm.SVC(kernel="rbf", probability=True, tol=1E-4)
    # Train Acc: 0.5714285714285714
    # Val Acc: 0.5405767940979208
    model = sklearn.svm.SVC(kernel="linear", probability=True)

    print("Loading dataset...")
    train_dataset = AvilaDataset(mode="train")
    val_dataset = AvilaDataset(mode="val")

    print("Training...")
    model.fit(train_dataset.data, train_dataset.label)

    print("Validating...")
    pred = model.predict(train_dataset.data)
    acc = sklearn.metrics.accuracy_score(train_dataset.label, pred)
    print(f"Train Acc: {acc}")
    pred = model.predict(val_dataset.data)
    acc = sklearn.metrics.accuracy_score(val_dataset.label, pred)
    print(f"Val Acc: {acc}")
