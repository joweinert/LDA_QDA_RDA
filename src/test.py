from typing import Type
from qda import QDA
from lda import LDA
from rda import RDA
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis as SkQDA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as SkLDA
import numpy as np
import pandas as pd
from enum import Enum
from sklearn.datasets import load_iris, load_wine, load_breast_cancer, load_digits
from sklearn.model_selection import train_test_split


class DatasetsEnum(str, Enum):
    """ "Enum for datasets."""

    FISH = "fish"
    IRIS = "iris"
    WINE = "wine"
    CANCER = "cancer"
    DIGITS = "digits"


class TestData:
    """Class to load datasets."""

    def __init__(
        self,
        dataset: DatasetsEnum = DatasetsEnum.FISH,
        test_size: float = 0.2,
    ):
        self.dataset = dataset
        self.X = None
        self.y = None
        self.load_dataset()
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X,
            self.y,
            test_size=test_size,
            random_state=42,
            stratify=self.y,  # for balanced classes
        )

    def load_dataset(self):
        if self.dataset == DatasetsEnum.FISH:
            X, y = self.load_fish()
        elif self.dataset == DatasetsEnum.IRIS:
            X, y = load_iris(return_X_y=True)
        elif self.dataset == DatasetsEnum.WINE:
            X, y = load_wine(return_X_y=True)
        elif self.dataset == DatasetsEnum.CANCER:
            X, y = load_breast_cancer(return_X_y=True)
        elif self.dataset == DatasetsEnum.DIGITS:
            X, y = load_digits(return_X_y=True)
        else:
            raise NotImplementedError(f"Dataset {self.dataset} is not yet implemented")

        self.X = X
        self.y = y

    def get_data(self):
        return self.X, self.y

    def get_train_data(self):
        return self.X_train, self.y_train

    def get_test_data(self):
        return self.X_test, self.y_test

    def get_train_test_data(self):
        return self.X_train, self.y_train, self.X_test, self.y_test

    @staticmethod
    def load_fish():
        """Load fish dataset."""
        df = pd.read_csv("../data/salmon_seabass.csv")
        df = df.drop(columns=["index"], errors="ignore")

        X = df.drop(columns=["species"])
        y = df["species"]

        return np.asarray(X), np.asarray(y)


class TestDiscriminantAnalysis:
    """Test the Discriminant Analysis models (QDA, LDA, RDA), optional: compare against each otheer or sklearn's implementation."""

    def __init__(
        self,
        classifier_cls: Type[QDA | LDA | RDA],
        compare_to_classifier_cls: Type[QDA | LDA | RDA | SkLDA | SkQDA | None] = None,
        theoretical: bool = False,
        classifier_kwargs: dict = None,
        compare_to_kwargs: dict = None,
        dataset: DatasetsEnum = DatasetsEnum.IRIS,
    ):
        self.classifier_name = classifier_cls.__name__

        classifier_kwargs = classifier_kwargs or {}
        classifier_kwargs["theoretical"] = theoretical
        compare_to_kwargs = compare_to_kwargs or {}

        self.X_train, self.y_train, self.X_test, self.y_test = TestData(dataset).get_train_test_data()
        self.classifier = classifier_cls(**classifier_kwargs).fit(self.X_train, self.y_train)
        if compare_to_classifier_cls is not None:
            self.comp_classifier_name = compare_to_classifier_cls.__name__
            self.compare_to_classifier = compare_to_classifier_cls(**compare_to_kwargs).fit(self.X_train, self.y_train)
        else:
            self.compare_to_classifier = None

    def test_classifier(self):
        preds = {self.classifier_name: self.classifier.predict(self.X_test)}
        probs = {self.classifier_name: self.classifier.predict_proba(self.X_test)}

        if self.compare_to_classifier is not None:
            preds[self.comp_classifier_name] = self.compare_to_classifier.predict(self.X_test)
            probs[self.comp_classifier_name] = self.compare_to_classifier.predict_proba(self.X_test)

        for name, pred in preds.items():
            print(f"\nPredictions ({name}):", pred[:5])

        for name, prob in probs.items():
            print(f"\nProbabilities ({name}):\n", prob[:5])

        if len(preds) > 1:
            print("\nSame predictions:", np.all(preds[self.classifier_name] == preds[self.comp_classifier_name]))
            print(
                "\nSame probabilities:",
                np.allclose(probs[self.classifier_name], probs[self.comp_classifier_name], atol=1e-6),
            )

    def dataset_preview(self):
        """Print dataset preview."""
        print("Dataset preview:\nX_test head:\n", self.X_test[:5])
        print("\ny_test head:\n", self.y_test[:5])
