from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from rich.console import Console
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
)
from rich.table import Table
from sklearn.datasets import make_classification
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
)
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

console = Console()


@dataclass
class CustomAdaboost:
    dataframe: pd.DataFrame
    target_variable: str
    learner: int = 10

    estimators: list[DecisionTreeClassifier] = field(init=False, default_factory=list)

    alphas: list[float] = field(init=False, default_factory=list)

    weights: np.ndarray = field(init=False)

    def __post_init__(self):

        self.dataframe = self.dataframe.copy()

        self.dataframe[self.target_variable] = self.dataframe[
            self.target_variable
        ].replace({0: -1, 1: 1})

    def __calculate_error(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:

        incorrect = y_true != y_pred

        return self.weights[incorrect].sum()

    def __calculate_alpha(self, error: float) -> float:

        error = np.clip(error, 1e-10, 1 - 1e-10)

        return 0.5 * np.log((1 - error) / error)

    def __update_weights(
        self, y_true: np.ndarray, y_pred: np.ndarray, alpha: float
    ) -> None:

        correct = y_true == y_pred

        self.weights[correct] *= np.exp(-alpha)

        self.weights[~correct] *= np.exp(alpha)

        self.weights /= self.weights.sum()

    def fit(self, X_train: pd.DataFrame, y_train: pd.Series) -> CustomAdaboost:

        y_train = y_train.to_numpy()

        n_samples = len(X_train)

        # Initial uniform distribution
        self.weights = np.full(n_samples, 1 / n_samples)

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("{task.completed}/{task.total}"),
            TimeElapsedColumn(),
            console=console,
        ) as progress:
            task = progress.add_task("Training Custom AdaBoost", total=self.learner)

            for _ in range(self.learner):
                # Create weak learner
                estimator = DecisionTreeClassifier(max_depth=1, random_state=42)

                # Train using current weights
                estimator.fit(X_train, y_train, sample_weight=self.weights)

                # Predict training data
                predictions = estimator.predict(X_train)

                # Weighted error
                error = self.__calculate_error(y_true=y_train, y_pred=predictions)

                # Calculate alpha
                alpha = self.__calculate_alpha(error)

                # Store learner
                self.estimators.append(estimator)

                self.alphas.append(alpha)

                # Update sample weights
                self.__update_weights(y_true=y_train, y_pred=predictions, alpha=alpha)

                progress.advance(task)

        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:

        final_score = np.zeros(len(X))

        for estimator, alpha in zip(self.estimators, self.alphas):
            predictions = estimator.predict(X)

            final_score += alpha * predictions

        return np.where(final_score >= 0, 1, -1)


def get_data() -> pd.DataFrame:

    X, y = make_classification(
        n_samples=10_000, n_features=6, n_informative=4, n_redundant=0, random_state=42
    )

    df = pd.DataFrame(
        X,
        columns=[
            "feature_1",
            "feature_2",
            "feature_3",
            "feature_4",
            "feature_5",
            "feature_6",
        ],
    )

    df["target"] = y

    return df


def build_results_table(y_true, custom_pred, sklearn_pred):

    custom_report = classification_report(y_true, custom_pred, output_dict=True)

    sklearn_report = classification_report(y_true, sklearn_pred, output_dict=True)

    table = Table(title="AdaBoost — Custom vs Scikit-Learn")

    table.add_column("Metric", style="bold")

    table.add_column("Custom", justify="right")

    table.add_column("Scikit-Learn", justify="right")

    table.add_column("Difference", justify="right")

    metrics = [
        ("Accuracy", "accuracy"),
        ("Macro Precision", "macro avg", "precision"),
        ("Macro Recall", "macro avg", "recall"),
        ("Macro F1", "macro avg", "f1-score"),
    ]

    for metric in metrics:
        if metric[0] == "Accuracy":
            custom_value = accuracy_score(y_true, custom_pred)

            sklearn_value = accuracy_score(y_true, sklearn_pred)

        else:
            custom_value = custom_report[metric[1]][metric[2]]

            sklearn_value = sklearn_report[metric[1]][metric[2]]

        difference = custom_value - sklearn_value

        table.add_row(
            metric[0],
            f"{custom_value:.4f}",
            f"{sklearn_value:.4f}",
            f"{difference:+.4f}",
        )

    return table


def main():

    # ==========================================
    # Dataset
    # ==========================================

    df = get_data()

    X = df.drop(columns="target")

    y = df["target"].replace({0: -1, 1: 1})

    # ==========================================
    # Train / Test Split
    # ==========================================

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    console.print(f"\n[bold]Dataset[/bold]  : {len(df):,}")

    console.print(f"[bold]Training[/bold] : {len(X_train):,}")

    console.print(f"[bold]Testing[/bold]  : {len(X_test):,}\n")

    # ==========================================
    # Custom AdaBoost
    # ==========================================

    custom_model = CustomAdaboost(dataframe=df, target_variable="target", learner=100)

    custom_model.fit(X_train=X_train, y_train=y_train)

    custom_train_pred = custom_model.predict(X_train)

    custom_test_pred = custom_model.predict(X_test)

    # ==========================================
    # Scikit-Learn AdaBoost
    # ==========================================

    console.print("\n[bold]Training Scikit-Learn AdaBoost[/bold]")

    sklearn_model = AdaBoostClassifier(
        estimator=DecisionTreeClassifier(max_depth=1, random_state=42),
        n_estimators=100,
        learning_rate=1.0,
        random_state=42,
    )

    sklearn_model.fit(X_train, y_train)

    sklearn_train_pred = sklearn_model.predict(X_train)

    sklearn_test_pred = sklearn_model.predict(X_test)

    # ==========================================
    # Accuracy
    # ==========================================

    train_table = Table(title="Training Accuracy")

    train_table.add_column("Model")
    train_table.add_column("Accuracy", justify="right")

    train_table.add_row(
        "Custom AdaBoost", f"{accuracy_score(y_train, custom_train_pred):.4%}"
    )

    train_table.add_row(
        "Scikit-Learn", f"{accuracy_score(y_train, sklearn_train_pred):.4%}"
    )

    console.print()
    console.print(train_table)

    # ==========================================
    # Test Classification Results
    # ==========================================

    console.print()

    console.print(build_results_table(y_test, custom_test_pred, sklearn_test_pred))


if __name__ == "__main__":
    main()
