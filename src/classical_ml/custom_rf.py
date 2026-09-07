from concurrent.futures import ThreadPoolExecutor
from enum import StrEnum
from functools import lru_cache

import numpy as np
import pandas as pd
from pydantic import ConfigDict, Field
from pydantic.dataclasses import dataclass
from rich.console import Console
from rich.table import Table
from rich.traceback import install
from scipy import stats
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from tqdm import tqdm

install()

console = Console()


class SamplingTechniques(StrEnum):
    ROW = "row"
    COLUMN = "column"
    MIXED = "mixed"


@dataclass(config=ConfigDict(arbitrary_types_allowed=True))
class CustomRandomForest:
    dataframe: pd.DataFrame
    sampling_technique: SamplingTechniques

    with_replacement: bool = True
    sampling_percent: float = Field(gt=0, le=1, default=0.2)
    n_estimators: int = Field(gt=0, default=100)
    target_column: str = "target"

    all_trained_models: list[DecisionTreeClassifier] | None = None
    sample_shape: tuple[int, int] | None = None
    sampled_df: pd.DataFrame | None = None

    @property
    def get_original_shape(self) -> tuple[int, int]:
        return self.dataframe.shape

    def generate_samples(self) -> pd.DataFrame:

        match self.sampling_technique:
            case SamplingTechniques.ROW:
                sampled = self.dataframe.sample(
                    frac=self.sampling_percent,
                    replace=self.with_replacement,
                )

            case SamplingTechniques.COLUMN:
                sampled = self.dataframe.sample(
                    frac=self.sampling_percent,
                    axis=1,
                )

            case SamplingTechniques.MIXED:
                sampled = self.dataframe.sample(
                    frac=self.sampling_percent,
                    replace=self.with_replacement,
                ).sample(
                    frac=self.sampling_percent,
                    axis=1,
                )

            case _:
                raise NotImplementedError(
                    f"{self.sampling_technique} is not implemented"
                )

        self.sample_shape = sampled.shape
        self.sampled_df = sampled.reset_index(drop=True)

        return sampled

    def get_estimators(self) -> list[DecisionTreeClassifier]:

        return [DecisionTreeClassifier() for _ in range(self.n_estimators)]

    def train_models(self) -> list[DecisionTreeClassifier]:

        all_estimators = self.get_estimators()

        for idx in tqdm(
            range(self.n_estimators),
            desc="Training Estimators",
            unit="DT",
        ):
            df = self.generate_samples()

            X = df.drop(columns=self.target_column)
            y = df[self.target_column]

            all_estimators[idx].fit(X, y)

        self.all_trained_models = all_estimators

        return all_estimators

    def predict(
        self,
        input: np.ndarray,
        max_workers: int | None = None,
    ) -> np.ndarray:

        if self.all_trained_models is None:
            raise RuntimeError("Models are not trained. Call train_models() first.")

        if max_workers and max_workers > 1:
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                predictions = list(
                    executor.map(
                        lambda model: model.predict(input),
                        self.all_trained_models,
                    )
                )
            return np.array(predictions)

        return np.array([model.predict(input) for model in self.all_trained_models])


@lru_cache
def get_data() -> pd.DataFrame:

    X, y = make_classification(
        random_state=42,
        n_samples=10_000,
        n_features=30,
        n_informative=10,
        n_redundant=5,
    )

    df = pd.DataFrame(X)
    df["target"] = y

    return df


def evaluate_model(
    model: CustomRandomForest,
    test_samples: pd.DataFrame,
    step: int = 100,
    max_workers: int | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:

    results = []
    accuracy_results = []

    total_samples = len(test_samples)

    def _predict_row(item: tuple) -> dict:
        idx, row = item
        X = row.drop(labels=model.target_column)

        true_prediction = row[model.target_column]

        tree_predictions = model.predict(input=X.to_numpy().reshape(1, -1))

        final_prediction = stats.mode(
            tree_predictions.ravel(),
            keepdims=False,
        ).mode

        return {
            "index": idx,
            "predicted": int(final_prediction),
            "true": int(true_prediction),
        }

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        for start in range(0, total_samples, step):
            end = min(
                start + step,
                total_samples,
            )

            current_batch = test_samples.iloc[start:end]

            if max_workers == 1:
                batch_results = []
                for idx, row in tqdm(
                    current_batch.iterrows(),
                    total=len(current_batch),
                    desc=f"Predicting {end}/{total_samples}",
                    unit="row",
                ):
                    batch_results.append(_predict_row((idx, row)))
            else:
                batch_results = list(
                    tqdm(
                        executor.map(_predict_row, current_batch.iterrows()),
                        total=len(current_batch),
                        desc=f"Predicting {end}/{total_samples}",
                        unit="row",
                    )
                )

            results.extend(batch_results)

        # -----------------------------------------
        # Cumulative accuracy
        # -----------------------------------------

        result_df = pd.DataFrame(results)

        correct = (result_df["predicted"] == result_df["true"]).sum()

        incorrect = len(result_df) - correct

        accuracy = correct / len(result_df)

        accuracy_results.append(
            {
                "Samples": len(result_df),
                "Correct": int(correct),
                "Incorrect": int(incorrect),
                "Accuracy": accuracy,
            }
        )

    return (
        pd.DataFrame(results),
        pd.DataFrame(accuracy_results),
    )


def display_accuracy_table(
    accuracy_results: pd.DataFrame,
) -> None:

    table = Table(
        title="Random Forest Test Accuracy",
        show_header=True,
        header_style="bold",
    )

    table.add_column(
        "Samples",
        justify="right",
    )

    table.add_column(
        "Correct",
        justify="right",
    )

    table.add_column(
        "Incorrect",
        justify="right",
    )

    table.add_column(
        "Accuracy",
        justify="right",
    )

    for _, row in accuracy_results.iterrows():
        table.add_row(
            str(int(row["Samples"])),
            str(int(row["Correct"])),
            str(int(row["Incorrect"])),
            f"{row['Accuracy']:.4f}",
        )

    console.print()
    console.print(table)


def main(
    test_size: float = 0.2,
    max_test_samples: int | None = None,
    step: int = 100,
    max_workers: int | None = None,
):

    # =================================================
    # 1. Load complete dataset
    # =================================================

    df = get_data()

    console.print(f"\n[bold]Original Dataset:[/bold] {df.shape}")

    # =================================================
    # 2. TRAIN / TEST SPLIT
    # =================================================

    train_df, test_df = train_test_split(
        df,
        test_size=test_size,
        random_state=42,
        stratify=df["target"],
    )

    console.print(f"[bold]Training Dataset:[/bold] {train_df.shape}")

    console.print(f"[bold]Test Dataset:[/bold] {test_df.shape}\n")

    # =================================================
    # 3. Create Random Forest using TRAINING data
    # =================================================

    model = CustomRandomForest(
        dataframe=train_df,
        sampling_technique=SamplingTechniques.ROW,
        sampling_percent=0.4,
        with_replacement=True,
        n_estimators=200,
        target_column="target",
    )

    # =================================================
    # 4. TRAIN ONLY ON TRAINING DATA
    # =================================================

    console.print("[bold]Training Random Forest[/bold]")

    model.train_models()

    # =================================================
    # 5. Select TEST samples
    # =================================================

    if max_test_samples is not None:
        test_df = test_df.sample(
            n=min(max_test_samples, len(test_df)),
            random_state=42,
        )

    # =================================================
    # 6. Incrementally evaluate TEST data
    # =================================================

    result_df, accuracy_results = evaluate_model(
        model=model,
        test_samples=test_df,
        step=step,
        max_workers=max_workers,
    )

    # =================================================
    # 7. Display accuracy table
    # =================================================

    display_accuracy_table(accuracy_results)

    # =================================================
    # 8. Final metrics
    # =================================================

    final_accuracy = (result_df["predicted"] == result_df["true"]).mean()

    console.print(f"\n[bold]Final Test Accuracy:[/bold] {final_accuracy:.4f}")

    console.print(f"[bold]Test Samples:[/bold] {len(result_df)}")

    return result_df


if __name__ == "__main__":
    main(
        test_size=0.2,
        max_test_samples=2_000,
        step=100,
    )
