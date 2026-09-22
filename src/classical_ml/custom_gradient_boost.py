import warnings
from dataclasses import dataclass, field

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from rich import print
from rich.progress import (
    BarColumn,
    Progress,
    TextColumn,
    TimeRemainingColumn,
)
from rich.table import Table
from sklearn.metrics import mean_squared_error
from sklearn.tree import DecisionTreeRegressor


@dataclass
class CustomGradientBoost:
    dataframe: pd.DataFrame
    learners: int = 10
    learning_rate: float = 0.1
    target_variable: str = "target"
    max_depth: int = 3

    models: list = field(default_factory=list, init=False)
    training_history: list = field(default_factory=list, init=False)

    def __post_init__(self):

        # ----------------------------------------------------------
        # Separate features and target
        # ----------------------------------------------------------

        self.X = self.dataframe.drop(columns=[self.target_variable])

        self.y = self.dataframe[self.target_variable].to_numpy()

        # ----------------------------------------------------------
        # Initial prediction
        #
        # F0(x) = mean(y)
        # ----------------------------------------------------------

        self.initial_prediction = np.full(
            shape=len(self.y),
            fill_value=self.y.mean(),
        )

        # Current ensemble prediction
        self.current_prediction = self.initial_prediction.copy()

    def run(self):
        """
        Train Gradient Boosting using squared-error loss.

        At every iteration:

            residual = y - current_prediction

            tree.fit(X, residual)

            current_prediction +=
                learning_rate * tree_prediction

        For squared-error loss, the residual is the
        negative gradient of the loss.
        """

        with Progress(
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("{task.completed}/{task.total}"),
            TimeRemainingColumn(),
        ) as progress:
            task = progress.add_task(
                "Training Gradient Boosting",
                total=self.learners,
            )

            for i in range(self.learners):
                # --------------------------------------------------
                # 1. Calculate residuals
                # --------------------------------------------------

                residuals = self.y - self.current_prediction

                # --------------------------------------------------
                # 2. Fit weak learner to residuals
                # --------------------------------------------------

                tree = DecisionTreeRegressor(
                    max_depth=self.max_depth,
                    random_state=42,
                    max_leaf_nodes=8
                )

                tree.fit(
                    self.X,
                    residuals,
                )

                # --------------------------------------------------
                # 3. Tree predicts the required correction
                # --------------------------------------------------

                tree_prediction = tree.predict(self.X)

                # --------------------------------------------------
                # 4. Scale correction
                # --------------------------------------------------

                update = self.learning_rate * tree_prediction

                # --------------------------------------------------
                # 5. Update ensemble
                # --------------------------------------------------

                self.current_prediction += update

                # --------------------------------------------------
                # 6. Save model
                # --------------------------------------------------

                self.models.append(tree)

                # --------------------------------------------------
                # 7. Calculate new residuals
                # --------------------------------------------------

                new_residuals = self.y - self.current_prediction

                # --------------------------------------------------
                # 8. Calculate MSE
                # --------------------------------------------------

                mse = mean_squared_error(
                    self.y,
                    self.current_prediction,
                )

                rmse = np.sqrt(mse)

                # --------------------------------------------------
                # 9. Calculate residual magnitude
                #
                # Mean Absolute Residual
                # --------------------------------------------------

                mean_absolute_residual = np.mean(np.abs(new_residuals))

                # --------------------------------------------------
                # 10. Store training history
                # --------------------------------------------------

                self.training_history.append(
                    {
                        "iteration": i + 1,
                        "mse": mse,
                        "rmse": rmse,
                        "mean_absolute_residual": mean_absolute_residual,
                    }
                )

                # --------------------------------------------------
                # 11. Store predictions in dataframe
                # --------------------------------------------------

                self.dataframe[f"Prediction_{i + 1}"] = self.current_prediction

                progress.update(
                    task,
                    advance=1,
                )

        return self

    def predict(self, X):
        """
        Generate predictions using the complete
        trained ensemble.
        """

        prediction = np.full(
            shape=len(X),
            fill_value=self.y.mean(),
        )

        for tree in self.models:
            prediction += self.learning_rate * tree.predict(X)

        return prediction

    def show_training_history(self):
        """
        Display training metrics using Rich table.
        """

        table = Table(title="Gradient Boosting Training History")

        table.add_column(
            "Iteration",
            justify="center",
        )

        table.add_column(
            "MSE",
            justify="right",
        )

        table.add_column(
            "RMSE",
            justify="right",
        )

        table.add_column(
            "Mean Absolute Residual",
            justify="right",
        )

        for row in self.training_history:
            table.add_row(
                str(row["iteration"]),
                f"{row['mse']:.4f}",
                f"{row['rmse']:.4f}",
                f"{row['mean_absolute_residual']:.4f}",
            )

        print(table)

    def show_predictions(self, n=10):
        """
        Display actual vs final predictions.
        """

        table = Table(title=f"Predictions - First {n} Samples")

        table.add_column(
            "Index",
            justify="center",
        )

        table.add_column(
            "Actual",
            justify="right",
        )

        table.add_column(
            "Prediction",
            justify="right",
        )

        table.add_column(
            "Residual",
            justify="right",
        )

        for i in range(min(n, len(self.y))):
            residual = self.y[i] - self.current_prediction[i]

            table.add_row(
                str(i),
                f"{self.y[i]:.3f}",
                f"{self.current_prediction[i]:.3f}",
                f"{residual:.3f}",
            )

        print(table)

    def plot_training_history(self):
        """
        Create two separate plots:

        1. Residual magnitude vs iteration
        2. MSE vs iteration
        """

        import matplotlib.pyplot as plt

        iterations = [row["iteration"] for row in self.training_history]

        residuals = [row["mean_absolute_residual"] for row in self.training_history]

        mse = [row["mse"] for row in self.training_history]

        # ----------------------------------------------------------
        # Graph 1: Residual magnitude
        # ----------------------------------------------------------

        plt.figure(figsize=(8, 5))

        plt.plot(
            iterations,
            residuals,
        )

        plt.xlabel("Boosting Iteration")
        plt.ylabel("Mean Absolute Residual")
        plt.title("Residual Magnitude vs Boosting Iteration")

        plt.grid(True)
        plt.tight_layout()
        plt.savefig('residual_vs_iteration.png')
        # plt.show()

        # ----------------------------------------------------------
        # Graph 2: MSE
        # ----------------------------------------------------------

        plt.figure(figsize=(8, 5))

        plt.plot(
            iterations,
            mse,
        )

        plt.xlabel("Boosting Iteration")
        plt.ylabel("MSE")
        plt.title("MSE vs Boosting Iteration")

        plt.grid(True)
        plt.tight_layout()
        plt.savefig('mse_vs_iteration.png')
        # plt.show()


def get_data():

    from sklearn.datasets import make_regression

    X, y = make_regression(
        n_samples=1000,
        n_features=6,
        noise=10,
        random_state=42,
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


def main():

    # --------------------------------------------------------------
    # Create dataset
    # --------------------------------------------------------------

    df = get_data()

    # --------------------------------------------------------------
    # Create model
    # --------------------------------------------------------------

    model = CustomGradientBoost(
        dataframe=df,
        learners=2600,
        learning_rate=0.1,
        max_depth=3,
    )

    # --------------------------------------------------------------
    # Train
    # --------------------------------------------------------------

    model.run()

    # --------------------------------------------------------------
    # Training metrics
    # --------------------------------------------------------------

    model.show_training_history()

    # --------------------------------------------------------------
    # Predictions
    # --------------------------------------------------------------

    model.show_predictions(n=10)

    # --------------------------------------------------------------
    # Graphs
    # --------------------------------------------------------------

    model.plot_training_history()

    # --------------------------------------------------------------
    # Final dataframe
    # --------------------------------------------------------------

    # print("\n[bold]Final DataFrame:[/bold]")
    # print(model.dataframe.head(2))
    model.dataframe.to_csv('grad_result.csv',index=False)

if __name__ == "__main__":
    main()
