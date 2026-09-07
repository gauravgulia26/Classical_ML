import json

import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import make_moons
from sklearn.ensemble import (
    GradientBoostingClassifier,
    RandomForestClassifier,
    StackingClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
)
from sklearn.model_selection import train_test_split
from tqdm import tqdm

# ============================================================
# 1. Generate a harder dataset
# ============================================================

X, y = make_moons(
    n_samples=3000,
    noise=0.50,
    random_state=42,
)


# ============================================================
# 2. Train / Test split
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.30,
    random_state=42,
    stratify=y,
)


# ============================================================
# 3. Create base models
# ============================================================

RANDOM_STATE = 42

rf = RandomForestClassifier(
    n_estimators=100,
    max_depth=5,
    min_samples_leaf=5,
    random_state=RANDOM_STATE,
)

g_boost = GradientBoostingClassifier(
    n_estimators=100,
    learning_rate=0.05,
    max_depth=2,
    min_samples_leaf=5,
    random_state=RANDOM_STATE,
)


# ============================================================
# 4. Create stacking model
# ============================================================

stacking = StackingClassifier(
    estimators=[
        ("rf", rf),
        ("g_boost", g_boost),
    ],
    final_estimator=LogisticRegression(
        C=0.5,
        random_state=RANDOM_STATE,
    ),
    cv=5,
)


# ============================================================
# 5. Train models
# ============================================================

models = {
    "Random Forest": rf,
    "Gradient Boosting": g_boost,
    "Stacking": stacking,
}

for model_name, model in tqdm(
    models.items(),
    desc="Training models",
    total=len(models),
):
    model.fit(X_train, y_train)


# ============================================================
# 6. Generate predictions
# ============================================================

predictions = {}

for model_name, model in tqdm(
    models.items(),
    desc="Generating predictions",
    total=len(models),
):
    predictions[model_name] = model.predict(X_test)


# ============================================================
# 7. Calculate metrics
# ============================================================

results = {}

for model_name, y_pred in predictions.items():
    report = classification_report(
        y_test,
        y_pred,
        output_dict=True,
        zero_division=0,
    )

    results[model_name] = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": report["weighted avg"]["precision"],
        "recall": report["weighted avg"]["recall"],
        "f1": report["weighted avg"]["f1-score"],
    }


# ============================================================
# 8. Print results
# ============================================================

print("\n" + "=" * 60)
print("MODEL PERFORMANCE")
print("=" * 60)

for model_name, metrics in results.items():
    print(f"\n{model_name}")

    for metric, value in metrics.items():
        print(f"  {metric:<15}: {value:.4f}")


# ============================================================
# 9. Save results to JSON
# ============================================================

with open("model_metrics.json", "w") as f:
    json.dump(results, f, indent=4)

print("\nMetrics saved to model_metrics.json")


# ============================================================
# 10. Accuracy comparison
# ============================================================

accuracy_results = {model: metrics["accuracy"] for model, metrics in results.items()}

plt.figure(figsize=(8, 5))

plt.bar(
    accuracy_results.keys(),
    accuracy_results.values(),
)

plt.ylim(
    min(accuracy_results.values()) - 0.05,
    1.0,
)

plt.ylabel("Accuracy")
plt.title("Random Forest vs Gradient Boosting vs Stacking")

plt.tight_layout()

plt.savefig(
    "accuracy_comparison.png",
    dpi=300,
)

plt.show()


# ============================================================
# 11. Decision boundary function
# ============================================================


def plot_decision_boundary(
    model,
    X,
    y,
    title,
):

    x_min = X[:, 0].min() - 0.3
    x_max = X[:, 0].max() + 0.3

    y_min = X[:, 1].min() - 0.3
    y_max = X[:, 1].max() + 0.3

    xx, yy = np.meshgrid(
        np.linspace(
            x_min,
            x_max,
            500,
        ),
        np.linspace(
            y_min,
            y_max,
            500,
        ),
    )

    grid = np.c_[
        xx.ravel(),
        yy.ravel(),
    ]

    predictions = model.predict(grid)

    predictions = predictions.reshape(xx.shape)

    plt.figure(figsize=(8, 6))

    plt.contourf(
        xx,
        yy,
        predictions,
        alpha=0.25,
    )

    plt.scatter(
        X[:, 0],
        X[:, 1],
        c=y,
        edgecolor="black",
        s=20,
    )

    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")

    plt.title(title)

    plt.tight_layout()

    filename = title.replace(" ", "_").replace(":", "")

    plt.savefig(
        f"{filename}.png",
        dpi=300,
    )

    plt.show()


# ============================================================
# 12. Visualize decision boundaries
# ============================================================

for model_name, model in tqdm(
    models.items(),
    desc="Generating decision boundaries",
    total=len(models),
):
    plot_decision_boundary(
        model,
        X_test,
        y_test,
        model_name,
    )


# ============================================================
# 13. Analyze disagreement between base models
# ============================================================

rf_pred = predictions["Random Forest"]
gb_pred = predictions["Gradient Boosting"]
stack_pred = predictions["Stacking"]

disagreement = rf_pred != gb_pred

print("\n" + "=" * 60)
print("BASE MODEL DISAGREEMENT")
print("=" * 60)

print(
    f"RF vs GB disagreement: "
    f"{disagreement.sum()} / {len(y_test)} "
    f"({disagreement.mean():.2%})"
)


# ============================================================
# 14. Where stacking differs from both models
# ============================================================

stack_differs = (stack_pred != rf_pred) | (stack_pred != gb_pred)

print(
    f"Stacking differs from at least one base model: "
    f"{stack_differs.sum()} / {len(y_test)} "
    f"({stack_differs.mean():.2%})"
)
