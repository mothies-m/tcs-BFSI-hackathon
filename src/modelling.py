from sklearn.model_selection import train_test_split
from catboost import CatBoostClassifier
from sklearn.metrics import (
    precision_score,
    recall_score,
    confusion_matrix,
    f1_score,
    accuracy_score,
)
from bayes_opt import BayesianOptimization
import joblib


def build_model(x, y):
    X_train, X_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=0.2,
        random_state=4,
    )

    def objective(iterations, learning_rate, depth, l2_leaf_reg):
        clf = CatBoostClassifier(
            iterations=int(iterations),
            learning_rate=learning_rate,
            depth=int(depth),
            l2_leaf_reg=l2_leaf_reg,
            loss_function="Logloss",
            random_seed=42,
            verbose=0,
        )
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        score = f1_score(y_test, y_pred)
        return score

    # Set the ranges of hyperparameters to optimize
    pbounds = {
        "iterations": (100, 1000),
        "learning_rate": (0.01, 0.3),
        "depth": (4, 10),
        "l2_leaf_reg": (1, 10),
    }

    # Define the Bayesian Optimization object
    optimizer = BayesianOptimization(
        f=objective,
        pbounds=pbounds,
        random_state=42,
    )

    # Run the optimization process
    optimizer.maximize(init_points=5, n_iter=10)

    # Print the best hyperparameters and score found
    print("Best score:", optimizer.max["target"])
    print("Best parameters:", optimizer.max["params"])

    # Model Traininig
    best_params = optimizer.max["params"]
    best_params["iterations"] = int(best_params["iterations"])
    best_params["depth"] = int(best_params["depth"])
    model = CatBoostClassifier(
        **best_params, loss_function="Logloss", random_seed=42, verbose=0
    )
    print("Initiating model fitting")
    model.fit(X_train, y_train)
    print("Model fitting complete")
    # Save the model to a file
    with open("./artifacts/model.pkl", "wb") as f:
        joblib.dump(model, f)

    y_pred = model.predict(X_test)

    print(
        f"The accuracy for the test set is given \
        as: {accuracy_score(y_test, y_pred):.2f}"
    )
    print(
        f"The F1-score for the test set is given \
    as: {f1_score(y_test, y_pred):.2f}"
    )
    print(
        f"The precision score for the test set is given \
        as: {precision_score(y_test, y_pred):.2f}"
    )
    print(
        f"The recall score for the test set is given \
        as: {recall_score(y_test, y_pred):.2f}"
    )
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
