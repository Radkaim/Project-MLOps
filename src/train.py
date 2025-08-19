import mlflow
import mlflow.sklearn
from mlflow.models.signature import infer_signature
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score
import pandas as pd
import numpy as np
import yaml
import itertools

# Import các model có thể train
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC


def main():
    # 1. Đọc config
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)

    ds_conf = config["dataset"]

    # 2. Sinh dữ liệu
    X, y = make_classification(
        n_samples=ds_conf["n_samples"],
        n_features=ds_conf["n_features"],
        n_informative=ds_conf["n_informative"],
        n_classes=ds_conf["n_classes"],
        random_state=ds_conf["random_state"]
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=ds_conf["test_size"], random_state=ds_conf["random_state"]
    )

    feature_names = [f"feature_{i}" for i in range(X.shape[1])]
    X_train_df = pd.DataFrame(X_train, columns=feature_names)
    X_test_df = pd.DataFrame(X_test, columns=feature_names)

    # 3. Cấu hình MLflow
    mlflow.set_tracking_uri("file:./mlruns")
    mlflow.set_experiment("MLOps_Project")

    best_acc = -np.inf
    best_model = None
    best_params = {}
    best_model_name = None

    # 4. Train cho từng model trong config
    for model_name, params in config["models"].items():
        ModelClass = globals()[model_name]   # lấy class từ tên (vd: LogisticRegression)

        # Tạo grid search param combinations
        keys, values = zip(*params.items())
        for v in itertools.product(*values):
            param_dict = dict(zip(keys, v))

            with mlflow.start_run():
                model = ModelClass(**param_dict)
                model.fit(X_train_df, y_train)

                # ---- Train accuracy ----
                y_train_pred = model.predict(X_train_df)
                train_acc = accuracy_score(y_train, y_train_pred)

                # ---- Test accuracy ----
                y_test_pred = model.predict(X_test_df)
                test_acc = accuracy_score(y_test, y_test_pred)

                # ---- Cross-validation ----
                cv_scores = cross_val_score(model, X_train_df, y_train, cv=5)
                cv_mean = cv_scores.mean()
                cv_std = cv_scores.std()

                # Log thông tin
                mlflow.log_param("model", model_name)
                for k, val in param_dict.items():
                    mlflow.log_param(k, val)
                mlflow.log_metric("train_accuracy", train_acc)
                mlflow.log_metric("test_accuracy", test_acc)
                mlflow.log_metric("cv_accuracy_mean", cv_mean)
                mlflow.log_metric("cv_accuracy_std", cv_std)

                # Log model
                signature = infer_signature(X_train_df, model.predict(X_train_df))
                mlflow.sklearn.log_model(
                    sk_model=model,
                    name="model",
                    signature=signature,
                    input_example=X_test_df.head(5)
                )

                print(f"[INFO] {model_name}({param_dict}) "
                      f"-> Train={train_acc:.4f}, Test={test_acc:.4f}, "
                      f"CV={cv_mean:.4f} (+/- {cv_std:.4f})")

                # Check best (theo test acc)
                if test_acc > best_acc:
                    best_acc = test_acc
                    best_model = model
                    best_params = param_dict
                    best_model_name = model_name

    # 5. Log best model vào registry
    print("\n[RESULT] Best Model:", best_model_name, best_params, "Test Accuracy:", best_acc)

    # Tính lại metrics cho best_model
    y_train_pred = best_model.predict(X_train_df)
    best_train_acc = accuracy_score(y_train, y_train_pred)

    y_test_pred = best_model.predict(X_test_df)
    best_test_acc = accuracy_score(y_test, y_test_pred)

    cv_scores = cross_val_score(best_model, X_train_df, y_train, cv=5)
    best_cv_mean = cv_scores.mean()
    best_cv_std = cv_scores.std()

    with mlflow.start_run(run_name="BestModel"):
        mlflow.log_param("model", best_model_name)
        mlflow.log_params(best_params)

        mlflow.log_metric("train_accuracy", best_train_acc)
        mlflow.log_metric("test_accuracy", best_test_acc)
        mlflow.log_metric("cv_accuracy_mean", best_cv_mean)
        mlflow.log_metric("cv_accuracy_std", best_cv_std)

        signature = infer_signature(X_train_df, best_model.predict(X_train_df))
        mlflow.sklearn.log_model(
            sk_model=best_model,
            name="BestClassifier",
            signature=signature,
            input_example=X_test_df.head(5),
            registered_model_name="BestClassifier"
        )


if __name__ == "__main__":
    main()
