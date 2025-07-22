import joblib
import mlflow
from mlflow.models.signature import infer_signature
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

class ModelTrainer:
    def __init__(self, cfg):
        self.cfg = cfg
        mlflow.set_tracking_uri(cfg.mlflow.tracking_uri)
        mlflow.set_experiment(cfg.mlflow.experiment_name)

    def run(self):
        art = joblib.load(self.cfg.data.processed_data_path)
        X_train, X_test = art["X_train"], art["X_test"]
        y_train, y_test = art["y_train"], art["y_test"]

        # train
        print("Training RandomForestClassifier …")
        model = RandomForestClassifier(**self.cfg.model.params)
        model.fit(X_train, y_train)

        # eval
        preds = model.predict(X_test)
        acc   = accuracy_score(y_test, preds)
        print(f"▶ Test accuracy: {acc:.4f}")

        # log
        with mlflow.start_run():
            mlflow.log_params(self.cfg.model.params)
            mlflow.log_metric("accuracy", acc)
            sig = infer_signature(X_train, model.predict(X_train))
            mlflow.sklearn.log_model(
                model,
                artifact_path="model",
                signature=sig,
                registered_model_name="AdultIncomeModel"
            )
        print("✔ Model logged to MLflow")
