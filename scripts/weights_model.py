import mlflow
from mlflow.pyfunc import PythonModel

class WeightOnly(PythonModel):
    def predict(self, context, model_input, params=None):
        raise NotImplementedError("Weights are stored here; the API serves the model.")


mlflow.models.set_model(WeightOnly())