import mlflow
from mlflow.pyfunc import PythonModel

class WeightOnly(PythonModel):
    def predict(self, context, model_input, params = None):
        raise NotImplementedError("Won;t do the model promotion here. API will serve.")


mlflow.models.set_model(WeightOnly())