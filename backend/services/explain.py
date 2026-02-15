import shap
import numpy as np

class Explainer:

    def __init__(self, model):
        self.explainer = shap.TreeExplainer(model)

    def explain(self, processed_df, feature_names):

        shap_values = self.explainer.shap_values(processed_df)
        shap_array = np.array(shap_values)

        if shap_array.ndim == 3:
            shap_array = np.mean(np.abs(shap_array), axis=0)
        else:
            shap_array = np.abs(shap_array)

        feature_importance = shap_array[0]

        explanation = dict(
            sorted(
                [(feature, float(value)) for feature, value in zip(feature_names, feature_importance)],
                key=lambda x: x[1],
                reverse=True
            )[:3]
        )

        return explanation
