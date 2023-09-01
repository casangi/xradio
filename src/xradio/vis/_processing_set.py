import pandas as pd


class _processing_set(dict):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def summary(self):
        summary_data = {"name": [], "intent": [], "field_name": [], "frequency": []}
        for key, value in self.items():
            summary_data["name"].append(key)
            summary_data["intent"].append(value.attrs["intent"])
            summary_data["field_name"].append(value.attrs["field_info"]["name"])
            summary_data["frequency"].append(value["frequency"].values)
        summary_df = pd.DataFrame(summary_data)
        return summary_df
