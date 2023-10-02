import pandas as pd


class _processing_set(dict):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def summary(self):
        summary_data = {"name": [], "ddi": [], "intent": [], "field_id": [], "field_name": [], "start_frequency": [], "end_frequency": []}
        for key, value in self.items():
            summary_data["name"].append(key)
            summary_data["ddi"].append(value.attrs["ddi"])
            summary_data["intent"].append(value.attrs["intent"])
            summary_data["field_id"].append(value.attrs["field_info"]["field_id"])
            summary_data["field_name"].append(value.attrs["field_info"]["name"])
            summary_data["start_frequency"].append(value["frequency"].values[0])
            summary_data["end_frequency"].append(value["frequency"].values[-1])
        summary_df = pd.DataFrame(summary_data)
        return summary_df
