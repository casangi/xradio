import pandas as pd


class processing_set(dict):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.meta = {'summary':{}}
    #     generate_meta(self)

    # def generate_meta(self):
    #     self.meta['summary'] = {"base": _summary(self)}
    #     self.meta['max_dims'] = _get_ps_max_dims(self)

    def summary(self, data_group="base"):
        if data_group in self.meta['summary']:
            return self.meta['summary'][data_group]
        else:
            self.meta['summary'][data_group] = self._summary(data_group)
            return self.meta['summary'][data_group]
        
    def get_ps_max_dims(self):
        if 'max_dims' in self.meta:
            return self.meta['max_dims']
        else:
            self.meta['max_dims'] = self._get_ps_max_dims()
            return self.meta['max_dims'] 

    def _summary(self, data_group="base"):
        summary_data = {
            "name": [],
            "ddi": [],
            "intent": [],
            "field_id": [],
            "field_name": [],
            "start_frequency": [],
            "end_frequency": [],
            "shape": []
        }
        for key, value in self.items():
            summary_data["name"].append(key)
            summary_data["ddi"].append(value.attrs["ddi"])
            summary_data["intent"].append(value.attrs["intent"])

            if "visibility" in value.attrs["data_groups"][data_group]:
                data_name = value.attrs["data_groups"][data_group]["visibility"]

            if "spectrum" in value.attrs["data_groups"][data_group]:
                data_name = value.attrs["data_groups"][data_group]["spectrum"]

            summary_data["shape"].append(
                value[data_name].shape
            )

            summary_data["field_id"].append(
                value[data_name].attrs[
                    "field_info"
                ]["field_id"]
            )
            summary_data["field_name"].append(
                value[data_name].attrs[
                    "field_info"
                ]["name"]
            )
            summary_data["start_frequency"].append(value["frequency"].values[0])
            summary_data["end_frequency"].append(value["frequency"].values[-1])
        summary_df = pd.DataFrame(summary_data)
        return summary_df
    
    def _get_ps_max_dims(self):
        max_dims = None
        for ms_xds in self.values():
            if max_dims is None:
                max_dims = dict(ms_xds.sizes)
            else:
                for dim_name, size in ms_xds.sizes.items():
                    if dim_name in max_dims:
                        if max_dims[dim_name] < size:
                            max_dims[dim_name] = size
                    else:
                        max_dims[dim_name] = size
        return max_dims
    
    def get(self, id):
        return self[list(self.keys())[id]]