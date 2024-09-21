
from typing import Dict

import pandas as pd



class DataFrameBuilder:
    def __init__(self):
        self.data = {}


    def push(self, item: Dict):
        for k,v in item.items():
            if not k in self.data:
                self.data[k] = []
            self.data[k].append(v)

    def as_dataframe(self):
        return pd.DataFrame(self.data)