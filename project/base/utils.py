



class Statistics:
    def __init__(self):
        self.values = dict()

    def step(self, key, value):
        sum, count = 0.0, 0.0
        if key in self.values:
            sum, count = self.values[key]
        sum += value
        count += 1.0
        self.values[key] = (sum, count)
        
    def set(self, key, value):
        self.values[key] = (value, 1.0)

    def get(self):
        result = dict()
        for k, (sum,count) in self.values.items():
            result[k] = float(sum/count)
        return result
  
    @staticmethod
    def merge(s1, s2):
        result = s1.get()
        result.update(s2.get())
        return result
    
    
    

class NamedCounter:
    def __init__(self):
        self.values = dict()
                
    def step(self, key):
        count = 0
        if (key in self.values):
            count = self.values[key]
        self.values[key] = count + 1
        
        return count