
class DatasetArgs:
    def __init__(self):
        self.global_test = True

class BaseExperimentParams:
    def __init__(self):
        self.use_sample_weight = True

    def __str__(self):
        # Get the dictionary of attributes (including those from subclasses)
        attrs = vars(self)
        
        # Format each attribute and value as 'key: value' pairs
        formatted_attrs = [f"{key}: {value}" for key, value in attrs.items()]
        
        # Join them into a single string with newlines
        return "\n".join(formatted_attrs)
