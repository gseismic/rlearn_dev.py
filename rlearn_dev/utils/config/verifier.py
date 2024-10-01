

class Schema:
    """
    A schema for validating a configuration dictionary.
    
    examples:
        - schema = Schema({'algorithm':{'batch_size':{'is_int':True, 'min':1, 'max':100}}})
        - schema.verify({'algorithm':{'batch_size':10}})
        - schema.verify({'algorithm':{'batch_size':1000}})
        - schema.verify({'algorithm':{'batch_size':0}})
    """
    def __init__(self, config, *args, **kwargs):
        self.schema = config
    
    def verify(self, config):
        for key, value in config.items():
            if key not in self.schema:
                raise ValueError(f"Key '{key}' not found in schema")
            if not isinstance(value, dict):
                raise ValueError(f"Value for key '{key}' is not a dictionary")
            self._verify_dict(value, self.schema[key])

    def _verify_dict(self, config, schema): 
        for key, value in config.items():
            if key not in schema:
                raise ValueError(f"Key '{key}' not found in schema")
            if not isinstance(value, dict):
                raise ValueError(f"Value for key '{key}' is not a dictionary")
            self._verify_dict(value, schema[key])

    def __setitem__(self, key, value):
        self.schema[key] = value

    def __getitem__(self, key):
        return self.schema[key] 

    def __contains__(self, key):
        return key in self.schema

    def __str__(self):
        return str(self.schema)

    def __repr__(self):
        return repr(self.schema)