import os
import yaml
import json
from ...logger import user_logger

    
"""
TODO: 
class Config:
    def __init__(self, config):
        pass
"""

class Config(dict):
    """
    A class for managing configuration settings. | 用于管理配置设置的类。
    allowed types: 
        - int
            - min <=> ge
            - max <=> le
            - gt: support str[field]
            - ge: support str[field]
            - lt: support str[field]
            - le: support str[field]
            - in_values
        - float
            - min
            - max
            - gt: support str[field]
            - ge: support str[field]
            - lt: support str[field]
            - le: support str[field]
            - in_values
        - str
            - min_length
            - max_length
            - in_values
        - list/tuple
            - min_length
            - max_length
            - element_type
            - element_values
        - dict
            - key_type
            - value_type
        - bool
            - is_bool
        - None
            - is_none
    examples:
        - config.get_optional('key', default=None, is_numeric=True, gt=0)
        - config.get_required('key', is_str=True, min_length=1)
        - config.get_required('key', is_list=True, min_length=1, element_type=int)
        - config.get_optional('key', default=None, is_dict=True, key_type=str, value_type=int)
        - config.get_required('key', is_bool=True)
        - config.get_required('key', is_none=True)
        - config.get_optional('key', default=None, is_numeric=True, gt=0)
        - config.get_required('key', is_str=True, min_length=1)
        - config.get_required('key', is_list=True, min_length=1, element_type=int)
        - config.get_optional('key', default=None, is_dict=True, key_type=str, value_type=int)
        - config.get_required('key', is_bool=True)
        - config.get_required('key', is_none=True)
        - config.to_json_file('path/to/config.json')
        - config.from_json_file('path/to/config.json')
        - config.to_yaml_file('path/to/config.yaml')
        - config.from_yaml_file('path/to/config.yaml')
        - config = Config();  config['algorithm.batch_size'] = 32
        - config.set_default_dict({'algorithm':{'batch_size':32}})
    FUTURE:
        - add schema e.g. cfg.make_schema({'method':{'lr':{'min':0.001, 'is_float':True}}})
    """
    logger = user_logger
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        for key, value in self.items():
            self[key] = self._make_config_value(value)
        super(Config, self).__setitem__('default_dict', {})

    @classmethod
    def set_logger(cls, logger):
        cls.logger = logger

    @classmethod
    def from_dict(cls, config):
        return make_config(config)
    
    @classmethod
    def from_json_file(cls, file_path, *args, **kwargs):
        with open(file_path, 'r') as f:
            return make_config(json.load(f, *args, **kwargs))
    
    @classmethod
    def from_yaml_file(cls, file_path, *args, **kwargs):
        with open(file_path, 'r') as f:
            return make_config(yaml.load(f, Loader=yaml.SafeLoader, *args, **kwargs))
    
    def to_dict(self):
        config = {}
        for key, value in self.items():
            if isinstance(value, Config):
                config[key] = value.to_dict()
            else:
                config[key] = value
        return config
        
    def to_json(self, indent=4, *args, **kwargs):
        return json.dumps(self.to_dict(), indent=indent, *args, **kwargs)
    
    def to_json_file(self, file_path, ensure_exist_dir=True, indent=4, *args, **kwargs):
        if ensure_exist_dir:
            os.makedirs(os.path.dirname(os.path.abspath(file_path)), exist_ok=True)
        with open(file_path, 'w') as f:
            json.dump(self.to_dict(), f, indent=indent, *args, **kwargs)
        
    def to_yaml_file(self, file_path, ensure_exist_dir=True, *args, **kwargs):
        if ensure_exist_dir:
            os.makedirs(os.path.dirname(os.path.abspath(file_path)), exist_ok=True)
        with open(file_path, 'w') as f:
            yaml.dump(self.to_dict(), f, *args, **kwargs)
    
    @staticmethod
    def _make_config_value(value):
        if isinstance(value, (dict, Config)):
            value = make_config(value)
        return value
        
    def set_default_dict(self, default_dict):
        # print(f'{default_dict=}')
        super(Config, self).__setattr__('default_dict', default_dict) # make_config(default_dict)
        return self
        
    def get_optional(self, key, default=None, **kwargs):
        keys = key.split('.')
        value = self
        for k in keys:
            try:
                value = super(Config, value).__getitem__(k)
            except KeyError:
                class_default = super(Config, self).__getattribute__('default_dict')[key]
                default = default if default is not None else class_default
                value = self._make_config_value(default)
                self.logger.info(f"Key `{key}` not found, using default value `{value}`")
                return self.validate(key, value, **kwargs)
        return self.validate(key, value, **kwargs)
    
    def get_required(self, key, **kwargs):
        keys = key.split('.')
        value = self
        for k in keys:
            try:
                value = super(Config, value).__getitem__(k)
            except KeyError:
                raise KeyError(f"Key `{key}` not found")
        return self.validate(key, value, **kwargs)
        
    def set(self, key, value, strict=False, **kwargs):
        keys = key.split('.')
        parent = self
        for key in keys[:-1]:
            if key in parent:
                if not isinstance(parent[key], Config):
                    msg = f"Key {key} is not a Config: {type(parent[key])}"
                    if strict:
                        raise ValueError(msg)
                    else:
                        self.logger.warning(msg)
            else:
                super(Config, parent).__setitem__(key, Config())
            parent = parent[key]
                
        value = self._make_config_value(value)
        value = self.validate(keys[-1], value, **kwargs)
        if keys[-1] in parent:
            type_match = isinstance(parent[keys[-1]], type(value))
            type_match |= type_match or (type(parent[keys[-1]]) in (tuple, list) and type(value) in (tuple, list))
            type_match |= type_match or (type(parent[keys[-1]]) in (dict, Config) and type(value) in (dict, Config))
            if not type_match:
                msg = f"Type mismatch for config item '{keys[-1]}'. Expected {type(parent[keys[-1]])}, but got {type(value)}"
                if strict:
                    raise TypeError(msg)
                else:
                    self.logger.warning(msg)
            
            super(Config, parent).__setitem__(keys[-1], value)
        else:
            super(Config, parent).__setitem__(keys[-1], value)
 
    def update(self, config, strict=True):
        """
        Update the configuration dictionary `recursively`. | 递归更新配置字典。

        Parameters:
        - config: A dictionary containing new configuration items | 包含新配置项的字典
        - strict: Boolean, if True, strictly check types; if False, allow type mismatches | 布尔值，如果为 True，则严格检查类型；如果为 False，则允许类型不匹配 

        Returns:
        - The updated configuration dictionary | 更新后的配置字典

        Notes:
        - This method recursively updates nested dictionaries | 此方法递归更新嵌套字典
        - If a new configuration item doesn't exist in the original config, it will be added | 如果新配置项在原始配置中不存在，则将其添加
        - When strict is True, a TypeError is raised if a new config item's type doesn't match the original | 当 strict 为 True 时，如果新配置项的类型与原始类型不匹配，则引发 TypeError
        - When strict is False, type mismatches are allowed but a warning is printed | 当 strict 为 False 时，允许类型不匹配，但会打印警告
        """
        def recursive_update(original, update):
            for key, value in update.items():
                key_in_original = key in original
                if key_in_original:
                    if isinstance(value, (Config, dict)) and isinstance(original[key], (Config, dict)):
                        recursive_update(original[key], value)
                    else:
                        type_match = isinstance(value, type(original[key]))
                        type_match |= type_match or (type(original[key]) in (tuple, list) and type(value) in (tuple, list))
                        type_match |= type_match or (type(original[key]) in (dict, Config) and type(value) in (dict, Config))
                        if not type_match:
                            msg = f"Type mismatch for config item '{key}'. Expected {type(original[key])}, but got {type(value)}"   
                            if strict:
                                raise TypeError(msg)
                            else:
                                self.logger.warning(msg)
                        original[key] = value
                else:
                    original[key] = value

        recursive_update(self, config)
        return self
    
    def get(self, key, default=None, **kwargs):
        return self.get_optional(key, default, **kwargs)
    
    def __getitem__(self, key):
        return self.get_required(key)
    
    def __getattr__(self, key):
        return self.get_required(key)
    
    def __setattr__(self, key, value):
        print(f'{key=}, {value=}')
        return self.set(key, value, strict=False)
    
    def __setitem__(self, key, value):
        return self.set(key, value, strict=False)

    def validate(self, key, value, **kwargs):
        # Check type validity | 检查类型有效性
        # 先整体赋值，然后check，可以比较相对大小
        if kwargs.get('is_numeric') and not isinstance(value, (int, float)):
            raise TypeError(f"Key `{key}`: Value `{value}` should be `numeric`")
        if kwargs.get('is_int') and not isinstance(value, int):
            raise TypeError(f"Key `{key}`: Value `{value}` should be `int`")
        if kwargs.get('is_float') and not isinstance(value, float):
            raise TypeError(f"Key `{key}`: Value `{value}` should be `float`")
        if kwargs.get('is_str') and not isinstance(value, str):
            raise TypeError(f"Key `{key}`: Value `{value}` should be `str`")
        if kwargs.get('is_list') and not isinstance(value, list):
            raise TypeError(f"Key `{key}`: Value `{value}` should be `list`")
        if kwargs.get('is_tuple') and not isinstance(value, tuple):
            raise TypeError(f"Key `{key}`: Value `{value}` should be `tuple`")
        if kwargs.get('is_dict') and not isinstance(value, (dict, Config)):
            raise TypeError(f"Key `{key}`: Value `{value}` should be `dict`")
        if kwargs.get('is_bool') and not isinstance(value, bool):
            raise TypeError(f"Key `{key}`: Value `{value}` should be `bool`")
        if kwargs.get('is_none') and not isinstance(value, type(None)):
            raise TypeError(f"Key `{key}`: Value `{value}` should be `None`")
                            
        # Check custom type | 检查自定义类型
        # custom_type NOT allowed
        # custom_type = kwargs.get('custom_type')
        # if custom_type:
        #     if not isinstance(value, custom_type):
        #         raise TypeError(f"Value {value} should be of custom type {custom_type}")
            
        # Check inclusion | 检查包含
        in_values = kwargs.get('in_values')
        if in_values is not None:
            if value not in in_values:
                raise ValueError(f"Key `{key}`: Value `{value}` must be one of `{in_values}`")
        
        # Check numeric range | 检查数值范围
        min_value = kwargs.get('min')
        max_value = kwargs.get('max')
        if min_value is not None and value < min_value:
            raise ValueError(f"Key `{key}`: Value `{value}` is less than the minimum allowed value `{min_value}`")
        if max_value is not None and value > max_value:
            raise ValueError(f"Key `{key}`: Value `{value}` is greater than the maximum allowed value `{max_value}`")
            
        # 额外的大小检查和包含检查
        gt = kwargs.get('gt')
        ge = kwargs.get('ge')
        lt = kwargs.get('lt')
        le = kwargs.get('le')
        ne = kwargs.get('ne')
        
        # allow non-int and non-float types to compare | 允许非int和非float类型比较
        if gt is not None:
            gt_value = gt if not isinstance(gt, str) else self[gt]
            if not (value > gt_value):
                raise ValueError(f"Key `{key}`: Value `{value}` must be greater than `{gt}`")
        if ge is not None:
            ge_value = ge if not isinstance(ge, str) else self[ge]
            if not (value >= ge_value):
                raise ValueError(f"Key `{key}`: Value `{value}` must be greater than or equal to `{ge}`")
        if lt is not None:
            lt_value = lt if not isinstance(lt, str) else self[lt]
            if not (value < lt_value):
                raise ValueError(f"Key `{key}`: Value `{value}` must be less than `{lt}`")
        if le is not None:
            le_value = le if not isinstance(le, str) else self[le]
            if not (value <= le_value):
                raise ValueError(f"Key `{key}`: Value `{value}` must be less than or equal to `{le}`")  
        if ne is not None:
            ne_value = ne if not isinstance(ne, str) else self[ne]
            if not (value != ne_value):
                raise ValueError(f"Key `{key}`: Value `{value}` must be not equal to `{ne}`")
            
        # Check string length | 检查字符串长度
        min_length = kwargs.get('min_length')
        max_length = kwargs.get('max_length')
        if min_length is not None:
            if isinstance(value, (str, list, tuple)):
                if len(value) < min_length:
                    raise ValueError(f"Key `{key}`: length `{len(value)}` is less than the minimum allowed length `{min_length}`")
            else:
                raise TypeError(f'Key `{key}`: Only (str|list|tuple) allowed to use min_length, but got {type(value)}')
        
        if max_length is not None:
            if isinstance(value, (str, list, tuple)):
                if len(value) > max_length:
                    raise ValueError(f"Key `{key}`: length `{len(value)}` is greater than the maximum allowed length `{max_length}`")
            else:
                raise TypeError(f'Key `{key}`: Only (str|list|tuple) allowed to use max_length, but got `{type(value)}`')
        
        # Check list/tuple | 检查列表/元组
        element_type = kwargs.get('element_type')
        element_values = kwargs.get('element_values')
        if element_type:
            if isinstance(value, (list, tuple)):
                for item in value:
                    if not isinstance(item, element_type):
                        raise TypeError(f"Key `{key}`: List/tuple element `{item}` should be of type `{element_type}`, but is `{type(item)}`")
            else:
                raise TypeError(f'Key `{key}`: Only (list|tuple) allowed to use element_type, but got `{type(value)}`')
        if element_values:
            if isinstance(value, (list, tuple)):
                for item in value:
                    if item not in element_values:
                        raise ValueError(f"Key `{key}`: List element `{item}` must be one of `{element_values}`") 
            else:
                raise ValueError(f'Key `{key}`: Only (list|tuple) allowed to use element_values, but got `{type(value)}`')
        
        # Check dict | 检查字典
        key_type = kwargs.get('key_type')
        value_type = kwargs.get('value_type')
        if key_type:
            if isinstance(value, (dict, Config)):
                for key in value.keys():
                    if not isinstance(key, key_type):
                        raise TypeError(f"Key `{key}`: Dict key `{key}` should be of type `{key_type}`, but is `{type(key)}`")
            else:
                raise TypeError(f'Key `{key}`: Only (dict) allowed to use key_type, but got `{type(value)}`')
        if value_type:
            if isinstance(value, (dict, Config)):
                for key in value.keys():
                    if not isinstance(value[key], value_type):
                        raise TypeError(f"Key `{key}`: Dict value `{value[key]}` should be of type `{value_type}`, but is `{type(value[key])}`")
            else:
                raise TypeError(f'Key `{key}`: Only (dict) allowed to use value_type, but got `{type(value)}`')
            
        return value
   
    def __repr__(self) -> str:
        return f'Config({super().__repr__()})'
    
    def __str__(self) -> str:
        return repr(self)
         
        
from_dict = Config.from_dict
from_json_file = Config.from_json_file
from_yaml_file = Config.from_yaml_file
set_logger = Config.set_logger
    
def make_config(config: dict|Config) -> Config:
    super_config = Config()
    for key, value in config.items():
        if isinstance(value, (dict, Config)):
            value = make_config(value)
        super_config[key] = value
    return super_config

def get_optional_config(config, key, default=None, **kwargs):
    return config.get_optional(key, default=default, **kwargs)

def get_required_config(config, key, **kwargs):
    return config.get_required(key, **kwargs)

def update_config(config, new_config, strict=True):
    return config.update(new_config, strict=strict)

__all__ = [
    'Config', 'from_dict', 'from_json_file', 'from_yaml_file', 'set_logger',
    'get_optional_config', 'get_required_config', 'update_config', 'make_config'
]