
def set_nested_value(d, keys, value):
    """make dict recursively"""
    keys = keys.split('.')
    for key in keys[:-1]:
        d = d.setdefault(key, {})  # 如果不存在嵌套字典则创建
    d[keys[-1]] = value
    return d

def get_nested_value(d, keys):
    """get dict recursively"""
    keys = keys.split('.')
    for key in keys:
        d = d[key]
    return d