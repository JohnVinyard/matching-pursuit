import os
import importlib.util

def readme(cls):
    spec = importlib.util.find_spec(cls.__module__)
    path, _ = os.path.split(spec.origin)
    readme_path = os.path.join(path, 'readme.md')
    with open(readme_path, 'r') as f:
        content = f.read()
    cls.__doc__ = content
    return cls