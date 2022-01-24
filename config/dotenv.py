import os

config_values = {}

with open('.env', 'r') as f:
    for line in f.readlines():
        print(line)
        key, value = line.split('=')
        os.environ[key] = value.strip()
        config_values[key] = value.strip()


class Config(object):
    def __init__():
        super().__init__()
    
    @staticmethod
    def audio_path():
        return os.environ['AUDIO_PATH']