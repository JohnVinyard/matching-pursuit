import os

config_values = {}

try:
    with open('.env', 'r') as f:
        for line in f.readlines():
            print(line)
            key, value = line.split('=')
            os.environ[key] = value.strip()
            config_values[key] = value.strip()
except IOError:
    print('WARNING, no .env file found')


class Config(object):
    def __init__(self):
        super().__init__()

    @staticmethod
    def audio_path():
        return os.environ['AUDIO_PATH']

    @staticmethod
    def impulse_response_path():
        return os.environ['IMPULSE_RESPONSE_PATH']
    
    @staticmethod
    def s3_bucket():
        return os.environ['S3_BUCKET']
