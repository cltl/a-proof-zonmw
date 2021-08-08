from configparser import ConfigParser
from pathlib import Path


BASEPATH = Path(__file__).parent.parent.resolve()


config = ConfigParser(converters = {'path': lambda i: BASEPATH / i})
read_config = lambda: config.read(BASEPATH / 'config.ini')
read_config()


PATHS = config['PATHS']