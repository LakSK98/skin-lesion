import configparser
import os

class ConfigManager:
    def __init__(self, config_file='config.ini'):
        self.config = configparser.ConfigParser()
        self.config_file = config_file
        self.load_config()

    def load_config(self):
        """Load configuration from a file."""
        if not os.path.exists(self.config_file):
            self.create_default_config()
        self.config.read(self.config_file)
        self.validate_config()

    def create_default_config(self):
        """Create a default configuration file."""
        self.config['DEFAULT'] = {
            'LogLevel': 'INFO',
            'DataPath': './data',
            'OutputPath': './output'
        }

        self.config['DATABASE'] = {
            'Host': 'localhost',
            'Port': '5432',
            'User': 'user',
            'Password': 'password',
            'Database': 'mydatabase'
        }

        self.config['MODEL'] = {
            'ModelPath': './models/model.pkl',
            'Epochs': '10',
            'BatchSize': '32'
        }

        with open(self.config_file, 'w') as configfile:
            self.config.write(configfile)
        print(f"Default configuration created at {self.config_file}")

    def validate_config(self):
        """Validate the configuration file."""
        try:
            log_level = self.config['DEFAULT']['LogLevel']
            if log_level not in ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']:
                raise ValueError("Invalid LogLevel in config file.")
            
            int(self.config['DATABASE']['Port'])
            int(self.config['MODEL']['Epochs'])
            int(self.config['MODEL']['BatchSize'])
        except KeyError as e:
            raise KeyError(f"Missing configuration for {e}")
        except ValueError as e:
            raise ValueError(f"Invalid configuration value: {e}")

    def get_database_config(self):
        """Get the database configuration."""
        db_config = self.config['DATABASE']
        return {
            'host': db_config['Host'],
            'port': int(db_config['Port']),
            'user': db_config['User'],
            'password': db_config['Password'],
            'database': db_config['Database']
        }

    def get_model_config(self):
        """Get the model configuration."""
        model_config = self.config['MODEL']
        return {
            'model_path': model_config['ModelPath'],
            'epochs': int(model_config['Epochs']),
            'batch_size': int(model_config['BatchSize'])
        }

    def update_config(self, section, key, value):
        """Update a configuration value."""
        if section in self.config and key in self.config[section]:
            self.config[section][key] = str(value)
            with open(self.config_file, 'w') as configfile:
                self.config.write(configfile)
            print(f"Updated {section} section: {key} = {value}")
        else:
            print(f"Invalid section or key: {section}/{key}")

    def print_config(self):
        """Print the current configuration."""
        for section in self.config.sections():
            print(f"[{section}]")
            for key in self.config[section]:
                print(f"{key} = {self.config[section][key]}")
            print()

def check_prefix_in_filename(filename, prefixes, names):
    for index, prefix in enumerate(prefixes):
        if prefix in filename:
            return True, names[index]
    return False, None

def configure(config, url):
    prefixes1 = ['SSM', 'NM', 'LMM']
    names1 = ['Superficial spreading melanoma (SSM)', 'Nodular Melanoma (NM)', 'Lentigo Maligna Melanoma (LMM)']
    prefixes2 = ['ND', 'SP', 'PG']
    names2 = ['Nodular BCC', 'Superficial BCC', 'Pigmented BCC']
    prefixes3 = ['BD', 'CSCC', 'KSCC']
    names3 = ['Bowens Disease', 'Conventional SCC', 'Keratoacanthoma-type SCC']
    
    if config == 0:
        return check_prefix_in_filename(url, prefixes1, names1)
    elif config == 1:
        return check_prefix_in_filename(url, prefixes2, names2)
    elif config == 2:
        return check_prefix_in_filename(url, prefixes3, names3)
    else:
        return False, None
