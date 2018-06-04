"""
test main
"""
from datagenerator import DataGenerator
import configparser


def process_config(conf_file):
    params = {}
    config = configparser.ConfigParser()
    config.read(conf_file)
    for section in config.sections():
        if section == 'DataSetHG':
            for option in config.options(section):
                params[option] = eval(config.get(section, option))
        if section == 'blouse':
            for option in config.options(section):
                params[option] = eval(config.get(section, option))
        if section == 'dress':
            for option in config.options(section):
                params[option] = eval(config.get(section, option))
        if section == 'outwear':
            for option in config.options(section):
                params[option] = eval(config.get(section, option))
        if section == 'skirt':
            for option in config.options(section):
                params[option] = eval(config.get(section, option))
        if section == 'trousers':
            for option in config.options(section):
                params[option] = eval(config.get(section, option))
    return params


if __name__ == '__main__':
    print('--Parsing Config File')
    params = process_config('config.cfg')
    print('--Creating Dataset')
    data_processor = DataGenerator(params['total_joints_list'], params['blouse_joints_list'], params['dress_joints_list'],
                                   params['outwear_joints_list'], params['skirt_joints_list'], params['trousers_joints_list'],
                                   params['blouse_index'], params['dress_index'], params['outwear_index'], params['skirt_index'],
                                   params['trousers_index'], params['img_directory'], params['training_data_file'])
    dataset_train_table, dataset_train_dict = data_processor._read_train_data()
    print(dataset_train_table)
    print(dataset_train_dict['Images/trousers/5a7098edc6a584436c85e573f468afe5.jpg'])
