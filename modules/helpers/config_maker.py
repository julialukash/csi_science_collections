import configparser
import datetime
from os import path, mkdir

class ConfigpPathes(object):
    """Config class for input
    Auxiliary class for keeping input parameters of project

    Args:
        file_name: Name of input config file (must be in wd)
    """

    def __init__(self, file_name):
        if not path.exists(file_name):
            raise IOError('Input config file not found')

        cfg = configparser.ConfigParser()
        cfg.read(file_name)
        path_cfg = cfg['Paths']
        if path_cfg is None:
            raise AttributeError('Section <<Paths>> is not found into config file')

        home_dir = path_cfg.get('home_dir')
        if home_dir is None:
            raise AttributeError('Field <<home_dir>> is not found into <<Paths>> section')

        self.collection_name = path_cfg.get('collection_name')
        if self.collection_name is None:
            raise AttributeError('Field <<collection_name>> is  not found into <<Paths>> section')

        self.dataset_folder_name = path_cfg.get('dataset_folder_name')
        if self.dataset_folder_name is None:
            raise AttributeError('Field <<dataset_folder_name>> is  not found into <<Paths>> section')

        self.experiment_folder_name = path_cfg.get('experiment_folder_name')
        if self.experiment_folder_name is None:
            raise AttributeError('Field <<experiment_folder_name>> is  not found into <<Paths>> section')

        datase_rel_path = '..\\data\postnauka\\UCI_collections'
        self.dataset_path = path.join(home_dir, datase_rel_path, self.dataset_folder_name)
        if not path.exists(self.dataset_path):
            raise SystemError('Path ' + self.dataset_path + ' not found')
        self.vocabulary_path = path.join(self.dataset_path, 'vocab.' + self.collection_name + '.txt')
        if not path.isfile(self.vocabulary_path):
            raise SystemError('Vocabulary file ' + self.vocabulary_path + ' not found')

        output_batches_rel_dir = '..\\data\postnauka\\bigARTM_files'
        self.output_batches_path = path.join(home_dir, output_batches_rel_dir, self.dataset_folder_name)
        if not path.exists(self.output_batches_path):
            mkdir(self.output_batches_path)
        self.dictionary_path = path.join(home_dir, output_batches_rel_dir, self.collection_name + '_dictionary')

        output_experiments_rel_dir = 'experiments'
        self.experiment_data_path = path.join(home_dir, output_experiments_rel_dir)
        if not path.exists(self.experiment_data_path ):
            mkdir(self.experiment_data_path)
        self.experiment_dataset_folder_name = path.join(self.experiment_data_path, self.dataset_folder_name)
        if not path.exists(self.experiment_dataset_folder_name):
            mkdir(self.experiment_dataset_folder_name)
        self.experiment_path = path.join(self.experiment_dataset_folder_name, self.experiment_folder_name)
        if not path.exists(self.experiment_path):
            mkdir(self.experiment_path)

        self.models_file_name = path.join(self.experiment_path, 'models.txt')
        # models_file = open(models_file_name, 'a')


    def __str__(self):
        return 'data_folder_name = {}, detector_name = {}, mode = {}, parts = {}'\
                .format(self.data_folder_name, self.detector_name, self.mode, self.parts)


class ConfigPaths(object):
    """Config class for paths
    Auxiliary class for keeping all the paths needed in project

    Args:
        input_config: instance of ConfigInput class
    """
    PLAGIARISM_FOLDER_RELATIVE_PATH = '..\\..\\data'
    TEST_MODE_FOLDER_NAME = 'test\\suspicious-documents'
    TRAIN_MODE_FOLDER_NAME = 'train\\intrinsic-detection-corpus'
    OUTPUT_FOLDER_NAME = 'RESULTS'
    PICS_FOLDER_NAME = 'PICS'
    R_FOLDER_NAME = 'r_files'
    LOGS_FOLDER_NAME = 'LOGS'

    def __init__(self, input_config):
        wd = os.getcwd()
        timestamp = datetime.datetime.now()
        time = '{}-{}-{}__{}-{}'.format(timestamp.day, timestamp.month, timestamp.year, timestamp.hour, timestamp.minute)
        self.data_folder_name = input_config.data_folder_name
        self.data_folder_path = os.path.abspath(os.path.join(wd, self.PLAGIARISM_FOLDER_RELATIVE_PATH, input_config.data_folder_name))
        if input_config.mode is Mode.train_grid or input_config.mode is Mode.train:
            self.input_folder_path = os.path.join(self.data_folder_path, self.TRAIN_MODE_FOLDER_NAME)
        elif input_config.mode is Mode.test:
            self.input_folder_path = os.path.join(self.data_folder_path, self.TEST_MODE_FOLDER_NAME)
        if input_config.parts is not None:
            self.input_folder_path = os.path.join(self.input_folder_path, input_config.parts)
        self.input_folder_path = os.path.abspath(self.input_folder_path)
        self.output_folder_path = os.path.abspath(os.path.join(self.data_folder_path, self.OUTPUT_FOLDER_NAME))
        self.pics_folder_path = os.path.abspath(os.path.join(self.data_folder_path, self.PICS_FOLDER_NAME))
        self.r_answers_path = os.path.abspath(os.path.join(self.data_folder_path, self.R_FOLDER_NAME))
        # self.chunks_info_path = ''
        self.log_file_name = 'log__{}_{}_{}__{}.txt'.format(input_config.data_folder_name, input_config.detector_name.value,
                                                            input_config.mode.value, time)
        self.log_folder_path = os.path.abspath(os.path.join(wd, self.PLAGIARISM_FOLDER_RELATIVE_PATH, '..', self.LOGS_FOLDER_NAME))
        self.init_folders()

    def init_folders(self):
        if not os.path.exists(self.input_folder_path):
            raise IOError('Input folder path not found')
        if not os.path.exists(self.log_folder_path):
            os.mkdir(self.log_folder_path)
        if not os.path.exists(self.output_folder_path):
            os.mkdir(self.output_folder_path)
        if not os.path.exists(self.pics_folder_path):
            os.mkdir(self.pics_folder_path)
        if not os.path.exists(self.r_answers_path):
            os.mkdir(self.r_answers_path)
        # if not os.path.exists(self.chunks_info_path):
        #     os.mkdir(self.chunks_info_path)

    def __str__(self):
        return 'data_folder_path = {}, \ninput_folder_path = {}, \noutput_folder_path = {},\n' + \
               'pics_folder_path = {}, \nlog_file_path = {},\nlog_file_name = {}'\
                .format(self.data_folder_path, self.input_folder_path, self.output_folder_path,
                        self.pics_folder_path, self.log_folder_path, self.log_file_name)


