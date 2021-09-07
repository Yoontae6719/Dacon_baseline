import os


import yaml
import socket
import random
import torch
import numpy as np
from path import Path
from typing import Optional
import termcolor
from datetime import datetime

def set_seed(seed=None):
    # type: (Optional[int]) -> int
    """
    set the random seed using the required value (`seed`)
    or a random value if `seed` is `None`
    :return: the newly set seed
    """
    if seed is None:
        seed = random.randint(1, 10000)
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True  # type: ignore
    torch.backends.cudnn.benchmark = True  # type: ignore
    return seed


class Conf(object):
    HOSTNAME = socket.gethostname()
    LOG_PATH = Path('./logs/')

    def __init__(self,
                 conf_file_path = None,
                 seed = None,
                 exp_name = None,
                 log = True):
        # type: (str, int, str, bool) -> None
        """
        Args:
            conf_file_path : optional path of the configuration file
            seed  : desired seed for the RNG. if "None" it will be chosen randomly
            exp_name : name of experiment
            log : "True" if you want to log each step
            api_token : it is Neptune API token
            neptune_path : naptune project path
        """

        self.exp_name = exp_name # it is yaml name
        self.log_each_step = log

        # Define project name and host name
        self.project_name= Path(__file__).parent.parent.basename() # Vegetable
        m_str = f'┃ {self.project_name}@{Conf.HOSTNAME} ┃'
        u_str = '┏' + '━' * (len(m_str) - 2) + '┓'
        b_str = '┗' + '━' * (len(m_str) - 2) + '┛'
        print(u_str + '\n' + m_str + '\n' + b_str)

        # Define output Path
        self.project_log_path = Path('./log')

        # Define random seed
        self.seed = set_seed(seed)             # type: int

        self.keys_to_hide = list(self.__dict__.keys()) + ['keys_to_hide']


        # if the configuration file is not specified
        # try to load a configuration file based on the experiment name
        tmp = Path(__file__).parent / (self.exp_name + ".yaml" )
        if conf_file_path is None and tmp.exists():
            conf_file_path = tmp


        # read the YAML configuation file
        if conf_file_path is None:
            y = {}
        else:
            conf_file = open(conf_file_path, 'r')
            y = yaml.load(conf_file, Loader=yaml.Loader)

        defalut_device = "cuda" if torch.cuda.is_available() else "cpu"

        ########################## Set yaml default value ###############################
        self.lr             = y.get("lr", 5e-4)
        self.target_n       = y.get("target_n", 21)
        self.batch_size     = y.get("batch_size", 128)
        self.epoch          = y.get("epoch", 50)
        self.hidden_layer   = y.get("hidden_layer", 3)
        self.dropout        = y.get("dropout", 0.2)  # type: float
        self.encoder_length = y.get("encoder_length", 28)
        self.decoder_length = y.get("decoder_length", 28)
        self.hidden_dim     = y.get("hidden_dim", 128)
        self.device         = y.get("DEVICE", defalut_device)
        self.model_name     = y.get("model_name", "seq2seq") # type: str
        self.api_token     = y.get("api_token", "eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiJkYjc5ZTEyZC02ZDQ3LTRmMzItODA2Zi03ODM3Y2E4ZDdiNTMifQ==") # type: str
        self.neptune_path     = y.get("neptune_path", "FELAB/TES") # type: str

        self.all_params     = y  # type: dict
        ##################################################################################


        #self.exp_log_path = self.project_log_path / self.all_params["model"] / exp_name / datetime.now().strftime("%m-%d-%Y - %H-%M-%S")
        self.exp_log_path = os.path.join(self.project_log_path, exp_name)
        if not os.path.exists(self.exp_log_path):
            os.makedirs(self.exp_log_path)

    def write_to_file(self, out_file_path):
        # type: (str) -> None
        """
        Writes configuration parameters to `out_file_path`
        :param out_file_path: path of the output file
        """
        import re

        ansi_escape = re.compile(r'\x1B\[[0-?]*[ -/]*[@-~]')
        text = ansi_escape.sub('', str(self))
        with open(out_file_path, 'w') as out_file:
            print(text, file=out_file)

    def __str__(self):
        # type: () -> str

        # colors list : grey, red, green, yellow, blue, megenta, cyan, white
        # attributed list : bold, dark, underline, blink, reverse, concealed

        out_str = ''
        for key in self.__dict__:
            if key in self.keys_to_hide:
                continue
            value = self.__dict__[key]
            if type(value) is Path or type(value) is str:
                value = value.replace(Conf.LOG_PATH, '$LOG_PATH')

                if(value == self.model_name):
                    value = termcolor.colored(value, 'cyan', attrs = ["bold", "reverse"])
                else:
                    value = termcolor.colored(value, 'yellow')
            else:
                value = termcolor.colored(f'{value}', 'magenta', attrs = ["bold"])



            out_str += termcolor.colored(f'{key.upper()}', 'blue')
            out_str += termcolor.colored(' : ', 'red')
            out_str += value
            out_str += '\n'

        return out_str[:-1]


def show_default_params():
    """
    Print default configuration parameters
    """
    cnf = Conf(exp_name='default')
    print(f'\nDefault configuration parameters: \n{cnf}')

if __name__ == '__main__':
    show_default_params()