import logging

import os
import datetime
import importlib
import random
import numpy as np
import torch
from tap import Tap
import pdb

from .serialization import mkdir
from .git_utils import (
    get_git_rev,
    save_git_diff,
)

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def watch(args_to_watch):
    def _fn(args):
        exp_name = []
        for key, label in args_to_watch:
            if not hasattr(args, key):
                continue
            val = getattr(args, key)
            if type(val) == dict:
                val = '_'.join(f'{k}-{v}' for k, v in val.items())
            exp_name.append(f'{label}{val}')
        exp_name = '_'.join(exp_name)
        exp_name = exp_name.replace('/_', '/')
        exp_name = exp_name.replace('(', '').replace(')', '')
        exp_name = exp_name.replace(', ', '-')
        return exp_name
    return _fn

def lazy_fstring(template, args):
    ## https://stackoverflow.com/a/53671539
    return eval(f"f'{template}'")

class Parser(Tap):

    def save(self, output=print):
        fullpath = os.path.join(self.savepath, 'args.json')
        output(f'[ utils/setup ] Saved args to {fullpath}')
        super().save(fullpath, with_reproducibility=False, skip_unpicklable=True)

    def print_cache_until_logfile(self, str):
        if not hasattr(self, 'print_cache'):
            self.print_cache = [str]
        else:
            self.print_cache.append(str)

    def print_cache_free(self):
        if hasattr(self, 'logger') and hasattr(self, 'print_cache'):
            for str in self.print_cache:
                self.logger.info(str)
        self.print_cache = []

    def parse_args(self, experiment=None):
        args = super().parse_args(known_only=True)
        ## if not loading from a config script, skip the result of the setup
        if not hasattr(args, 'config'): return args
        args = self.read_config(args, experiment, output=self.print_cache_until_logfile)
        self.add_extras(args, output=self.print_cache_until_logfile)
        self.eval_fstrings(args, output=self.print_cache_until_logfile)
        self.set_seed(args, output=self.print_cache_until_logfile)
        self.get_commit(args)
        self.generate_exp_name(args, output=self.print_cache_until_logfile)
        self.mkdir(args, output=self.print_cache_until_logfile)
        self.create_logger()
        self.print_cache_free()
        self.save_diff(args)
        return args

    def read_config(self, args, experiment, output=print):
        '''
            Load parameters from config file
        '''
        dataset = args.dataset.replace('-', '_')
        output(f'[ utils/setup ] Reading config: {args.config}:{dataset}')
        module = importlib.import_module(args.config)
        params = getattr(module, 'base')[experiment]

        if hasattr(module, dataset) and experiment in getattr(module, dataset):
            output(f'[ utils/setup ] Using overrides | config: {args.config} | dataset: {dataset}')
            overrides = getattr(module, dataset)[experiment]
            params.update(overrides)
        else:
            output(f'[ utils/setup ] Not using overrides | config: {args.config} | dataset: {dataset}')

        self._dict = {}
        for key, val in params.items():
            setattr(args, key, val)
            self._dict[key] = val

        return args

    def add_extras(self, args, output=print):
        '''
            Override config parameters with command-line arguments
        '''
        extras = args.extra_args
        if not len(extras):
            return

        output(f'[ utils/setup ] Found extras: {extras}')
        assert len(extras) % 2 == 0, f'Found odd number ({len(extras)}) of extras: {extras}'
        for i in range(0, len(extras), 2):
            key = extras[i].replace('--', '')
            val = extras[i+1]
            assert hasattr(args, key), f'[ utils/setup ] {key} not found in config: {args.config}'
            old_val = getattr(args, key)
            old_type = type(old_val)
            output(f'[ utils/setup ] Overriding config | {key} : {old_val} --> {val}')
            if val == 'None':
                val = None
            elif val == 'latest':
                val = 'latest'
            elif old_type in [bool, type(None)]:
                try:
                    val = eval(val)
                except:
                    output(f'[ utils/setup ] Warning: could not parse {val} (old: {old_val}, {old_type}), using str')
            else:
                val = old_type(val)
            setattr(args, key, val)
            self._dict[key] = val

    def eval_fstrings(self, args, output=print):
        for key, old in self._dict.items():
            if type(old) is str and old[:2] == 'f:':
                val = old.replace('{', '{args.').replace('f:', '')
                new = lazy_fstring(val, args)
                output(f'[ utils/setup ] Lazy fstring | {key} : {old} --> {new}')
                setattr(self, key, new)
                self._dict[key] = new

    def set_seed(self, args, output=print):
        if not 'seed' in dir(args):
            return
        output(f'[ utils/setup ] Setting seed: {args.seed}')
        set_seed(args.seed)

    def generate_exp_name(self, args, output=print):
        if not 'exp_name' in dir(args):
            return
        exp_name = getattr(args, 'exp_name')
        if callable(exp_name):
            exp_name_string = exp_name(args)
            output(f'[ utils/setup ] Setting exp_name to: {exp_name_string}')
            setattr(args, 'exp_name', exp_name_string)
            self._dict['exp_name'] = exp_name_string

    def mkdir(self, args, output=print):
        if 'logbase' in dir(args) and 'dataset' in dir(args) and 'exp_name' in dir(args):
            now_str = datetime.datetime.now().strftime('%m%d_%H%M%S')
            args.savepath = os.path.join(args.logbase, args.dataset, args.exp_name, now_str)
            self._dict['savepath'] = args.savepath
            if 'suffix' in dir(args):
                args.savepath = os.path.join(args.savepath, args.suffix)
            if mkdir(args.savepath):
                output(f'[ utils/setup ] Made savepath: {args.savepath}')
                mkdir(os.path.join(args.savepath, 'sample'))
                mkdir(os.path.join(args.savepath, 'test'))
            self.save(output)

    def create_logger(self):
        # Used for console and file logging
        logger = logging.getLogger(__name__)
        logger.propagate = False

        formatter = logging.Formatter("[%(asctime)s] %(levelname)s: %(message)s","%Y-%m-%d %H:%M:%S")

        strHandler = logging.StreamHandler()
        strHandler.setFormatter(formatter)
        logger.addHandler(strHandler)
        logger.setLevel(logging.INFO)

        log_file = os.path.join(self.savepath, 'training.log')
        log_fileHandler = logging.FileHandler(log_file)
        log_fileHandler.setFormatter(formatter)
        logger.addHandler(log_fileHandler)

        self.logger = logger

    def get_commit(self, args):
        args.commit = get_git_rev()

    def save_diff(self, args):
        try:
            save_git_diff(os.path.join(args.savepath, 'diff.txt'))
        except:
            self.logger('[ utils/setup ] WARNING: did not save git diff')
