import os.path as osp
import aim

from enum import Enum
from omegaconf import DictConfig
from reLLMs.logger.base import BaseLogger


def safe_dict(dic):
    def default(o):
        if isinstance(o, type):
            return {'$class': o.__module__ + "." + o.__name__}
        elif isinstance(o, Enum):
            return {
                '$enum': o.__module__ + "." + o.__class__.__name__ + '.' + o.name
            }
        elif callable(o):
            return {
                '$function': o.__module__ + "." + o.__name__
            }
        else:
            return o

    for key, val in dic.items():
        dic[key] = default(val)

    return dic


class AimLogger(BaseLogger):
    def __init__(self):
        super().__init__()
        self._aim_run = None

    def set_snapshot_dir(
        self,
        dir_name,
        description=None,
    ):
        ''' set snapshot dir and init aim run '''
        self._snapshot_dir = dir_name

        run_name = dir_name.split("/")[-1]
        exp_name = dir_name.split("/")[-2]
        run_name_no_timestamp = '_'.join(run_name.split("_")[-3:])
        run_name_no_timestamp_no_seed = '_'.join(run_name.split("_")[-3:-1])
        timestamp = '_'.join(run_name.split("_")[0:2])
        seed_exp_id = run_name.split("_")[-1]
        aim_dir = osp.dirname(osp.dirname(osp.dirname(dir_name)))

        self._aim_run = aim.Run(
            repo=aim_dir,
            experiment=exp_name
        )
        self._aim_run.name = run_name_no_timestamp
        self._aim_run.description = timestamp
        self._aim_run.hash = run_name
        self._aim_run.experiment_description = description
        # Add description and change Run name
        print('########################')
        print('aim outputs to ', aim_dir)
        print('########################')

    def log_variant(self, file_name, variant_data, **kwargs):
        # super(AimLogger, self).log_variant(file_name, variant_data)
        super().log_variant(file_name, variant_data, **kwargs)
        assert isinstance(file_name, str), 'file_name must be std'
        assert isinstance(variant_data, (dict, DictConfig)), 'file_type must be dict or DictConfig'
        file_name = osp.splitext(file_name)[-2]  # Remove the extension name which Aim does not support.
        self._aim_run[file_name] = safe_dict(variant_data)

    def log_scalar(self, scalar, name, step_, context=None):
        self._aim_run.track(value=scalar, name=name, step=step_, context=context)

    def dump_tabular(self, *args, **kwargs):

        tabular_dict = dict(self._tabular)

        # assert  'Epoch' in tabular_dict.keys()
        # epoch = tabular_dict.pop('Epoch')

        # for key, value in tabular_dict.items():
        #     self.log_scalar(value, key, epoch)

        super().dump_tabular(*args, **kwargs)

    def save_params(self, itr, params):
        super().save_params(itr, params)

    def close(self):
        self._aim_run.close()
