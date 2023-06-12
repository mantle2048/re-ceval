"""
File taken from RLKit (https://github.com/vitchyr/rlkit).
Based on rllab's logger.
https://github.com/rll/rllab
"""
import os
import os.path as osp
import datetime
import dateutil.tz
import json

from reLLMs.logger.aim import AimLogger


def safe_json(data):
    if data is None:
        return True
    elif isinstance(data, (bool, int, float)):
        return True
    elif isinstance(data, (tuple, list)):
        return all(safe_json(x) for x in data)
    elif isinstance(data, dict):
        return all(isinstance(k, str) and safe_json(v) for k, v in data.items())
    return False


def dict_to_safe_json(d):
    """
    Convert each value in the dictionary into a JSON'able primitive.
    :param d:
    :return:
    """
    new_d = {}
    for key, item in d.items():
        if safe_json(item):
            new_d[key] = item
        else:
            if isinstance(item, dict):
                new_d[key] = dict_to_safe_json(item)
            else:
                new_d[key] = str(item)
    return new_d


def create_run_name(
    exp_name,
    seed=0,
    with_timestamp=True,
):
    """
    Create a semi-unique experiment name that has a timestamp
    :param prefix:
    :param exp_id:
    :return:
    """
    now = datetime.datetime.now(dateutil.tz.tzlocal())
    if with_timestamp:
        timestamp = now.strftime('%Y-%m-%d_%H-%M-%S')
        return "%s_%s_%d" % (timestamp, exp_name, seed)
    else:
        return "%s_%d" % (exp_name, seed)


def create_log_dir(
    exp_name,
    seed=0,
    base_log_dir=None,
    prefix=None,
    include_prefix_sub_dir=True,
):
    """
    Creates and returns a unique log directory.
    :param prefix: All experiments with this prefix will have log
    directories be under this directory.
    :param exp_id: The number of the specific experiment run within this
    experiment.
    :param base_log_dir: The directory where all log should be saved.
    :return:
    """
    run_name = create_run_name(exp_name, seed=seed)
    assert base_log_dir is not None

    if include_prefix_sub_dir:
        # log_dir = osp.join(base_log_dir, prefix.replace("_", "-"), exp_name)
        log_dir = osp.join(base_log_dir, exp_name, run_name)
    else:
        log_dir = osp.join(base_log_dir, exp_name)
    if osp.exists(log_dir):
        print("WARNING: Log directory already exists {}".format(log_dir))
    os.makedirs(log_dir, exist_ok=True)

    print('########################')
    print('logging outputs to ', log_dir)
    print('########################')

    return log_dir


def setup_logger(
    exp_name="default",
    variant=None,
    text_log_file="debug.log",
    variant_log_file="variant.json",
    tabular_log_file="progress.csv",
    snapshot_mode="last",
    snapshot_gap=10,
    log_tabular_only=False,
    base_log_dir=None,
    prefix=None,
    description=None,
    **create_log_dir_kwargs
):
    """
    Set up logger to have some reasonable default settings.
    Will save log output to
        based_log_dir/exp_name/run_name.
    exp_name will be auto-generated to be unique.
    If log_dir is specified, then that directory is used as the output dir.
    :param exp_name: The sub-directory for this specific experiment.
    :param variant:
    :param text_log_file:
    :param variant_log_file:
    :param tabular_log_file:
    :param snapshot_mode:
    :param log_tabular_only:
    :param snapshot_gap:
    :param base_log_dir:
    :return:
    """
    logger = AimLogger()

    if prefix:
        exp_name = f"{prefix}{exp_name}"

    log_dir = create_log_dir(
        exp_name,
        base_log_dir=base_log_dir,
        prefix=prefix,
        **create_log_dir_kwargs
    )

    if variant is not None:
        logger.log("Variant:")
        logger.log(json.dumps(dict_to_safe_json(variant), indent=2))
        variant_log_path = osp.join(log_dir, variant_log_file)
        logger.log_variant(variant_log_path, variant)

    tabular_log_path = osp.join(log_dir, tabular_log_file)
    text_log_path = osp.join(log_dir, text_log_file)

    logger.add_text_output(text_log_path)
    logger.add_tabular_output(tabular_log_path)
    logger.set_snapshot_dir(log_dir, description)
    logger.set_snapshot_mode(snapshot_mode)
    logger.set_snapshot_gap(snapshot_gap)
    logger.set_log_tabular_only(log_tabular_only)
    exp_name = log_dir.split("/")[-1]
    logger.push_prefix("[%s] " % exp_name)

    return logger
