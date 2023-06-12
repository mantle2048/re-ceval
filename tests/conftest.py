import pytest
from hydra import compose, initialize
from hydra.utils import instantiate

@pytest.fixture(scope='class')
def model(request):
    with initialize(version_base=None, config_path='../cfgs/model'):
        cfg = compose(config_name=request.param[0], overrides=[f'name={request.param[1]}'])
    return instantiate(cfg)


@pytest.fixture(scope='class')
def task(request):
    with initialize(version_base=None, config_path='../cfgs/task'):
        cfg = compose(config_name=request.param)
    return instantiate(cfg)
