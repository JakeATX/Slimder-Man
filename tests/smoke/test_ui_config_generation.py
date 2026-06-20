import warnings

from slimder_man.config.schema import SlimderConfig
from slimder_man.ui.app import build_config_yaml, create_app
import yaml


def test_ui_config_generation():
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        app = create_app(test_mode=True)
    assert app
    assert not any("gr.Tabs() can only contain gr.Tab()" in str(item.message) for item in caught)
    data = yaml.safe_load(build_config_yaml())
    cfg = SlimderConfig.model_validate(data)
    assert cfg.project.paper_faithful
    assert cfg.quantization.enabled is False
