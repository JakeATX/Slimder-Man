from slimder_man.config.schema import SlimderConfig
from slimder_man.orchestration.skypilot import skypilot_yaml
from slimder_man.orchestration.ssh import ssh_dry_run_commands
from slimder_man.utils.hashing import redact_secret


def test_ssh_and_skypilot_dry_runs_redact():
    cfg = SlimderConfig(project={"paper_faithful": False}, runtime={"backend": "ssh", "ssh": {"host": "host", "user": "user"}})
    cmds = ssh_dry_run_commands(cfg).commands
    assert any("rsync" in c for c in cmds)
    assert any("nvidia-smi" in c for c in cmds)
    yml = skypilot_yaml(cfg)
    assert "accelerators" in yml and "slimder run" in yml
    assert "hf_***REDACTED***" in redact_secret("token=hf_abcdef123")
