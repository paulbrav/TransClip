from __future__ import annotations

from transclip.product import SERVICE_NAME

from .common import CommandResult, ServiceState, logs_dir, run_command, service_command, toggle_log_path
from .lifecycle import (
    install_daemon,
    service_action,
    service_state,
    uninstall_daemon,
)
from .status import (
    append_toggle_log,
    collect_status,
    last_toggle_log_event,
    run_smoke_test,
    stream_logs,
)

# Platform-specific installers (build_systemd_unit, install_linux_daemon,
# install_macos_daemon, …) are intentionally not re-exported here. Import them
# from their platform submodules (transclip.daemon.linux/macos/windows) for
# tests and adapter work; the public surface is the dispatch layer below.
__all__ = [
    "SERVICE_NAME",
    "CommandResult",
    "ServiceState",
    "append_toggle_log",
    "collect_status",
    "install_daemon",
    "last_toggle_log_event",
    "logs_dir",
    "run_command",
    "run_smoke_test",
    "service_action",
    "service_command",
    "service_state",
    "stream_logs",
    "toggle_log_path",
    "uninstall_daemon",
]
