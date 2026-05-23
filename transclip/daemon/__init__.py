from __future__ import annotations

from transclip.product import SERVICE_NAME

from . import linux as linux_daemon
from . import macos as macos_daemon
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

build_systemd_unit = linux_daemon.build_systemd_unit
install_linux_daemon = linux_daemon.install_linux_daemon
install_macos_daemon = macos_daemon.install_macos_daemon

__all__ = [
    "SERVICE_NAME",
    "CommandResult",
    "ServiceState",
    "append_toggle_log",
    "build_systemd_unit",
    "collect_status",
    "install_daemon",
    "install_linux_daemon",
    "install_macos_daemon",
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
