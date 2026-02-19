import logging
import sys

from code.code_optimizer_ai.config.settings import settings

try:
    from rich.logging import RichHandler
    from rich.console import Console
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False

try:
    import structlog
    STRUCTLOG_AVAILABLE = True
except ImportError:
    STRUCTLOG_AVAILABLE = False


class _KwargLoggerWrapper:

    def __init__(self, logger: logging.Logger):
        self._logger = logger

    def _emit(self, level: str, event: str, **kwargs):
        if kwargs:
            event = f"{event} | {kwargs}"
        getattr(self._logger, level)(event)

    def info(self, event: str, **kwargs):
        self._emit("info", event, **kwargs)

    def warning(self, event: str, **kwargs):
        self._emit("warning", event, **kwargs)

    def error(self, event: str, **kwargs):
        self._emit("error", event, **kwargs)

    def debug(self, event: str, **kwargs):
        self._emit("debug", event, **kwargs)


def setup_logging():
    log_level = getattr(logging, settings.LOG_LEVEL.upper(), logging.INFO)
    
    handlers = []
    if RICH_AVAILABLE:
        console = Console()
        handlers.append(RichHandler(
            console=console,
            show_path=False,
            show_time=False,
            rich_tracebacks=True
        ))
    else:
        handlers.append(logging.StreamHandler(sys.stdout))

    logging.basicConfig(
        level=log_level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="[%Y-%m-%d %H:%M:%S]",
        handlers=handlers
    )
    
    if STRUCTLOG_AVAILABLE:
        structlog.configure(
            processors=[
                structlog.stdlib.filter_by_level,
                structlog.stdlib.add_logger_name,
                structlog.stdlib.add_log_level,
                structlog.stdlib.PositionalArgumentsFormatter(),
                structlog.processors.TimeStamper(fmt="iso"),
                structlog.processors.StackInfoRenderer(),
                structlog.processors.format_exc_info,
                structlog.processors.UnicodeDecoder(),
                structlog.processors.JSONRenderer()
            ],
            context_class=dict,
            logger_factory=structlog.stdlib.LoggerFactory(),
            wrapper_class=structlog.stdlib.BoundLogger,
            cache_logger_on_first_use=True,
        )


def get_logger(name=None):
    """Get a logger for the given module name."""
    if not name:
        frame = sys._getframe(1)
        name = frame.f_globals.get('__name__', 'code_optimizer_ai')

    if STRUCTLOG_AVAILABLE:
        return structlog.get_logger(name)
    return _KwargLoggerWrapper(logging.getLogger(name))


# Initialize logging on module import
setup_logging()
