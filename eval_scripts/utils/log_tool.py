"""
Colored terminal logging utility.

Provides timestamped, component-tagged log output with ANSI color support
for use across the evaluation server and related scripts.
"""
from datetime import datetime


class LogTool:
    """Unified logging utility with colored, timestamped output."""

    # ANSI color codes used for terminal formatting
    COLORS = {
        'reset': '\033[0m',
        'bold': '\033[1m',
        'cyan': '\033[36m',
        'green': '\033[32m',
    }

    @staticmethod
    def _get_timestamp():
        return datetime.now().strftime("%H:%M:%S.%f")[:-3]

    @staticmethod
    def _format_message(component, message, indent=0, symbol=""):
        timestamp = LogTool._get_timestamp()
        indent_str = "  " * indent
        return (
            f"{LogTool.COLORS['cyan']}[{timestamp}]{LogTool.COLORS['reset']} "
            f"{LogTool.COLORS['bold']}{component}{LogTool.COLORS['reset']} "
            f"{indent_str}{symbol}{message}"
        )

    @staticmethod
    def info(component, message, indent=0):
        """Log an informational message."""
        print(LogTool._format_message(component, message, indent))

    @staticmethod
    def success(component, message, indent=0):
        """Log a success message with a green checkmark."""
        symbol = f"{LogTool.COLORS['green']}✓{LogTool.COLORS['reset']} "
        print(LogTool._format_message(component, message, indent, symbol))

    @staticmethod
    def section(title):
        """Print a prominent section separator."""
        print(f"\n{LogTool.COLORS['bold']}{'='*60}{LogTool.COLORS['reset']}")
        print(f"{LogTool.COLORS['bold']}{title.center(60)}{LogTool.COLORS['reset']}")
        print(f"{LogTool.COLORS['bold']}{'='*60}{LogTool.COLORS['reset']}")
