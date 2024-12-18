# debug_utils.py
# Utilities for logging, error handling, and performance tracking in Python applications.

import os
import sys
import time
import inspect


def LOG_DEBUG_PRINT(title="", *var_pairs, extra_lines=None):
    """
    Logs debugging information about variables, file location, and function context.

    Args:
        title (str): A title for the debug output.
        *var_pairs (tuple): Pairs of variable names and their values to print.
        extra_lines (list): Additional lines of text to include in the output.
    """
    DARK_BLUE = "\033[34m"
    ORANGE = "\033[33m"
    YELLOW = "\033[93m"
    DARK_GREEN = "\033[32m"
    BOLD = "\033[1m"
    BRIGHT_WHITE = "\033[97m"
    LIGHT_GREY = "\033[90m"
    RESET = "\033[0m"

    frame = inspect.currentframe().f_back
    file_path = frame.f_code.co_filename
    directory = os.path.basename(os.path.dirname(file_path))
    file_name = os.path.basename(file_path)
    function_name = frame.f_code.co_name
    line_number = frame.f_lineno

    print(f"{BOLD}{LIGHT_GREY}{'-' * 95}{RESET}")
    print(
        f"{BOLD}{DARK_BLUE}[DIR: {directory}/{file_name}] {YELLOW}[FUNC: {function_name}] {DARK_GREEN}[LINE: {line_number}]{RESET}")

    if title:
        print(f"{BOLD}{BRIGHT_WHITE}{title}{RESET}\n", end="")

    for pair in var_pairs:
        if pair is not None and isinstance(pair, tuple) and len(pair) == 2:
            var_name, var_value = pair
            if var_value is not None:
                print(f"{BOLD}{ORANGE}{var_name}:{RESET}\n{var_value}\n")

    if extra_lines:
        for line in extra_lines:
            print(f"{BOLD}{ORANGE}NOTE:{RESET} {line}")

    print(f"{BOLD}{LIGHT_GREY}{'-' * 95}{RESET}")
    print("\n\n")


def CUSTOM_RAISE_ERROR(message="An error occurred!", error_type=ValueError):
    """
    Custom error-raising function with detailed context.

    Args:
        message (str): The error message to display.
        error_type (Exception): The type of error to raise (default is ValueError).

    Raises:
        error_type: The specified error with a custom message.
    """
    DARKER_RED = "\033[38;2;180;0;0m"
    DARKER_YELLOW = "\033[38;2;184;134;11m"
    BOLD = "\033[1m"
    ITALIC = "\033[3m"
    RESET = "\033[0m"
    REVERSE = "\033[7m"

    frame = inspect.currentframe().f_back
    file_path = frame.f_code.co_filename
    directory = os.path.basename(os.path.dirname(file_path))
    file_name = os.path.basename(file_path)
    function_name = frame.f_code.co_name
    line_number = frame.f_lineno

    location_info = f"LOCATION: [DIR: {directory}/{file_name}] [FUNC: {function_name}] [LINE: {line_number}]"
    error_body = f"MESSAGE: {message}"

    max_length = max(len(location_info), len(error_body), len(f"Exception Type: {error_type.__name__}")) + 4
    divider_line = '─' * max_length

    error_header = (
        f"\n{BOLD}{DARKER_RED}{REVERSE} !!! ERROR !!! {RESET}\n"
        f"{BOLD}{DARKER_RED}Exception Type: {error_type.__name__}{RESET}\n"
        f"{BOLD}{DARKER_YELLOW}{divider_line}{RESET}\n"
    )

    location_info = (
        f"{BOLD}{DARKER_RED}{location_info}{RESET}\n"
    )

    error_body = (
        f"{BOLD}{DARKER_RED}{ITALIC}{error_body}{RESET}\n"
        f"{BOLD}{DARKER_YELLOW}{divider_line}{RESET}\n"
    )

    complete_message = error_header + location_info + error_body

    sys.tracebacklimit = 0
    raise error_type(complete_message)


def TIME_FUNCTION(func, *args, **kwargs):
    """
    Measures and logs the execution time of a function.

    Args:
        func (callable): The function to time.
        *args: Positional arguments for the function.
        **kwargs: Keyword arguments for the function.

    Returns:
        Any: The result of the function call.
    """
    start_time = time.time()
    result = func(*args, **kwargs)
    elapsed_time = time.time() - start_time

    LOG_DEBUG_PRINT(
        title=f"ELAPSED TIME FOR FUNCTION: {func.__name__}",
        extra_lines=[f"Time taken: {elapsed_time:.2f} seconds"]
    )

    return result

def CONSOLE_BANNER(title):
    """
    Displays a console banner with the given title, formatted for emphasis.

    Args:
        title (str): The title to display within the banner.
    """
    RED = "\033[91m"      # Bright red color
    ORANGE = "\033[33m"   # Orange color
    YELLOW = "\033[93m"   # Bright yellow color
    BOLD = "\033[1m"      # Bold text
    RESET = "\033[0m"     # Reset formatting

    # Create a banner with a top and bottom border
    border = f"{BOLD}{RED}{'=' * 50}{RESET}"  # Red border line
    centered_title = f"{BOLD}{YELLOW}{title.upper().center(50)}{RESET}"  # Centered and uppercase title

    # Print the banner with formatting
    print(border)
    print(centered_title)
    print(border)

if __name__ == "__main__":
    CONSOLE_BANNER("Program Start")
