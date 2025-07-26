# Summary: this file implementes a comprehensive logger class with multiple backends (such as WandB) to sync the logs

import logging
import time
from contextlib import contextmanager
from functools import wraps


class Logger(logging.Logger):
    @contextmanager
    def timer(self, counter_name, use_wandb=True, level=logging.DEBUG):
        """
        Context manager to time a function. Example usage:
        with logger.timer("my cool code time"):
            # cool code to run

        Args:
            counter_name: Name of the counter to be logged.
            use_wandb: If true, the time is logged to W&B.
            level: Logging level for the message.
        """
        start_time = time.time()
        yield
        end_time = time.time()
        elapsed_time = end_time - start_time
        self.log(level, f"{counter_name} took {elapsed_time:.4f} seconds")
        if use_wandb:
            self.wandb({counter_name: elapsed_time})

    def wandb(self, *args, **kwargs):
        """
        A function to log data to Weights & Biases if a run is active.

        Parameters:
            *args: positional arguments passed to wandb.log.
            **kwargs: keyword arguments passed to wandb.log.

        Returns:
            None
        """
        try:
            from wandb import log
            from wandb.errors import Error

            try:
                log(*args, **kwargs)
            except Error:
                pass
        except ImportError:
            pass


def getLogger(
    name: str,
    use_root_logger: bool = False,
) -> Logger:
    """
    Returns logger object with the given name. Wrapper for logging.getLogger() function.

    Args:
        name: Name of the logger to be returned.
        use_root_logger: If true, the root logger is returned. This is useful for
            capturing all logger output in one place.

    Returns:
        Logger object with the given name.
    """
    if use_root_logger:
        logger = logging.getLogger()
        logger.name = name
    else:
        logger = logging.getLogger(name)
    # Check if the logger is already an instance of MyLogger
    if not isinstance(logger, Logger):
        # If not, dynamically change class to Logger
        logger.__class__ = Logger
    return logger


def timing_logger(logger, use_wandb=True, level=logging.DEBUG):
    """
    Decorator to log the time taken by a function with a provided logger object.

    Example usage:
    @timing_logger(logger)
    def my_function():
        pass

    Args:
        logger: Logger object to use for logging.
        use_wandb: If true, the time is logged to W&B.
        level: Logging level for the message.
    """

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()
            result = func(*args, **kwargs)
            end_time = time.time()
            elapsed_time = end_time - start_time
            logger.log(level, f"{func.__name__} took {elapsed_time:.4f} seconds")
            if use_wandb:
                try:
                    from wandb import log
                    from wandb.errors import Error

                    try:
                        log({f"{func.__name__}_time": elapsed_time})
                    except Error:
                        pass
                except ImportError:
                    pass
            return result

        return wrapper

    return decorator


def main():
    import argparse

    parser = argparse.ArgumentParser(
        "Logger example. To run all the functionality run something like:\n"
        "    python myo_utils/logger.py --wandb-project-name test"
        " --use-root-logger --path-to-log /tmp/log.log"
    )
    parser.add_argument("--name", type=str, default="test")
    parser.add_argument("--use-root-logger", action="store_true")
    parser.add_argument(
        "--wandb-project-name",
        default=None,
        type=str,
        help="Project name for W&B. If not provided, W&B is not used.",
    )
    parser.add_argument(
        "--wandb-run-id",
        default=None,
        type=str,
        help="Run name used for wandb if --wandb-project is provided.",
    )
    parser.add_argument(
        "--path-to-log",
        default=None,
        type=str,
        help="If provided the logging output goes to the file as well. If --use-root-logger is provided it will capture all the logger output.",
    )
    args = parser.parse_args()

    if args.wandb_project_name is not None:
        import wandb

        wandb.init(project=args.wandb_project_name, id=args.wandb_run_id)

    logger_name = args.name
    # create the logger
    logger = getLogger(logger_name, use_root_logger=args.use_root_logger)
    # sets the logging level
    # any messages with level lower than defined are filtered out for all handler
    logger.setLevel(logging.DEBUG)
    # retrieve the logger for the second time
    logger2 = getLogger(logger_name)
    assert logger is logger2, (
        "Both loggers should be the same object. "
        "use_root_logger argument only matters at first invocation"
    )
    if args.path_to_log is not None:
        # Set up logging to file
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        file_hangler = logging.FileHandler(args.path_to_log)
        # define the logging level
        # level lower than defined is not written to file
        file_hangler.setLevel(logging.DEBUG)
        # define the formatting for writing to file
        file_hangler.setFormatter(formatter)
        logger.addHandler(file_hangler)

    # Create console handler and set level level lower than defined is not printed out
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    # You can set up formatting if needed
    logger.addHandler(console_handler)

    # shows logging example
    logger.debug("Logging debug message")
    logger.info("Logging info message")
    logger.warning("Logging warning message")
    logger.error("Logging error message")
    logger.critical("Logging critical message")

    # shows logging to wandb. Works only if W&B is installed and wandb is initialized
    # otherwise nothing happens
    # Check under the W&B project that logging charts are there
    logger.wandb({"data": 123, "another data": 321})

    # shows how to use the timer context manager
    # use_wandb is optional
    with logger.timer("Logging sleep time", use_wandb=True):
        time.sleep(0.2)

    # shows how to use the timer decorator
    # use_wandb is optional
    @timing_logger(logger, use_wandb=True)
    def sleep_function():
        time.sleep(0.1)

    # actual function call
    sleep_function()


if __name__ == "__main__":
    main()