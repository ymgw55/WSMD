import logging


def get_logger(log_file=None, log_level=logging.INFO, stream=True):

    logger = logging.getLogger(__name__)
    handlers = []
    if stream:
        stream_handler = logging.StreamHandler()
        handlers.append(stream_handler)

    if log_file is not None:
        file_handler = logging.FileHandler(str(log_file), 'w')
        handlers.append(file_handler)

    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    for handler in handlers:
        handler.setFormatter(formatter)
        handler.setLevel(log_level)
        logger.addHandler(handler)

    logger.setLevel(log_level)

    return logger
