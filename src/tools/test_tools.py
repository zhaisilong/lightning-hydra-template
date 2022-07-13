from log import get_logger, run_once

def _log():
    log = get_logger(level='DEBUG')
    log.info('haha')
    log.debug('debug')

def test_once():
    @run_once
    def once():
        return 0

    def many():
        return 1

    assert once() == 0
    assert once() == None
    assert many() == 1
    assert many() == 1


if __name__ == "__main__":
    _log()
    test_once()
