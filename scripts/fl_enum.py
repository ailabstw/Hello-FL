from enum import Enum
import string

class LogLevel(Enum):
    INFO = 1
    WARNING = 2
    ERROR = 3


def PackageLogMsg(loglevel: LogLevel, message: string)-> object:
    return {"level":loglevel.name, "message":message}

def UnPackageLogMsg(log :object):
    return log["level"] , log["message"]