import os
import pickle
from typing import Any


def to_pickle(obj: Any, path: str, force: bool = False) -> str:
    if force:
        with open(path, 'wb') as f:
            pickle.dump(obj, f)
        print(f'pickle in to {path} in force')
        return path
    else:
        if exists_pickle(path):
            print(f'pickle already exists in {path} skip dumping')
            return path
        else:
            with open(path, 'wb') as f:
                pickle.dump(obj, f)
            return path


def from_pickle(path: str) -> Any:
    if exists_pickle(path):
        with open(path, 'rb') as f:
            _ = pickle.load(f)
        return _
    else:
        return None

def exists_pickle(path: str) -> bool:
    if os.path.exists(path):
        return True
    else:
        print(f'cannot find pickle file in {path}')
        return False
