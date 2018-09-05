#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import errno
import json


def module_exists(module_name):
    try:
        __import__(module_name)
    except ImportError:
        return False
    else:
        return True


def choose_first_existing_path(path_list):
    for path in path_list:
        if os.path.exists(path):
            return path
    return None


def get_display_availability():
    return "DISPLAY" in os.environ


def get_filepaths(dir_path, endswith_str="", startswith_str=""):
    if os.path.isdir(dir_path):
        image_filepaths = []
        for path, dnames, fnames in os.walk(dir_path):
            image_filepaths.extend([os.path.join(path, x) for x in fnames if x.endswith(endswith_str) and x.startswith(startswith_str)])
        return image_filepaths
    else:
        raise NotADirectoryError(errno.ENOENT, os.strerror(errno.ENOENT), dir_path)


def get_dir_list_filepaths(dir_path_list, endswith_str="", startswith_str=""):
    image_filepaths = []
    for dir_path in dir_path_list:
        image_filepaths.extend(get_filepaths(dir_path, endswith_str=endswith_str, startswith_str=startswith_str))
    return image_filepaths


def save_json(filepath, data):
    with open(filepath, 'w') as outfile:
        json.dump(data, outfile)


def split_list_into_chunks(l, n, pad=False):
    """Yield successive n-sized chunks from l."""
    for i in range(0, len(l), n):
        if pad:
            chunk = l[i:i + n]
            if len(chunk) < n:
                chunk.extend([chunk[-1]]*(n - len(chunk)))
            yield chunk
        else:
            yield l[i:i + n]


def main():
    l = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    batches = split_list_into_chunks(l, 4, pad=True)
    for batch in batches:
        print(batch)


if __name__ == '__main__':
    main()
