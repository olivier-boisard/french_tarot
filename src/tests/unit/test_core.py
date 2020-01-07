import glob
import os

from french_tarot.core import merge_files


def test_merge_files(request):
    input_folder = os.path.join(os.path.dirname(__file__), os.pardir, 'resources')
    input_filepaths = glob.glob(os.path.join(input_folder, '*'))
    output_filepath = 'tmp.txt'

    request.addfinalizer(lambda: os.remove(output_filepath))

    merge_files(input_filepaths, output_filepath)

    assert count_lines_in_file(output_filepath) == 12


def count_lines_in_file(output_filepath):
    with open(output_filepath) as f:
        n_lines = len(f.readlines())
    return n_lines