import os

from french_tarot.play import create_batch


def test_convert_to_timeline_format(request):
    batch_output_filepath = os.path.join(".tmp", "dummy.json")
    request.addfinalizer(lambda: os.remove(batch_output_filepath))

    create_batch(n_episodes=2, output_file_path=batch_output_filepath)
