import argparse
import signal

from french_tarot.reagent.data_producer import DataProducer
from french_tarot.reagent.play_episodes import play_episodes


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("output_file_path")
    parser.add_argument("--n-max-episodes", type=int, default=None)
    args = parser.parse_args()

    with open(args.output_file_path, "w") as f:
        producer = DataProducer(play_episodes(), f)
        signal.signal(signal.SIGINT, lambda *_: producer.stop())
        producer.run()


if __name__ == "__main__":
    main()
