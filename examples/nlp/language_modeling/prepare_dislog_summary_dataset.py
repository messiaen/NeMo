import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--input", "-i", type=Path, required=True)
    parser.add_argument("--output", "-o", type=Path, required=True)
    parser.add_argument("--task_name", "-t", required=True)
    parser.add_argument("--keys_to_keep", "-k", required=True, nargs="+")
    args = parser.parse_args()
    args.input = args.input.expanduser()
    args.output = args.output.expanduser()
    return args


def main() -> None:
    args = parse_args()
    with args.input.open() as in_f, args.output.open('w') as out_f:
        for line in in_f:
            d = {k: v for k, v in json.loads(line) if k in args.keys_to_keep}
            d['taskname'] = args.task_name
            new_line = json.dumps(d)
            out_f.write(new_line + '\n')


if __name__ == "__main__":
    main()