import argparse
import os
import random
from pathlib import Path
from subprocess import PIPE, run
from typing import Union


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--input_file", "-i", type=Path, required=True)
    parser.add_argument("--num_train_dev_test_lines", "-n", type=int, nargs=3, required=True)
    parser.add_argument("--output_dir", "-o", type=Path, required=True)
    args = parser.parse_args()
    args.input_file = args.input_file.expanduser()
    args.output_dir = args.output_dir.expanduser()
    if any([x < 0 for x in args.num_train_dev_test_lines]):
        parser.error(
            f"All numbers passed to to parameter `--num_train_dev_test_lines` hast to non-negative, whereas "
            f"{args.num_train_dev_test_lines} was given."
        )
    return args


def get_num_lines(input_file: Union[str, os.PathLike]) -> int:
    result = run(['wc', '-l', str(input_file)], stdout=PIPE, stderr=PIPE)
    stdout = result.stdout.decode('utf-8')
    if result.returncode != 0:
        print("STDOUT:", stdout, "STDERR:", result.stderr.decode('utf-8'), sep='\n')
        raise RuntimeError(f"Command `wc -l {input_file}` failed with return code {result.returncode}")
    if not result:
        raise ValueError(
            f"Bash command `wc -l {input_file}` returned and empty string. "
            f"Possibly, file {input_file} does not exist."
        )
    return int(result.stdout.decode('utf-8').split()[0])


def main() -> None:
    args = parse_args()
    num_lines = get_num_lines(args.input_file)
    if num_lines != sum(args.num_train_dev_test_lines):
        raise ValueError(
            f"Wrong split sizes: train - {args.num_train_dev_test_lines[0]}, dev - {args.num_train_dev_test_lines[1]}"
            f", test - {args.num_train_dev_test_lines[2]}, total - {sum(args.num_train_dev_test_lines)}, whereas "
            f"number of lines in file: {num_lines}."
        )
    extracted_lines = set()
    while len(extracted_lines) < args.num_train_dev_test_lines[1] + args.num_train_dev_test_lines[2]:
        extracted_lines.add(random.randrange(num_lines))
    extracted_lines = list(extracted_lines)
    assert len(extracted_lines) == args.num_train_dev_test_lines[1] + args.num_train_dev_test_lines[2]
    random.shuffle(extracted_lines)
    dev_lines = set(extracted_lines[:args.num_train_dev_test_lines[1]])
    test_lines = set(extracted_lines[args.num_train_dev_test_lines[1]:])
    args.output_dir.mkdir(exist_ok=True, parents=True)
    with (args.output_dir / 'dev.jsonl').open('w') as dev_f, \
            (args.output_dir / 'test.jsonl').open('w') as test_f, \
            (args.output_dir / 'train.jsonl').open('w') as train_f, \
            args.input_file.open() as in_f:
        for i, line in enumerate(in_f):
            if i in dev_lines:
                dev_f.write(line)
            elif i in test_lines:
                test_f.write(line)
            else:
                train_f.write(line)


if __name__ == "__main__":
    main()
