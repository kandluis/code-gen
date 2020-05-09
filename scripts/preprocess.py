"""Load dataset from CodeSearchNet and transform into baseline.

For char-rnn baseline, we just dump all the comments + text. We hope the model
learns to pick up on this autocompletion.

TODO: For the GPT baseline, we introduce the token separators.
"""
import argparse
import pathlib
import pandas as pd


def _argument_parser():
  """Retrives the command lines available for this script."""
  parser = argparse.ArgumentParser(
      description='Transform CodeSearchNet dataset')
  parser.add_argument(
      '-l',
      '--languages',
      nargs='+',
      help='The programming languages from CodeSearchNet to transform into '
      'other suitable input formats.',
      required=True)

  return parser


def load_codesearch_net(file_list):
  columns = ['code', 'docstring', 'language', 'partition']

  return pd.concat([
      pd.read_json(f, orient='records', compression='gzip',
                   lines=True)[columns] for f in file_list
  ],
                   sort=False)


if __name__ == '__main__':
  parser = _argument_parser()
  args = parser.parse_args()
  files = {
      'python':
      sorted(
          pathlib.Path('../CodeSearchNet/resources/data/python/').glob(
              '**/*.gz')),
      'java':
      sorted(
          pathlib.Path('../CodeSearchNet/resources/data/java/').glob(
              '**/*.gz')),
      'go':
      sorted(
          pathlib.Path('../CodeSearchNet/resources/data/go/').glob('**/*.gz')),
      'php':
      sorted(
          pathlib.Path('../CodeSearchNet/resources/data/php/').glob(
              '**/*.gz')),
      'javascript':
      sorted(
          pathlib.Path('../CodeSearchNet/resources/data/javascript/').glob(
              '**/*.gz')),
      'ruby':
      sorted(
          pathlib.Path('../CodeSearchNet/resources/data/ruby/').glob(
              '**/*.gz'))
  }
