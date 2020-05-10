"""Load dataset from CodeSearchNet and transform into baseline.

For char-rnn baseline, we just dump all the comments + text. We hope the model
learns to pick up on this autocompletion.

TODO: For the GPT baseline, we introduce the token separators.
"""
from typing import Any, Dict, List, Text

import argparse
import pathlib
import pandas as pd

# Note that we assume these resourcese have already been download by running
# CodeSearchNet/scripts/setup
_FILES: Dict[Text, Any] = {
    'python':
    sorted(
        pathlib.Path('../CodeSearchNet/resources/data/python/').glob(
            '**/*.gz')),
    'java':
    sorted(
        pathlib.Path('../CodeSearchNet/resources/data/java/').glob('**/*.gz')),
    'go':
    sorted(
        pathlib.Path('../CodeSearchNet/resources/data/go/').glob('**/*.gz')),
    'php':
    sorted(
        pathlib.Path('../CodeSearchNet/resources/data/php/').glob('**/*.gz')),
    'javascript':
    sorted(
        pathlib.Path('../CodeSearchNet/resources/data/javascript/').glob(
            '**/*.gz')),
    'ruby':
    sorted(
        pathlib.Path('../CodeSearchNet/resources/data/ruby/').glob('**/*.gz'))
}


def _argument_parser() -> argparse.ArgumentParser:
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

  parser.add_argument(
      '-o',
      '--outpath',
      type=str,
      help='Output path where files for each language to be processed should be '
      'dumped. Relative to script execution',
      default='tensorflow-char-rnn/data')

  return parser


def load_codesearch_net_lite(file_list: List[Text]) -> pd.DataFrame:
  """Loads provide file list into pandas dataframe for analysis."""
  columns: List[Text] = ['code', 'docstring', 'language', 'partition']

  return pd.concat([
      pd.read_json(f, orient='records', compression='gzip',
                   lines=True)[columns] for f in file_list
  ],
                   sort=False)


def tocharrn(df: pd.DataFrame, language: Text) -> Text:
  """Returns the contents of the .txt file to train char-rnn.

  Generally, this is just going to concatenate all of the docstring + code
  snippets.

  We'll set-it up so that we have
  <START>
  <docstring>
  <code>
  <END>
  """
  data = df[df.language == language]
  text_snippets = []
  for _, row in data.iterrows():
    code = row['code']
    doc = row['docstring']
    text_snippets.append(f'{doc}\n{code}')

  return '<START>' + '<END><START>'.join(text_snippets) + '<END>'


def main(args):
  file_list = []
  for language in args.languages:
    file_list += _FILES[language.lower()]
  data = load_codesearch_net_lite(file_list)

  for language in args.languages:
    text = tocharrn(data, language)
    with open(pathlib.Path(args.outpath, f'{language}.txt'), 'w') as out:
      out.write(text)


if __name__ == '__main__':
  parser = _argument_parser()
  args = parser.parse_args()
  main(args)
