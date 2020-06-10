#!/usr/bin/env python3

from typing import Text, Dict, Any, List

import fire
import json
import os
import pathlib
import numpy as np
import pandas as pd
import tensorflow as tf

import nltk
import model, sample, encoder

_TEST_FILE = sorted(pathlib.Path('../CodeSearchNet/resources/data/python/final/jsonl/test').glob('**/*.gz'))

_TRIPLE_QUOTES = "\"\"\""

def extract_definition_and_documentation(code: Text):
  docstart = code.find(_TRIPLE_QUOTES)
  docend = code.find(_TRIPLE_QUOTES, docstart + 1)
  return code[:docend + 3]

def load_codesearch_net_lite(file_list: List[Text]) -> pd.DataFrame:
  """Loads provide file list into pandas dataframe for analysis."""
  columns: List[Text] = ['code', 'docstring', 'language', 'partition']

  return pd.concat([
	  pd.read_json(f, orient='records', compression='gzip',
				   lines=True)[columns] for f in file_list
  ],
				   sort=False)

def generate_output(
	enc,
	nsamples,
	sess,
	context,
	output,
	leading_text
):
	context_tokens = enc.encode(leading_text)
	out = sess.run(output, feed_dict={
		context: [context_tokens]
	})[:, len(context_tokens):]
	return enc.decode(out[0])

def interact_model(
	model_name='117M',
	seed=None,
	nsamples=1,
	batch_size=1,
	length=None,
	temperature=1,
	top_k=0,
	top_p=0.0,
	output_file='output.txt',
	test_cutoff=999,
	bleu_cutoff=0.4,
):
	"""
	Interactively run the model
	:model_name=117M : String, which model to use
	:seed=None : Integer seed for random number generators, fix seed to reproduce
	 results
	:nsamples=1 : Number of samples to return total
	:batch_size=1 : Number of batches (only affects speed/memory).	Must divide nsamples.
	:length=None : Number of tokens in generated text, if None (default), is
	 determined by model hyperparameters
	:temperature=1 : Float value controlling randomness in boltzmann
	 distribution. Lower temperature results in less random completions. As the
	 temperature approaches zero, the model will become deterministic and
	 repetitive. Higher temperature results in more random completions.
	:top_k=0 : Integer value controlling diversity. 1 means only 1 word is
	 considered for each step (token), resulting in deterministic completions,
	 while 40 means 40 words are considered at each step. 0 (default) is a
	 special setting meaning no restrictions. 40 generally is a good value.
	:top_p=0.0 : Float value controlling diversity. Implements nucleus sampling,
	 overriding top_k if set to a value > 0. A good setting is 0.9.
	"""
	if batch_size is None:
		batch_size = 1
	assert nsamples % batch_size == 0

	enc = encoder.get_encoder(model_name)
	hparams = model.default_hparams()
	with open(os.path.join('models', model_name, 'hparams.json')) as f:
		hparams.override_from_dict(json.load(f))

	if length is None:
		length = hparams.n_ctx // 2
	elif length > hparams.n_ctx:
		raise ValueError("Can't get samples longer than window size: %s" % hparams.n_ctx)

	with tf.Session(graph=tf.Graph()) as sess:
		context = tf.placeholder(tf.int32, [batch_size, None])
		np.random.seed(seed)
		tf.set_random_seed(seed)
		output = sample.sample_sequence(
			hparams=hparams, length=length,
			context=context,
			batch_size=batch_size,
			temperature=temperature, top_k=top_k, top_p=top_p
		)

		saver = tf.train.Saver()
		ckpt = tf.train.latest_checkpoint(os.path.join('models', model_name))
		saver.restore(sess, ckpt)
		data = load_codesearch_net_lite(_TEST_FILE)
		avg_bleu = 0
		with open(output_file, "w") as out:
			for i, row in data.iterrows():
				code = row['code']
				input_snippet = extract_definition_and_documentation(code)
				output_code = generate_output(
						enc,
						nsamples,
						sess,
						context,
						output,
						input_snippet)
				output_code = output_code[:output_code.find("<|endoftext|>")]
				output_code = output_code[:output_code.find("<END>")]
				BLEUscore = nltk.translate.bleu_score.sentence_bleu([code], output_code)
				avg_bleu += BLEUscore
				out.write(f"\ni = {i}, bleu = {BLEUscore}")
				print(f"i = {i}, bleu = {BLEUscore}, avg_bleu = {avg_bleu/(i + 1)}")
				if BLEUscore > bleu_cutoff:
					out.write(f"\ninput_snippet = {input_snippet}, output_code = {output_code}")
					out.flush()
				if i > test_cutoff: break
			avg_bleu /= i
			print(f"Bleu score = {avg_bleu}")
			out.write(f"\nBleu score = {avg_bleu}")
		

if __name__ == '__main__':
	fire.Fire(interact_model)
