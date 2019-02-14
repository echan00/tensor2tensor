
# coding=utf-8
# Copyright 2018 The Tensor2Tensor Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Query an exported model. Py2 only. Install tensorflow-serving-api."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os, sys, glob
import requests
import codecs, ftfy
from shutil import copyfile
from multiprocessing import Pool
from future.utils import raise_from
class FailedJob(Exception):
    pass
from itertools import chain, islice

from oauth2client.client import GoogleCredentials
from six.moves import input  # pylint: disable=redefined-builtin
from time import sleep

from tensor2tensor import problems as problems_lib  # pylint: disable=unused-import
from tensor2tensor.serving import serving_utils
from tensor2tensor.utils import registry
from tensor2tensor.utils import usr_dir
import tensorflow as tf

flags = tf.flags
FLAGS = flags.FLAGS

flags.DEFINE_string("server", None, "Address to Tensorflow Serving server.")
flags.DEFINE_string("servable_name", None, "Name of served model.")
flags.DEFINE_string("problem", None, "Problem name.")
flags.DEFINE_string("data_dir", None, "Data directory, for vocab files.")
flags.DEFINE_string("t2t_usr_dir", None, "Usr dir for registrations.")
flags.DEFINE_string("inputs_once", None, "Query once with this input.")
flags.DEFINE_integer("timeout_secs", 10, "Timeout for query.")
flags.DEFINE_integer("TFX", 0, "Translate all files in directory for TFX.")
flags.DEFINE_integer("bleualign_upload", 0, "Align and upload files to s3")
flags.DEFINE_string("subdir", None, "dir_001")

# For Cloud ML Engine predictions.
flags.DEFINE_string("cloud_mlengine_model_name", None,
                    "Name of model deployed on Cloud ML Engine.")
flags.DEFINE_string(
    "cloud_mlengine_model_version", None,
    "Version of the model to use. If None, requests will be "
    "sent to the default version.")

number_of_workers = 1  # number of works for pooling
chunksize = 30     # number of jobs before bleualign_upload
max_lines = 500   # number of lines to split each file for supervision
slack_hook = 'https://hooks.slack.com/services/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'
T2T_Model_Path = '/root/T2T_Model'
bleualign_upload_location = "s3://nda-ai/final-dec-14-2018/5_aligned-zh"

def validate_flags():
  """Validates flags are set to acceptable values."""
  if FLAGS.cloud_mlengine_model_name:
    assert not FLAGS.server
    assert not FLAGS.servable_name
  else:
    assert FLAGS.server
    assert FLAGS.servable_name


def make_request_fn():
  """Returns a request function."""
  if FLAGS.cloud_mlengine_model_name:
    request_fn = serving_utils.make_cloud_mlengine_request_fn(
        credentials=GoogleCredentials.get_application_default(),
        model_name=FLAGS.cloud_mlengine_model_name,
        version=FLAGS.cloud_mlengine_model_version)
  else:

    request_fn = serving_utils.make_grpc_request_fn(
        servable_name=FLAGS.servable_name,
        server=FLAGS.server,
        timeout_secs=FLAGS.timeout_secs)
  return request_fn

def chunks(iterable, n):
  "chunks(ABCDE,2) => AB CD E"
  iterable = iter(iterable)
  while True:
    try:
      yield chain([next(iterable)], islice(iterable, n-1))
    except StopIteration:
      return

def convert_file(file):
  problem = registry.problem(FLAGS.problem)
  hparams = tf.contrib.training.HParams(
      data_dir=os.path.expanduser(FLAGS.data_dir))
  problem.get_hparams(hparams)
  if os.path.isfile(T2T_Model_Path+"/4b_zh-tokenized-sample-en/"+file):
    print(file+" exists already")
    return None
  else:
    print(file)
    with codecs.open(T2T_Model_Path+"/4a_zh-tokenized-converted/"+file, mode='w+') as new_file1:
      with codecs.open(T2T_Model_Path+"/4b_zh-tokenized-sample-en/"+file, mode='w+') as new_file:
        with codecs.open("./"+file, mode='r', encoding='utf-8', errors='ignore') as lines:
          for inputs in lines:
            try:
              inputs = ftfy.fix_text(inputs.replace('\n','')).encode('utf-8')
              print(inputs)
              outputs = serving_utils.predict([inputs], problem, make_request_fn())
              outputs, = outputs
              output, score = outputs
              new_file1.write(inputs+'\n')
              new_file.write(output+'\n')
              print(output+'\n')
            except Exception as error:
              print("error: "+str(error))
              print("error input: "+inputs)
              print("error output: "+output)
        new_file.close()    
    return file

def job(file):
    """Wraps any exception that occurs with FailedJob so we can identify which job failed 
    and why""" 
    try:
        return convert_file(file)
    except BaseException as ex:
        raise_from(FailedJob(file), ex)
        #raise FailedJob(file) from ex

def main(_):
  tf.logging.set_verbosity(tf.logging.INFO)
  validate_flags()
  usr_dir.import_usr_dir(FLAGS.t2t_usr_dir)
  problem = registry.problem(FLAGS.problem)
  hparams = tf.contrib.training.HParams(
      data_dir=os.path.expanduser(FLAGS.data_dir))
  problem.get_hparams(hparams)

  if FLAGS.TFX == 1:
    os.chdir(T2T_Model_Path+"/4a_zh-tokenized-converted/"+FLAGS.subdir)
    files = []
    for file in glob.glob("*.txt"):
      files.append(file)
        
    pool = Pool(number_of_workers)
    # we use imap_unordered as we don't care about order, we want the result of the 
    # jobs as soon as they are done
    iter_ = pool.imap_unordered(job, files)
    while True:
        completed = []
        while len(completed) < chunksize:
            # collect results from iterator until we reach the dispatch threshold
            # or until all jobs have been completed
            try:
                result = next(iter_)
            except StopIteration:
                print('all child jobs completed')
                # only break out of inner loop, might still be some completed
                # jobs to dispatch
                break
            except FailedJob as ex:
                print('processing of {} job failed'.format(ex.args[0]))
                sleep(300)
                headers = {'Content-type': 'application/json'}
                data = '{"SERVER ERROR: ":"'+str(FLAGS.subdir)+'-'+str(ex.args[0])+'"}'
                response = requests.post(slack_hook, headers=headers, data=data)
            else:
                completed.append(result)

        if completed:
            print('completed:', completed)
            if FLAGS.bleualign_upload == 1:
              for file in filter(None, completed):
                with codecs.open(T2T_Model_Path+"/4b_zh-tokenized-sample-en/"+file, encoding='utf-8', errors='ignore') as infile, codecs.open("/root/T2T_Model/temp"+FLAGS.subdir+".txt", encoding='utf-8', mode='w') as outfile:
                  for line in infile:
                    if not line.strip(): continue  # skip the empty line
                    outfile.write(line)  # non-empty line. Write it to output
                copyfile(T2T_Model_Path+"/temp"+FLAGS.subdir+".txt", T2T_Model_Path+"/4b_zh-tokenized-sample-en/"+file) 
                os.remove(T2T_Model_Path+"/temp"+FLAGS.subdir+".txt")

                # run bleualign
                cmd = "python3 '"+T2T_Model_Path+"/bleualign/bleualign.py' -v 0 -f sentences --filterthreshold 95 -s '"+T2T_Model_Path+"/4a_zh-tokenized-converted/"+file+"' -t '"+T2T_Model_Path+"/3_en-tokenized/"+file+"' --srctotarget '"+T2T_Model_Path+"/4b_zh-tokenized-sample-en/"+file+"' -o '"+T2T_Model_Path+"/5_aligned-zh/"+file+"'"
                os.system(cmd)

                completed_files = []
                completed_files.append(T2T_Model_Path+"/5_aligned-zh/"+file[0:-4]+".txt-s")
                completed_files.append(T2T_Model_Path+"/5_aligned-zh/"+file[0:-4]+".txt-t")                
                for file_large in completed_files: 
                  with open(file_large) as bigfile:
                    for i, lines in enumerate(chunks(bigfile, max_lines)):
                      file_split = '{}_{}.{}'.format(file_large.split('.')[0], i, file_large.split('.')[1])
                      with open(file_split, 'w') as f:
                        f.writelines(lines)
                  os.remove(file_large)

                # upload to s3
                cmd = "aws s3 sync "+T2T_Model_Path+"/5_aligned-zh "+bleualign_upload_location
                os.system(cmd)

        if len(completed) < chunksize:
            print('all jobs completed and all job completion notifications'
               ' dispatched to central server')
            return

    # notify 'erik' on slack when done
    headers = {'Content-type': 'application/json'}
    data = '{"SERVER DONE: ":"'+str(FLAGS.subdir)+'"}'
    response = requests.post(slack_hook, headers=headers, data=data)

  else:
    while True:
      inputs = FLAGS.inputs_once if FLAGS.inputs_once else input(">> ")
      outputs = serving_utils.predict([inputs], problem, make_request_fn())
      outputs, = outputs
      output, score = outputs
      print_str = """
  Input:
  {inputs}

  Output (Score {score:.3f}):
  {output}
      """
      print(print_str.format(inputs=inputs, output=output, score=score))
      if FLAGS.inputs_once:
        break


if __name__ == "__main__":
  flags.mark_flags_as_required(["problem", "data_dir"])
  tf.app.run()

