#! /usr/bin/env python
#
#   gdynet.py -- Front-end script for running DyNet: Dynamic Neural Network Toolkit
#   function over a large parameter range.
#
#   Copyright (c) 2018 2019 S3IT, University of Zurich, http://www.s3it.uzh.ch/
#
#   This program is free software: you can redistribute it and/or
#   modify
#   it under the terms of the GNU General Public License as published by
#   the Free Software Foundation, either version 3 of the License, or
#   (at your option) any later version.
#
#   This program is distributed in the hope that it will be useful,
#   but WITHOUT ANY WARRANTY; without even the implied warranty of
#   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#   GNU General Public License for more details.
#
#   You should have received a copy of the GNU General Public License
#   along with this program.  If not, see <http://www.gnu.org/licenses/>.
#
"""
Front-end script for running Matlab function
#   function over a large parameter range.
It uses the generic `gc3libs.cmdline.SessionBasedScript` framework.

See the output of ``gdynet.py --help`` for program usage
instructions.

run_transducer.py --dynet-seed 1 --epochs=50 --patience=10 --dropout=0 --transducer=haem \
--sigm2017_format welsh-train-low welsh-dev x-haemsmrt-e50-p10-d0-b1-x/seed-1/welsh-low \
--test-path=welsh-covered-test > ../results/x-haemsmrt-e50-p10-d0-b1-x/seed-1/welsh-low.dev.eval.err

"""

# summary of user-visible changes
__changelog__ = """
  2018-03-07:
  * Use docker image
  2018-02-05:
  * Initial version
"""
__author__ = 'Sergio Maffioletti <sergio.maffioletti@uzh.ch>'
__docformat__ = 'reStructuredText'
__version__ = '0.0.1'
# run script, but allow GC3Pie persistence module to access classes defined here;
# for details, see: http://code.google.com/p/gc3pie/issues/detail?id=95
if __name__ == "__main__":
    import gdynet
    gdynet.GdynetScript().run()

import os
import tempfile

from pkg_resources import Requirement, resource_filename

import gc3libs
import gc3libs.exceptions
from gc3libs import Application, Run, Task
from gc3libs.cmdline import SessionBasedScript, executable_file, existing_directory, existing_file, positive_int
import gc3libs.utils
from gc3libs.quantity import Memory, kB, MB, GB, Duration, hours, minutes, seconds
from gc3libs.workflow import RetryableTask


DEFAULT_OUTPUT_FOLDER = "./results"
CONLL_DEFAULT_LOCATION = "/apps/conll2017"
DOCKER_COMMAND = "sudo docker run -v $PWD/{execution_script}:/tmp/runme.sh -v {conll_folder}:/data -v $PWD/results:/data/results -t -i --entrypoint /tmp/runme.sh smaffiol/dynet:2.0.3-nn "

DOCKER_RUNME_SCRIPT = """#!/bin/bash
echo "[`date`: Start]"
cd /data/src/
pip install docopt
pip install progressbar
{0}
RET=$?
echo "[`date`: End with exit code $RET]"
exit $RET
"""

## utility funtions

def get_execution_script(command_file, foldername, group_cmd):
    """
    For each line of the command file, create a separate
    execution script and include it as part
    of the input files.
    """
    inputs = []
    with open(command_file,'r') as cf:
        # prepare execution script from command
        # commands = cf.readlines()
        commands = [ cmd for cmd in cf if cmd.strip()]
        for i in range(0,len(commands),group_cmd):
            cmd = '\n'.join(commands[i:i+group_cmd])
            try:
                # create script file
                (handle, tmp_filename) = tempfile.mkstemp(dir=foldername,
                                                          prefix='gdynet',
                                                          suffix='.sh')
                                                               
                with open(tmp_filename,'w+') as fd:
                    fd.write(DOCKER_RUNME_SCRIPT.format(cmd))
                inputs.append(fd.name)
            except Exception, ex:
                gc3libs.log.debug("Error creating execution script" +
                                  "Error type: %s." % type(ex) +
                                  "Message: %s"  %ex.message)
                raise
    return inputs
    
## custom application class

class GdynetApplication(Application):
    """
    Custom class to wrap the execution of the exec_file passed as input argument.
    """
    application_name = 'gdynet'
    
    def __init__(self, input_file, conll_folder, **extra_args):

        inputs = dict()
        outputs = []

        inputs[input_file] = os.path.basename(input_file)
        inputs[DEFAULT_OUTPUT_FOLDER] = DEFAULT_OUTPUT_FOLDER

        if conll_folder:
            inputs[conll_folder] = os.path.basename(conll_folder)
            arguments = DOCKER_COMMAND.format(execution_script=inputs[input_file],
                                              conll_folder=inputs[conll_folder])
        else:
            # assume conll folder will be available on execution node
            arguments = DOCKER_COMMAND.format(execution_script=inputs[input_file],
                                              conll_folder=CONLL_DEFAULT_LOCATION)

        outputs.append(DEFAULT_OUTPUT_FOLDER)

        
        Application.__init__(
            self,
            arguments = arguments,
            inputs = inputs,
            outputs = outputs,
            stdout = 'gdynet.log',
            join=True,
            executables = "./{0}".format(inputs[input_file]),
            **extra_args)
        
class GdynetScript(SessionBasedScript):
    """
    Takes 1 input file containing the list of commands to be executed.
    group commands together into single execution
    Then submits one execution for each grouped command.

    The ``gdynet`` command keeps a record of jobs (submitted, executed
    and pending) in a session file (set name with the ``-s`` option); at
    each invocation of the command, the status of all recorded jobs is
    updated, output from finished jobs is collected, and a summary table
    of all known jobs is printed.
    
    Options can specify a maximum number of jobs that should be in
    'SUBMITTED' or 'RUNNING' state; ``gdynet`` will delay submission of
    newly-created jobs so that this limit is never exceeded.

    Once the processing of all chunked files has been completed, ``gdynet``
    aggregates them into a single larger output file located in 
    'self.params.output'.
    """

    def __init__(self):
        SessionBasedScript.__init__(
            self,
            version = __version__,
            application = GdynetApplication, 
            stats_only_for = GdynetApplication,
            )

    def setup_args(self):
        self.add_param('command_file', type=existing_file,
                       help="Path to commands file.")
        
    def setup_options(self):
        self.add_param("-L", "--conll",
                       metavar="[PATH]", 
                       dest="conll_folder",
                       default=None,
                       help="Location of alternative conll fodler. " \
                       " Default: %(default)s.")
        self.add_param("-G", "--group",
                       metavar="[INT]", 
                       dest="group_commands",
                       type=positive_int,
                       default=1,
                       help="Group together consecutive lines from the " \
                       " command file. " \
                       " Default: %(default)s.")

    def parse_args(self):
        """
        Use this method to simply create an empty
        result folder.
        """
        if not os.path.isdir(DEFAULT_OUTPUT_FOLDER):
            gc3libs.log.info("Creating empty result folder [{0}]".format(DEFAULT_OUTPUT_FOLDER))
            os.makedirs(DEFAULT_OUTPUT_FOLDER)

    def new_tasks(self, extra):
        """
        For each line from the command file, produce a single execution sript
        """
        tasks = []

        for exec_file in get_execution_script(self.params.command_file,
                                              os.path.abspath(self.session.name),self.params.group_commands):

            jobname = "gdynet-{0}".format(os.path.basename(exec_file))
            extra_args = extra.copy()
            
            extra_args['jobname'] = jobname
            
            extra_args['output_dir'] = self.params.output
            extra_args['output_dir'] = extra_args['output_dir'].replace('NAME',
                                                                        os.path.join(os.path.basename(exec_file),
                                                                                     jobname))
            extra_args['session_output_dir'] = os.path.dirname(self.params.output)
          
            tasks.append(GdynetApplication(
                exec_file,
                os.path.abspath(self.params.conll_folder),
                **extra_args))
            
        return tasks
