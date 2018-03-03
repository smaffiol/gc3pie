#! /usr/bin/env python
#
#   gpyrad.py -- Front-end script for submitting multiple `PyRAD` jobs.
#
#   Copyright (C) 2013, 2014 GC3, University of Zurich
#
#   This program is free software: you can redistribute it and/or modify
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
Front-end script for submitting multiple `PyRAD` jobs.
It uses the generic `gc3libs.cmdline.SessionBasedScript` framework.

See the output of ``gipyrad --help`` for program usage instructions.
"""
__docformat__ = 'reStructuredText'


# summary of user-visible changes
__changelog__ = """

  2014-04-16:
  * Retrieve only specific result folder as output.
  * Initial experimental support for S3 repository

  2014-02-24:
  * Initial release, forked off the ``ggeosphere`` sources.
"""
__author__ = 'Sergio Maffioletti <sergio.maffioletti@gc3.uzh.ch>'
__docformat__ = 'reStructuredText'
__version__ = '$Revision$'


# run script, but allow GC3Pie persistence module to access classes defined here;
# for details, see: https://github.com/uzh/gc3pie/issues/95
if __name__ == "__main__":
    import gipyrad
    gipyrad.GipyradScript().run()

#from pkg_resources import Requirement, resource_filename
import os
import posix

# gc3 library imports
import gc3libs
from gc3libs import Application, Run, Task
from gc3libs.cmdline import SessionBasedScript, executable_file, existing_directory
import gc3libs.utils
from gc3libs.quantity import Memory, kB, MB, GB, Duration, hours, minutes, seconds, GiB
from gc3libs.workflow import RetryableTask
import gc3libs.url

# Default values
FASTQ_FORMATS = ["fastq","fastq.gz"]
DATA_PATH="/data"
IPYRAD_PARAMFILE_NAME="params.txt.modified"
IPYRAD_COMMAND="sudo docker run --rm -v $PWD:{0} smaffiol/ipyrad -p {0}/{1} -s23".format(DATA_PATH,IPYRAD_PARAMFILE_NAME)
IPYRAD_PARAMFILE_PATTERN="sorted_fastq_path"
IPYRAD_PARAMFILE_REPLACEMENT_STRING="{0}/*.fastq.gz             ## [4] [sorted_fastq_path]: Location of demultiplexed/sorted fastq files\n'".format(DATA_PATH)

## Utility methods

def get_valid_input_pair(input_folder):
    """
    scans input folder [non recursively]
    for each valid compressed fastq file [fastq.gz]
    search for a pair of type [R1,R2].
    Search is done at filename level.
    """

    # Get a list of all R1 files
    # for each R1, search for corresponding R2

    R1list = [infile.split('R1')[0] for infile in os.listdir(input_folder) \
              if infile.endswith('R1.fastq.gz')]

    input_list = [(os.path.join(input_folder,
                                "{0}R1.fastq.gz".format(infile.split('R2')[0])),
                   os.path.join(input_folder,
                                "{0}R2.fastq.gz".format(infile.split('R2')[0]))) for infile in os.listdir(input_folder) if infile.endswith('R2.fastq.gz') and infile.split('R2')[0] in R1list]

    return input_list
    
def prepare_ipyrad_param_file(pyrad_param_file):
    """
    Replace input data references and prepare for docker execution
    change [4] to reflect docker invokation:
    /data/*.fastq.gz
    """
    with open(IPYRAD_PARAMFILE_NAME,'w+') as wd:    
        with open(pyrad_param_file,'r') as fd:
            for line in fd:
                if IPYRAD_PARAMFILE_PATTERN in line:
                    wd.write(IPYRAD_PARAMFILE_REPLACEMENT_STRING)
                else:
                    wd.write(line)
    return wd.name

## custom application class

class GipyradApplication(Application):
    """
    Fetches execution wrapper, input file and checks
    whether optional arguments have been passed.
    Namely ''wclust'' that sets the pyRAD clustering
    treshold and ''paramsfile'' needed to run PyRAD.
    """

    application_name = 'pyrad'

    def __init__(self, input_files, dockerimage, paramsfile, **extra_args):
        """
        The wrapper script is being used for start the simulation. 
        """

        inputs = []
        outputs = []

        inputs.append(paramsfile)
        for input_file in input_files:
            inputs.append(input_file)

        Application.__init__(
            self,
            # arguments should mimic the command line interfaca of the command to be
            # executed on the remote end
            arguments = IPYRAD_COMMAND,
            inputs = inputs,
            outputs = outputs,
            stdout = 'gipyrad.log',
            join=True,
            **extra_args)

## main script class

class GipyradScript(SessionBasedScript):
    """
Scan the specified INPUT directories recursively for simulation
directories and submit a job for each one found; job progress is
monitored and, when a job is done, its output files are retrieved back
into the simulation directory itself.

The ``gipyrad`` command keeps a record of jobs (submitted, executed
and pending) in a session file (set name with the ``-s`` option); at
each invocation of the command, the status of all recorded jobs is
updated, output from finished jobs is collected, and a summary table
of all known jobs is printed.  New jobs are added to the session if
new input files are added to the command line.

Options can specify a maximum number of jobs that should be in
'SUBMITTED' or 'RUNNING' state; ``gipyrad`` will delay submission of
newly-created jobs so that this limit is never exceeded.
    """

    def __init__(self):
        SessionBasedScript.__init__(
            self,
            version = __version__,
            application = GipyradApplication,
            stats_only_for = GipyradApplication,
            )

    def setup_args(self):
        self.add_param('input_folder', type=existing_directory,
                       help="Path to folder containing fastq files.")
        
    def setup_options(self):
        self.add_param("-D", "--dockerimage", metavar="STRING",
                       dest="dockerimage", default="smaffiol/ipyrad",
                       help="Docker image to be used to execute ipyrad." \
                       " Default: %(default)s")

        self.add_param("-P", "--params", metavar="PATH",
                       dest="paramsfile", default='./params.txt',
                       help="Location of params.txt file required by pyrad. " \
                       "WARNING: The default params.txt is used to make " \
                       "assumption on the location of pyRAD and the " \
                       "input folder where the *.fastq files are located. " \
                       "Passing an alternative params.txt file might break " \
                       "such assumptions resulting in a falied execution. "  \
                       "Please check the default params.txt file located in " \
                       "'test.pyRAD' folder where gipyrad.py is located.")

    def new_tasks(self, extra):

        paramsfile = prepare_ipyrad_param_file(self.params.paramsfile)
        tasks = []

        # for input_file in self._list_folder_by_url(self.input_folder_url):
        for input_file in get_valid_input_pair(self.params.input_folder):
        # for input_file in [ os.path.join(self.params.input_folder,infile) for infile in os.listdir(self.params.input_folder) if infile.endswith('.fastq') or infile.endswith('.fastq.gz') ]:

            # extract jobname from the 1st file of the input_file pair
            jobname = "%s" % os.path.basename(input_file[0]).split(".")[0]
            
            extra_args = extra.copy()
            extra_args['jobname'] = jobname

            # FIXME: ignore SessionBasedScript feature of customizing 
            # output folder
            extra_args['output_dir'] = self.params.output
                
            extra_args['output_dir'] = extra_args['output_dir'].replace('NAME', jobname)
            extra_args['output_dir'] = extra_args['output_dir'].replace('SESSION', jobname)
            extra_args['output_dir'] = extra_args['output_dir'].replace('DATE', jobname)
            extra_args['output_dir'] = extra_args['output_dir'].replace('TIME', jobname)

            self.log.info("Creating Task: [{0}]".format(jobname))
                
            tasks.append(GipyradApplication(
                input_file,
                self.params.dockerimage,
                paramsfile,
                **extra_args
            ))

        return tasks





