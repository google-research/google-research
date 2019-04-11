import pybullet_utils.mpi_util as MPIUtil

"""

Some simple logging functionality, inspired by rllab's logging.
Assumes that each diagnostic gets logged each iteration

Call logz.configure_output_file() to start logging to a 
tab-separated-values file (some_file_name.txt)

To load the learning curves, you can do, for example

A = np.genfromtxt('/tmp/expt_1468984536/log.txt',delimiter='\t',dtype=None, names=True)
A['EpRewMean']

"""

import os.path as osp, shutil, time, atexit, os, subprocess

class Logger:
    def print2(str):
        if (MPIUtil.is_root_proc()):
            print(str)
        return

    def __init__(self):
        self.output_file = None
        self.first_row = True
        self.log_headers = []
        self.log_current_row = {}
        self._dump_str_template = ""
        return

    def reset(self):
        self.first_row = True
        self.log_headers = []
        self.log_current_row = {}
        if self.output_file is not None:
            self.output_file = open(output_path, 'w')
        return

    def configure_output_file(self, filename=None):
        """
        Set output directory to d, or to /tmp/somerandomnumber if d is None
        """
        self.first_row = True
        self.log_headers = []
        self.log_current_row = {}

        output_path = filename or "output/log_%i.txt"%int(time.time())

        out_dir = os.path.dirname(output_path)
        if not os.path.exists(out_dir) and MPIUtil.is_root_proc():
            os.makedirs(out_dir)

        if (MPIUtil.is_root_proc()):
            self.output_file = open(output_path, 'w')
            assert osp.exists(output_path)
            atexit.register(self.output_file.close)

            Logger.print2("Logging data to " + self.output_file.name)
        return

    def log_tabular(self, key, val):
        """
        Log a value of some diagnostic
        Call this once for each diagnostic quantity, each iteration
        """
        if self.first_row and key not in self.log_headers:
            self.log_headers.append(key)
        else:
            assert key in self.log_headers, "Trying to introduce a new key %s that you didn't include in the first iteration"%key
        self.log_current_row[key] = val
        return

    def get_num_keys(self):
        return len(self.log_headers)

    def print_tabular(self):
        """
        Print all of the diagnostics from the current iteration
        """
        if (MPIUtil.is_root_proc()):
            vals = []
            Logger.print2("-"*37)
            for key in self.log_headers:
                val = self.log_current_row.get(key, "")
                if isinstance(val, float):
                    valstr = "%8.3g"%val
                elif isinstance(val, int):
                    valstr = str(val)
                else: 
                    valstr = val
                Logger.print2("| %15s | %15s |"%(key, valstr))
                vals.append(val)
            Logger.print2("-" * 37)
        return

    def dump_tabular(self):
        """
        Write all of the diagnostics from the current iteration
        """
        if (MPIUtil.is_root_proc()):
            if (self.first_row):
                self._dump_str_template = self._build_str_template()

            vals = []
            for key in self.log_headers:
                val = self.log_current_row.get(key, "")
                vals.append(val)
            
            if self.output_file is not None:
                if self.first_row:
                    header_str = self._dump_str_template.format(*self.log_headers)
                    self.output_file.write(header_str + "\n")

                val_str = self._dump_str_template.format(*map(str,vals))
                self.output_file.write(val_str + "\n")
                self.output_file.flush()

        self.log_current_row.clear()
        self.first_row=False
        return

    def _build_str_template(self):
        num_keys = self.get_num_keys()
        template = "{:<25}" * num_keys
        return template
