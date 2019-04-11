from pybullet_utils.logger import Logger
logger = Logger()
logger.configure_output_file("e:/mylog.txt")
for i in range (10):
	logger.log_tabular("Iteration", 1)
Logger.print2("hello world")

logger.print_tabular()
logger.dump_tabular()