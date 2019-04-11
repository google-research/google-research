import pybullet_data
from pybullet_utils.arg_parser import	ArgParser
from pybullet_utils.logger import	Logger
import sys

def	build_arg_parser(args):
		arg_parser = ArgParser()
		arg_parser.load_args(args)

		arg_file = arg_parser.parse_string('arg_file', '')
		if (arg_file !=	''):
				path = pybullet_data.getDataPath()+"/args/"+arg_file
				succ = arg_parser.load_file(path)
				Logger.print2(arg_file)
				assert succ, Logger.print2('Failed to	load args	from:	'	+	arg_file)

		return arg_parser

args = sys.argv[1:]
arg_parser = build_arg_parser(args)
motion_file	=	arg_parser.parse_string("motion_file")
print("motion_file=",motion_file)
bodies = arg_parser.parse_ints("fall_contact_bodies")
print("bodies=",bodies)
int_output_path = arg_parser.parse_string("int_output_path")
print("int_output_path=",int_output_path)