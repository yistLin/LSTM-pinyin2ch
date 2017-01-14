#!/bin/python
# -*- coding: utf-8 -*-
#encoding:utf8

import os,sys,math,codecs,string,time
import glob
import shutil
import ConfigParser , fnmatch

def convert_simplified_to_traditional(t_str, s_str, o_str):
	str = u""
	for i in range(0, len(o_str)) :
		if s_str.find(o_str[i]) != -1 :
			str += t_str[s_str.index(o_str[i])]
		else :
			str += o_str[i]
	return str


resource_file = 'convert2utf8.res';
if not os.path.isfile (resource_file):
	exit(MSG_RESOURCE_FILE_NOT_FOUND)
config = ConfigParser.ConfigParser()
config.readfp(open(resource_file))
t_str = config.get("mapping","traditional_str")
s_str = config.get("mapping","simplified_str" )

sFilePath = sys.argv[1]
sZh = sys.argv[2]
sMode = sys.argv[3]

print "===== User Set ====="
print "File:" + sFilePath
print "zh(tw,cn):" + sZh
print "sMode(utf8...):" + sMode
print "=====  ====="

pattern_config = ConfigParser.ConfigParser()
pattern_config.readfp(open('cover_config.res'))
# pattern = '*.txt;*.php;*.htm;*.html;*.tpl;*.tmpl;*.js'
pattern = pattern_config.get("mapping","pattern")
patterns = pattern.split(';')
print "rule: " + pattern

for dirPath, dirNames, fileNames in os.walk(sFilePath):
	# print dirPath
	for f in fileNames:
		# print "f:" + f
		is_check_file = 'no'
		for patt in patterns:
			# print "patt:" + patt
			if fnmatch.fnmatch( f , patt ):
				# print "is f:" + f
				is_check_file = 'ok'
				# yield os.path.join(dirPath,name)
				break
		if is_check_file == 'no':
			continue

		result_content = u''
		original_content = u''

		_file_path = os.path.join(dirPath, f)
		# print "load: " + _file_path
		dir_separator = os.path.sep
		current_dir = os.getcwd()
		current_file = _file_path
		target_file = current_dir + dir_separator + current_file
		print "----"
		print "load: " + target_file
		# if not os.path.isfile (current_file): 
		# 	target_file = current_dir + dir_separator + current_file
		# else :
		# 	target_file = current_file
		# if not os.path.isfile (current_file):
		# 	exit( "%s" % current_file)
		fp = open(target_file, 'r')
		original_content = fp.read()
		fp.close()
		# print original_content
		if sZh == 'tw':
			result_content = convert_simplified_to_traditional(t_str.decode('utf8'), s_str.decode('utf8'), original_content.decode(sMode))	
		elif sZh == 'cn':
			result_content = convert_simplified_to_traditional(s_str.decode('utf8'), t_str.decode('utf8'), original_content.decode(sMode))	
		else :
			result_content = convert_simplified_to_traditional(t_str.decode('utf8'), t_str.decode('utf8'), original_content.decode(sMode))	

		if len(result_content) != 0 :
			# shutil.copy2(target_file, target_file + '.bak')
			fp = open(target_file, 'w')
			fp.write(result_content.encode('utf8'))
			fp.close()
			print target_file + " is ok "
			
			# exit(MSG_CONVERT_FINISH)
		# print "----"

