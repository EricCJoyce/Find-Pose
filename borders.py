#  Eric Joyce, Stevens Institute of Technology, 2019

#  Given an OBJ mesh (and all of its texture maps), produce copies of the mesh's texture maps with the face
#  borders drawn on.

#  python borders.py Panel/Panel.obj -ll

import sys
import re
import os
import numpy as np													#  Always necessary
import cv2															#  Core computer vision engine
from face import *													#  Our custom OBJ unpacker

#   argv[0] = borders.py
#   argv[1] = mesh file
#  {argv[2..n] = flags}

def main():
	if len(sys.argv) < 2:  ##########################################  Step 1: check arguments and files
		usage()
		return
	if not os.path.exists(sys.argv[1]):								#  Must have a 3D file
		print('Unable to find mesh file "' + sys.argv[1] + '"')
		return

	params = parseRunParameters()									#  Get command-line options
	if params['helpme']:											#  Did user ask for help?
		usage()														#  Display options
		return

	params['mesh'] = sys.argv[1]									#  Add required argument to the parameter dictionary
	params['meshpath'] = '/'.join(sys.argv[1].split('/')[:-1])		#  so it can get conveniently passed around

	#################################################################  Step 2: Load the OBJ mesh model
	obj = Mesh(params['mesh'], params['meshpath'])					#  Build a model we can query
	obj.reconcile = False											#  In this application we do not care about which
																	#  vertices are "close enough" to call the same.
	obj.texmaporigin = params['origin']								#  Tell Mesh object where its texture map origin should be
	obj.load()

	texmaps = {}													#  Draw the triangle borders of all texture maps
	for f in obj.faces.values():									#  BUild dictionary: key = file name, value = image
		txname = f.texmap.split('.')[0]								#  Make images black-and-white so our red borders
		if txname not in texmaps:									#  will stand out.
			texmaps[txname] = cv2.imread(params['meshpath'] + '/' + f.texmap, cv2.IMREAD_COLOR)
			texmaps[txname] = cv2.cvtColor(texmaps[txname], cv2.COLOR_BGR2GRAY)
			texmaps[txname] = cv2.cvtColor(texmaps[txname], cv2.COLOR_GRAY2RGB)
	for i in range(0, len(obj.faces)):
		txname = obj.faces[i].texmap.split('.')[0]
		cv2.line(texmaps[ txname ], \
			     (int(round(obj.faces[i].t2D[0][0])), int(round(obj.faces[i].t2D[0][1]))), \
			     (int(round(obj.faces[i].t2D[1][0])), int(round(obj.faces[i].t2D[1][1]))), (0, 0, 255), 1)
		cv2.line(texmaps[ txname ], \
			     (int(round(obj.faces[i].t2D[1][0])), int(round(obj.faces[i].t2D[1][1]))), \
			     (int(round(obj.faces[i].t2D[2][0])), int(round(obj.faces[i].t2D[2][1]))), (0, 0, 255), 1)
		cv2.line(texmaps[ txname ], \
			     (int(round(obj.faces[i].t2D[2][0])), int(round(obj.faces[i].t2D[2][1]))), \
			     (int(round(obj.faces[i].t2D[0][0])), int(round(obj.faces[i].t2D[0][1]))), (0, 0, 255), 1)
	for k, v in texmaps.items():									#  Write all annotated texture maps
		cv2.imwrite(k + '.borders.'+params['origin']+'.jpg', v)

	return

#  Parse the command line and set variables accordingly
def parseRunParameters():
	origin = 'ul'													#  Default assumption: origin is upper-left
																	#  of every texture map
	helpme = False

	argtarget = None												#  Current argument to be set
																	#  Permissible setting flags
	flags = ['-ul', '-ll', '-lr', '-ur', \
	         '-?', '-help', '--help']
	for i in range(2, len(sys.argv)):
		if sys.argv[i] in flags:
			argtarget = sys.argv[i]
			if argtarget == '-ll':
				origin = 'll'
			elif argtarget == '-ul':
				origin = 'ul'
			elif argtarget == '-ur':
				origin = 'ur'
			elif argtarget == '-lr':
				origin = 'lr'
			elif argtarget == '-?' or argtarget == '-help' or argtarget == '--help':
				helpme = True

	params = {}
	params['origin'] = origin
	params['helpme'] = helpme

	return params

#  Explain usage of this script and its options to the user.
def usage():
	print('Usage:  python borders.py mesh-filename <options, each preceded by a flag>')
	print(' e.g.:  python borders.py mesh.obj -ll')
	print('Flags:  -ul      tells the script to assume texture map origins are upper-left.')
	print('                 (This is the default assumption.)')
	print('        -ll      tells the script to assume texture map origins are lower-left.')
	print('        -lr      tells the script to assume texture map origins are lower-right.')
	print('        -ur      tells the script to assume texture map origins are upper-right.')
	print('        -?')
	print('        -help')
	print('        --help   displays this message.')
	return

if __name__ == '__main__':
	main()