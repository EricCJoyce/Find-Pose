#  Eric Joyce, Stevens Institute of Technology, 2019

#  Given an image, an OBJ mesh, and a camera matrix file, estimate where in the mesh the picture of the object
#  seen in the image was taken.

#  python find.py image.JPG Panel/Panel.obj -K SONY-DSLR-A580-P30mm.mat -ll -v -m 0 0 0 -m 0 0 255 -SIFT N -ORB N -showInliers -reprojErr 1.0 -o P -o Blender -showFeatures
#  python find.py image.JPG matterpak/mesh.obj -K SONY-DSLR-A580-P30mm.mat -ll -v -showInliers -reprojErr 1.0 -iter 100000 -o P -showFeatures

import sys
import re
import os
import numpy as np													#  Always necessary
import cv2															#  Core computer vision engine
import matplotlib.pyplot as plt										#  Optionally display intermediate images
from face import *													#  Our custom OBJ unpacker

#   argv[0] = find.py
#   argv[1] = query image (photo)
#   argv[2] = mesh file
#  {argv[3..n] = flags}

def main():
	if len(sys.argv) < 3:  ##########################################  Step 1: check arguments and files
		usage()
		return
	if not os.path.exists(sys.argv[1]):								#  Must have a query image
		print('Unable to find query image "' + sys.argv[1] + '"')
		return
	if not os.path.exists(sys.argv[2]):								#  Must have a 3D file
		print('Unable to find mesh file "' + sys.argv[2] + '"')
		return

	params = parseRunParameters()									#  Get command-line options
	if params['helpme']:											#  Did user ask for help?
		usage()														#  Display options
		return

	if not os.path.exists(params['Kmat']):							#  Must have a camera intrinsics file
		print('Unable to find camera intrinsics file "' + params['Kmat'] + '"')
		return

	params['photo'] = sys.argv[1]									#  Add required arguments to the parameter dictionary
	params['mesh'] = sys.argv[2]									#  so they can get conveniently passed around
	params['meshpath'] = '/'.join(sys.argv[2].split('/')[:-1])

	if params['verbose']:											#  Give a run-down of this script-call's settings
		print('>>> Searching for the object seen in "'+params['photo']+'" in the 3D mesh "'+params['mesh']+'".')
		print('>>> Target object was seen by a camera described in "'+params['Kmat']+'".')
		if params['SIFT']:
			print('>>> We will use SIFT detectors.')
		if params['ORB']:
			print('>>> We will use ORB detectors.')
		if params['BRISK']:
			print('>>> We will use BRISK detectors.')
		if params['showFeatures']:
			print('>>> We will write detected features to a new image, "features.jpg", and a new point cloud, "features.ply".')
		if len(params['maskedcolors']) > 0:
			for color in params['maskedcolors']:
				print('>>> We will mask out the color ('+str(color[0])+', '+str(color[1])+', '+str(color[2])+') in the texture maps.')
			print('>>> We will erode the unmasked texture map by '+str(params['erosion'])+'.')
		if params['iter'] > 0:
			print('>>> RANSAC will run for '+str(params['iter'])+' iterations and excuse projections off by at most '+str(params['reprojErr'])+' pixels.')
		else:
			print('>>> RANSAC will run adaptively until '+str(params['confidence'])+' confident, with initial outlier ratio assumed '+str(params['outlierRatio'])+'.')
		print('>>> We will write inliers the "inliers.log" file.')
		for outputformat in params['output']:
			if outputformat == 'rt':
				print('>>> We will write OpenCV\'s rotation and translation 3-vectors to the "find.log" file.')
			elif outputformat == 'Rt':
				print('>>> We will write the rotation matrix and (camera center) translation vector to the "find.log" file.')
			elif outputformat == 'K':
				print('>>> We will write the intrinsic matrix to the "find.log" file.')
			elif outputformat == 'Ext':
				print('>>> We will write the extrinsic matrix to the "find.log" file.')
			elif outputformat == 'P':
				print('>>> We will write the projection matrix to the "find.log" file.')
			elif outputformat == 'Meshlab':
				print('>>> We will write Meshlab pose code to the "find.log" file.')
			elif outputformat == 'Blender':
				print('>>> We will write Blender pose code to the "find.log" file.')
		print('>>> Starting search.\n')

	#################################################################  Step 2: store key-points and descriptors
	imgformats = ['png', 'jpg', 'jpeg']								#          for the query image (photo)
	photo = cv2.imread(params['photo'], cv2.IMREAD_COLOR)			#  Read photo
	photo = cv2.cvtColor(photo, cv2.COLOR_BGR2RGB)					#  Convert photo image to RGB

	params['Kmat'] = loadKFromFile(params['Kmat'], photo)			#  Build and store intrinsic matrix from given file,
																	#  but override these if necessary from the image itself.
																	#  The image is likely to have been scaled.
	if params['SIFT']:												#  User asked for SIFT, find SIFT features
		if params['verbose']:
			print('>>> Generating SIFT descriptors for "' + params['photo'] + '"')
		sift = cv2.xfeatures2d.SIFT_create(nOctaveLayers=7)
		siftKeyPoints, siftDesc = sift.detectAndCompute(photo, None)#  Find keypoints and descriptors with SIFT
	if params['ORB']:												#  User asked for ORB, find ORB features
		if params['verbose']:
			print('>>> Generating ORB descriptors for "' + params['photo'] + '"')
		orb = cv2.ORB_create()
		orbKeyPoints, orbDesc = orb.detectAndCompute(photo, None)	#  Find keypoints and descriptors with ORB
	if params['BRISK']:												#  User asked for BRISK, find BRISK features
		if params['verbose']:
			print('>>> Generating BRISK descriptors for "' + params['photo'] + '"')
		brisk = cv2.BRISK_create()									#  Find keypoints and descriptors with BRISK
		briskKeyPoints, briskDesc = brisk.detectAndCompute(photo, None)

	#################################################################  Step 3: build a list of all texmaps'
																	#          key-points and descriptors
	features = []													#  Will be a list of tuples,
																	#  each   ( (x, y), string, (x, y) float )
																	#  That's: [src pt] filenm [map pt] cost
	reducedExcludeList = [re.sub(params['meshpath'] + '/', '', x) for x in params['exclude']]

	for imgfile in os.listdir(params['meshpath']):					#  For every texmap in a valid format that
		imgfilename = imgfile.split('.')							#  has not been marked for exclusion...
		if imgfilename[-1].lower() in imgformats and imgfile not in reducedExcludeList:
			texmap = cv2.imread(params['meshpath'] + '/' + imgfile, cv2.IMREAD_COLOR)
			texmap = cv2.cvtColor(texmap, cv2.COLOR_BGR2RGB)
			if params['SIFT']:										#  Get SIFT features
				if params['verbose']:
					print('>>> Collecting SIFT descriptors for "' + imgfile + '"')
				features += SIFT(photo, siftKeyPoints, siftDesc, imgfile, texmap, params)
			if params['ORB']:										#  Get ORB features
				if params['verbose']:
					print('>>> Collecting ORB descriptors for "' + imgfile + '"')
				features += ORB(photo, orbKeyPoints, orbDesc, imgfile, texmap, params)
			if params['BRISK']:										#  Get BRISK features
				if params['verbose']:
					print('>>> Collecting BRISK descriptors for "' + imgfile + '"')
				features += BRISK(photo, briskKeyPoints, briskDesc, imgfile, texmap, params)

	#  By now, features = [ ((x, y), string, (x, y) float),
	#                       ((x, y), string, (x, y) float),
	#                        ...,
	#                       ((x, y), string, (x, y) float) ]
	if params['showFeatures']:										#  Draw the features in 2D for reference
																	#  Color-code all features
		colors = np.array([ [int(x * 255)] for x in np.linspace(0.0, 1.0, len(features)) ], np.uint8)
		colors = cv2.applyColorMap(colors, cv2.COLORMAP_JET)

		img = photo.copy()											#  Clone the source image:
		img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)					#  make the clone black-and-white, then
		img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)					#  color again so the features pop out.
		texmaps = {}												#  Will be tuples : (filename, NumPy array image)
		i = 0
		for f in features:
			c = [int(x) for x in colors[i][0]]						#  Note: reverse() is necessary to match OpenCV
			c.reverse()												#  function's expectation of BGR, despite img's formatting
			cv2.circle(img, (int(np.round(f[0][0])), int(np.round(f[0][1]))), 3, tuple(c), thickness=2)
			txname = f[1].split('.')[0]
			if txname not in texmaps:
				texmaps[txname] = cv2.imread(params['meshpath'] + '/' + f[1], cv2.IMREAD_COLOR)
				texmaps[txname] = cv2.cvtColor(texmaps[txname], cv2.COLOR_BGR2GRAY)
				texmaps[txname] = cv2.cvtColor(texmaps[txname], cv2.COLOR_GRAY2RGB)
			cv2.circle(texmaps[txname], \
			                (int(np.round(f[2][0])), int(np.round(f[2][1]))), 3, tuple(c), thickness=2)
			i += 1
																	#  Finally write the annotated source image
		cv2.imwrite(params['photo'].split('.')[0] + '.features.jpg', img)
		for k, v in texmaps.items():								#  Write all annotated texture maps
			cv2.imwrite(k + '.features.jpg', v)

	if params['verbose']:
		print('>>> ' + str(len(features)) + ' putative feature matches (2D to 2D)')

	#################################################################  Step 4: Load the OBJ mesh model
	if params['verbose']:
		print('>>> Loading OBJ file...')							#  http://www.andrewnoske.com/wiki/OBJ_file_format
																	#  Build a model we can query
	obj = Mesh(params['mesh'], params['meshpath'], reducedExcludeList, params['epsilon'], params['verbose'])
	obj.reconcile = False											#  In this application we do not care about which
																	#  vertices are "close enough" to call the same.
	obj.texmaporigin = params['txorigin']							#  Tell Mesh object where its texture map origin should be
	obj.load()
	if params['verbose']:
		print('    done')

	if params['showFeatures']:										#  Draw the triangle borders of all texture maps
		texmaps = {}
		for f in features:
			txname = f[1].split('.')[0]
			if txname not in texmaps:
				texmaps[txname] = cv2.imread(params['meshpath'] + '/' + f[1], cv2.IMREAD_COLOR)
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
		for k, v in texmaps.items():								#  Write all annotated texture maps
			cv2.imwrite(k + '.borders.'+params['txorigin']+'.jpg', v)

	#################################################################  Step 5: Locate interest points in 3D
	#  For each interest point: (texmapfilename, (texmap_x, texmap_y))
	#  which 2D triangle is it in? (intermediate goal)
	#  which 3D triangle is it in? (what we care about)
	#  Then end up with this:
	#  (photo_x, photo_y), (computed_x, computed_y, computed_z)
	#  That's a point in the 2D photograph and its corresponding point in the 3D mesh by way of the 2D texmap.
	if params['verbose']:
		print('>>> Locating 2D features in 3D space')

	correspondences2_3 = []											#  To be a list of 2D-->3D correspondences
	if not os.path.exists('correspondences.txt'):					#  No pre-computed list of 2D-->3D exists
		fctr = 1													#  (Counter really only for verbose display)
		for f in features:											#  Each f = ( (x, y), string, (x, y) float )
			indices = obj.query2d(f[2], 0, len(obj.faces))			#  Get list of all faces in mesh, sorted by
			i = 0													#  barycenter-proximity to feature point.
			while i < len(indices) and not in2DTriangle(f[2], obj.faces[indices[i]].t2D, f[1], obj.faces[indices[i]].texmap):
				i += 1												#  Find the 2D triangle containing the detected feature, f[2]

			if i < len(indices):									#  Triangle-test passed, and we broke out of the loop:
				triangle = indices[i]								#  find the 3D equivalent

				if params['verbose']:
					sys.stdout.write('  Feature '+str(fctr)+ ' found in triangle '+str(triangle)+ ' ' * 24 + "\r")
					sys.stdout.flush()

				#  The influences exerted by the 2D triangle's defining points upon the point inside will be
				#  the same as the influences exerted by the 3D triangle's defining points upon the point inside of it!
				#  Let's use some simpler notation:
				x = f[2][0]											#  Point inside the 2D texmap triangle
				y = f[2][1]
				x1 = obj.faces[triangle].t2D[0][0]					#  First point defining the 2D texmap triangle
				y1 = obj.faces[triangle].t2D[0][1]
				x2 = obj.faces[triangle].t2D[1][0]					#  Second point defining the 2D texmap triangle
				y2 = obj.faces[triangle].t2D[1][1]
				x3 = obj.faces[triangle].t2D[2][0]					#  Third point defining the 2D texmap triangle
				y3 = obj.faces[triangle].t2D[2][1]
																	#  Guard against zero-division
				denom = ((y2 - y3) * (x1 - x3) + (x3 - x2) * (y1 - y3))
				if denom == 0.0:
					lambda1 = 0.0
				else:
					lambda1 = ((y2 - y3) * (x - x3) + (x3 - x2) * (y - y3)) / denom
																	#  Guard against zero-division
				denom = ((y2 - y3) * (x1 - x3) + (x3 - x2) * (y1 - y3))
				if denom == 0.0:
					lambda2 = 0.0
				else:
					lambda2 = ((y3 - y1) * (x - x3) + (x1 - x3) * (y - y3)) / denom

				lambda3 = 1.0 - lambda1 - lambda2
																	#  The corresponding 3D point is under comparable influences
																	#  from corners as its 2D counterpart.
				alpha = ( lambda1 * obj.faces[triangle].t3D[0][0], \
				          lambda1 * obj.faces[triangle].t3D[0][1], \
				          lambda1 * obj.faces[triangle].t3D[0][2]  )
				beta  = ( lambda2 * obj.faces[triangle].t3D[1][0], \
				          lambda2 * obj.faces[triangle].t3D[1][1], \
				          lambda2 * obj.faces[triangle].t3D[1][2]  )
				gamma = ( lambda3 * obj.faces[triangle].t3D[2][0], \
				          lambda3 * obj.faces[triangle].t3D[2][1], \
				          lambda3 * obj.faces[triangle].t3D[2][2]  )

				pt3   = ( alpha[0] + beta[0] + gamma[0], \
				          alpha[1] + beta[1] + gamma[1], \
				          alpha[2] + beta[2] + gamma[2]  )

				correspondences2_3.append( (f[0], pt3) )			#  Add to the NEW list of 2D-->3D correspondences

			elif params['verbose']:									#  Feature not found in any triangle
				sys.stdout.write('  Feature '+str(fctr)+ ' not found in any triangle' + ' ' * 24 + "\r")
				sys.stdout.flush()

			fctr += 1												#  Increment feature counter

		if params['verbose']:										#  Loop's over:
			print('')												#  skip a line

		fh = open('correspondences.txt', 'w')						#  Computing these is expensive: save correspondences to file!
		fh.write(params['photo'] + '\n')							#  File header tells us which source image was used
		for corr in correspondences2_3:								#  File format: 2Dx 2Dy 3Dx 3Dy 3Dz
			fh.write(str(corr[0][0]) + '\t' + str(corr[0][1]) + '\t' + str(corr[1][0]) + '\t' + str(corr[1][1]) + '\t' + str(corr[1][2]) + '\n')
		fh.close()

		if params['showFeatures']:									#  Create a point cloud of the features found in 3D

			colors = np.array([ [int(x * 255)] for x in np.linspace(0.0, 1.0, len(correspondences2_3)) ], np.uint8)
			colors = cv2.applyColorMap(colors, cv2.COLORMAP_JET)

			img = photo.copy()										#  Clone the source image:
			img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)				#  make the clone black-and-white, then
			img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)				#  color again so the features pop out.
			i = 0
			for corr in correspondences2_3:
				c = [int(x) for x in colors[i][0]]					#  Note: reverse() is necessary to match OpenCV
				c.reverse()											#  function's expectation of BGR color tuple
				cv2.circle(img, (int(np.round(corr[0][0])), int(np.round(corr[0][1]))), 3, tuple(c), thickness=2)
				i += 1

			cv2.imwrite(params['photo'].split('.')[0] + '.correspondences.jpg', img)

			fstr  = 'ply\n'
			fstr += 'format ascii 1.0\n'
			fstr += 'comment https://github.com/EricCJoyce\n'
			fstr += 'element vertex ' + str(len(correspondences2_3)) + '\n'
			fstr += 'property float x\n'
			fstr += 'property float y\n'
			fstr += 'property float z\n'
			fstr += 'property uchar red\n'
			fstr += 'property uchar green\n'
			fstr += 'property uchar blue\n'
			fstr += 'end_header\n'
			i = 0
			for corr in correspondences2_3:
				fstr += str(corr[1][0]) + ' '
				fstr += str(corr[1][1]) + ' '
				fstr += str(corr[1][2]) + ' '
				fstr += str(colors[i][0][0]) + ' '
				fstr += str(colors[i][0][1]) + ' '
				fstr += str(colors[i][0][2]) + '\n'
				i += 1
			pointcloudfn = params['mesh'].split('/')[-1].split('.')[0]
			fh = open(pointcloudfn + '.correspondences.ply', 'w')
			fh.write(fstr)
			fh.close()
	else:															#  We already computed the 2D-->3D correspondences!
		fh = open('correspondences.txt', 'r')
		lines = fh.readlines()
		fh.close()
		for line in lines[1:]:
			arr = line.strip().split()
			correspondences2_3.append( ((float(arr[0]), float(arr[1])), (float(arr[2]), float(arr[3]), float(arr[4]))) )

	if params['verbose']:
		print('>>> ' + str(len(correspondences2_3)) + ' putative correspondences (2D to 3D)')

	#################################################################  Step 6: Run the PnP-solver in RANSAC
																	#  (Side note: there IS an OpenCV function named
																	#   solvePnPRansac, but it's buggy. Leave it alone!)
	distortionCoeffs = np.zeros((4, 1))								#  Assume no lens distortion.
	s = 4															#  A camera pose is determined by at least 4 points.
	sampleCount = 0													#  Initialize to zero.
	champion = None													#  Will become a tuple, best (rotation, translation) so far.
	championInliers = []											#  Indices of correspondences supporting champion hypothesis.

	if params['iter'] > 0:											#  Run for a specified number of iterations

		while params['iter'] > sampleCount:							#  While we've not yet run enough random samples...
			A = np.random.randint(0, len(correspondences2_3))		#  Randomly select four points
			B = A													#  Make sure they're all unique
			while B == A:
				B = np.random.randint(0, len(correspondences2_3))
			C = B
			while C == A or C == B:
				C = np.random.randint(0, len(correspondences2_3))
			D = C
			while D == A or D == B or D == C:
				D = np.random.randint(0, len(correspondences2_3))

																	#  Data-type counts! Convert these explicitly!
			corr3 = np.array([ [ correspondences2_3[x][1][0], \
			                     correspondences2_3[x][1][1], \
			                     correspondences2_3[x][1][2] ] for x in [A, B, C, D] ], dtype=np.float64)
			corr2 = np.array([ [ correspondences2_3[x][0][0], \
			                     correspondences2_3[x][0][1] ] for x in [A, B, C, D] ], dtype=np.float64)
																	#  Run solver
			success, rotation, translation = cv2.solvePnP(corr3, corr2, \
			                                              params['Kmat'], distortionCoeffs, \
			                                              flags=cv2.SOLVEPNP_ITERATIVE)
																	#  Build P matrix so we can reproject using hypothesis
			P = buildProjectionMatrix(rotation, translation, params['Kmat'])

			inliers = [A, B, C, D]									#  Compute reprojection error for all other correspondences
			for q in range(0, len(correspondences2_3)):
				if q != A and q != B and q != C and q != D:			#  Make sure we exclude points that define the hypothesis!
																	#  Build a vector: the homogeneous point in 3D space
					X = np.array([correspondences2_3[q][1][0], \
					              correspondences2_3[q][1][1], \
					              correspondences2_3[q][1][2], 1.0], dtype=np.float64)
					x = P.dot(X)									#  Run it through the projection matrix defined by our hypothesis
					x[0] /= x[2]									#  Normalize image X and Y coordinates
					x[1] /= x[2]
																	#  How off the mark is this reprojection from the correspondence?
					dist = np.sqrt(pow(x[0] - correspondences2_3[q][0][0], 2) + pow(x[1] - correspondences2_3[q][0][1], 2))

					if dist < params['reprojErr']:					#  Within forgivable bounds?
						inliers.append(q)							#  Then it's an inlier, and we keep it
																	#  How does this hypothesized pose compare against others?
			if champion is None or len(inliers) > len(championInliers):
				champion = (rotation, translation)
				championInliers = inliers[:]

			sampleCount += 1

			if params['verbose']:
				sys.stdout.write('  RANSAC sampling ' + str(sampleCount) + '/' + str(params['iter']) + ' ' * 24 + "\r")
				sys.stdout.flush()
	else:															#  Run adaptive RANSAC
		outlierRatio = params['outlierRatio']						#  Worst-case assumption: this portion of points are outliers
		inlierRatio = 0.0
		prob = params['confidence']									#  How confident do we wish to be
		N = float('inf')											#  Initialize N to infinity

		if params['verbose']:
			print('>>> Initial assumption: '+str(int(round(params['outlierRatio']*100.0)))+'% of points are outliers.')
			print('>>> Seeking a purely inlier sample with '+str(int(round(params['confidence']*100.0)))+'% confidence.')

		while N > float(sampleCount):
			A = np.random.randint(0, len(correspondences2_3))		#  Randomly select four points
			B = A													#  Make sure they're all unique
			while B == A:
				B = np.random.randint(0, len(correspondences2_3))
			C = B
			while C == A or C == B:
				C = np.random.randint(0, len(correspondences2_3))
			D = C
			while D == A or D == B or D == C:
				D = np.random.randint(0, len(correspondences2_3))

																	#  Data-type counts! Convert these explicitly!
			corr3 = np.array([ [ correspondences2_3[x][1][0], \
			                     correspondences2_3[x][1][1], \
			                     correspondences2_3[x][1][2] ] for x in [A, B, C, D] ], dtype=np.float64)
			corr2 = np.array([ [ correspondences2_3[x][0][0], \
			                     correspondences2_3[x][0][1] ] for x in [A, B, C, D] ], dtype=np.float64)
																	#  Run solver
			success, rotation, translation = cv2.solvePnP(corr3, corr2, \
			                                              params['Kmat'], distortionCoeffs, \
			                                              flags=cv2.SOLVEPNP_ITERATIVE)
																	#  Build P matrix so we can reproject using hypothesis
			P = buildProjectionMatrix(rotation, translation, params['Kmat'])

			inliers = [A, B, C, D]									#  Compute reprojection error for all other correspondences
			for q in range(0, len(correspondences2_3)):
				if q != A and q != B and q != C and q != D:			#  Make sure we exclude points that define the hypothesis!
																	#  Build a vector: the homogeneous point in 3D space
					X = np.array([correspondences2_3[q][1][0], \
					              correspondences2_3[q][1][1], \
					              correspondences2_3[q][1][2], 1.0], dtype=np.float64)
					x = P.dot(X)									#  Run it through the projection matrix defined by our hypothesis
					x[0] /= x[2]									#  Normalize image X and Y coordinates
					x[1] /= x[2]
																	#  How off the mark is this reprojection from the correspondence?
					dist = np.sqrt(pow(x[0] - correspondences2_3[q][0][0], 2) + pow(x[1] - correspondences2_3[q][0][1], 2))

					if dist < params['reprojErr']:					#  Within forgivable bounds?
						inliers.append(q)							#  Then it's an inlier, and we keep it
																	#  How does this hypothesized pose compare against others?
			if champion is None or len(inliers) > len(championInliers):
				champion = (rotation, translation)
				championInliers = inliers[:]

			inlierRatio = float(len(championInliers)) / float(len(correspondences2_3))
			outlierRatio = 1.0 - inlierRatio
			N = int(np.log(1.0 - prob) / np.log(1.0 - pow(1.0 - outlierRatio, s)))

			sampleCount += 1

			if params['verbose']:
				sys.stdout.write('  RANSAC sampling ' + str(sampleCount) + '/' + str(N) + ' ' * 24 + "\r")
				sys.stdout.flush()

	if params['verbose']:
		print('')													#  Skip a line
		print('>>> Best camera pose estimate ratified by '+str(len(championInliers))+' inliers.')

	if params['showInliers']:										#  The 'inliers' list returned is a list of
																	#  indices into 'correspondences2_3'
																	#  Build color-map of distinct colors
		colors = np.array([ [int(x * 255)] for x in np.linspace(0.0, 1.0, len(championInliers)) ], np.uint8)
		colors = cv2.applyColorMap(colors, cv2.COLORMAP_JET)
		writeInlierPly(colors, championInliers, correspondences2_3)
		writeInlierImage(colors, championInliers, correspondences2_3, photo)

	#################################################################  Step 7: Pack up and deliver information
	fstr  = ''
	for i in championInliers:
		fstr += str(correspondences2_3[i][0][0])+'\t'				#  First and second columns: 2D feature in source (x, y)
		fstr += str(correspondences2_3[i][0][1])+'\t'
		fstr += str(correspondences2_3[i][1][0])+'\t'				#  Third, fourth, fifth: 3D points of those features
		fstr += str(correspondences2_3[i][1][1])+'\t'
		fstr += str(correspondences2_3[i][1][2])+'\n'
	fh = open('inliers.log', 'w')									#  Always, at the very least, save the inliers to file
	fh.write(fstr)
	fh.close()

	print('r\n'+str(champion[0]))									#  Always print to screen
	print('t\n'+str(champion[1]))

	if len(params['output']) > 0:									#  At least one output format was specified
		fstr = ''													#  Prepare string to write to file
		if 'rt' in params['output']:								#  OpenCV rotation vector and
			if params['verbose']:									#  OpenCV translation vector
				print('Writing OpenCV rotation vector and translation vector to file')
			fstr += 'r: '+str(champion[0][0][0])+' '+str(champion[0][1][0])+' '+str(champion[0][2][0])+'\n'
			fstr += 't: '+str(champion[1][0][0])+' '+str(champion[1][1][0])+' '+str(champion[1][2][0])+'\n\n'
		if 'Rt' in params['output']:								#  Rotation matrix and
			if params['verbose']:									#  camera-center translation vector
				print('Writing rotation matrix and (camera center) translation vector to file')
			RotMat, _ = cv2.Rodrigues(champion[0])
			fstr += 'R:\n'
			fstr += str(RotMat[0][0])+' '+str(RotMat[0][1])+' '+str(RotMat[0][2])+'\n'
			fstr += str(RotMat[1][0])+' '+str(RotMat[1][1])+' '+str(RotMat[1][2])+'\n'
			fstr += str(RotMat[2][0])+' '+str(RotMat[2][1])+' '+str(RotMat[2][2])+'\n'
			t = -RotMat.dot(champion[1])
			fstr += 't: '+str(t[0][0])+' '+str(t[1][0])+' '+str(t[2][0])+'\n\n'
		if 'K' in params['output']:									#  Intrinsic matrix (3x3)
			if params['verbose']:
				print('Writing intrinsic matrix to file')
			fstr += 'K:\n'
			fstr += str(params['Kmat'][0][0])+' '+str(params['Kmat'][0][1])+' '+str(params['Kmat'][0][2])+'\n'
			fstr += str(params['Kmat'][1][0])+' '+str(params['Kmat'][1][1])+' '+str(params['Kmat'][1][2])+'\n'
			fstr += str(params['Kmat'][2][0])+' '+str(params['Kmat'][2][1])+' '+str(params['Kmat'][2][2])+'\n\n'
		if 'Ext' in params['output']:								#  Extrinsic matrix (3x4)
			if params['verbose']:
				print('Writing extrinsic matrix to file')
			RotMat, _ = cv2.Rodrigues(champion[0])
			fstr += 'Ext:\n'
			fstr += str(RotMat[0][0])+' '+str(RotMat[0][1])+' '+str(RotMat[0][2])+' '+str(champion[1][0][0])+'\n'
			fstr += str(RotMat[1][0])+' '+str(RotMat[1][1])+' '+str(RotMat[1][2])+' '+str(champion[1][1][0])+'\n'
			fstr += str(RotMat[2][0])+' '+str(RotMat[2][1])+' '+str(RotMat[2][2])+' '+str(champion[1][2][0])+'\n\n'
		if 'P' in params['output']:									#  Projection matrix = Intrinsic (3x3) * Extrinsic (3x4)
			if params['verbose']:
				print('Writing projection matrix to file')
			RotMat, _ = cv2.Rodrigues(champion[0])
			Ext = np.array( [ [RotMat[0][0], RotMat[0][1], RotMat[0][2], champion[1][0][0]], \
			                  [RotMat[1][0], RotMat[1][1], RotMat[1][2], champion[1][1][0]], \
			                  [RotMat[2][0], RotMat[2][1], RotMat[2][2], champion[1][2][0]] ] )
			P = params['Kmat'].dot(Ext)
			fstr += 'P:\n'
			fstr += str(P[0][0])+' '+str(P[0][1])+' '+str(P[0][2])+' '+str(P[0][3])+'\n'
			fstr += str(P[1][0])+' '+str(P[1][1])+' '+str(P[1][2])+' '+str(P[1][3])+'\n'
			fstr += str(P[2][0])+' '+str(P[2][1])+' '+str(P[2][2])+' '+str(P[2][3])+'\n\n'
		'''
		if 'Meshlab' in params['output']:							#  Meshlab pasty
			if params['verbose']:
				print('Writing Meshlab markup to file')
			RotMat, _ = cv2.Rodrigues(champion[0])
			t = -RotMat.dot(champion[1])
			#  https://sourceforge.net/p/meshlab/discussion/499533/thread/cc40efe0/
			mmPerPixelx = 0.0369161
			mmPerPixely = 0.0369161
			fstr += '<!DOCTYPE ViewState>'
			fstr += '<project>'
																	#  Translation is a column vector:
			fstr += '<VCGCamera '									#  must be indexed as 2D, row, col
			fstr += 'TranslationVector="'+str(t[0][0])+' '+str(t[1][0])+' '+str(t[2][0])+' 1" '
			fstr += 'LensDistortion="0 0" '
			fstr += 'ViewportPx="'+str(params['Kmat'][0][2] * 2.0)+' '+str(params['Kmat'][1][2] * 2.0)+'" '
			fstr += 'PixelSizeMm="'+str(mmPerPixelx)+' '+str(mmPerPixely)+'" '
			fstr += 'CenterPx="'+str(int(round(params['Kmat'][0][2])))+' '+str(int(round(params['Kmat'][1][2])))+'" '
			fstr += 'FocalMm="30.00" '
			fstr += 'RotationMatrix="'+str(RotMat[0][0])+' '+str(RotMat[1][0])+' '+str(RotMat[2][0])+' 0 '
			fstr +=                    str(RotMat[0][1])+' '+str(RotMat[1][1])+' '+str(RotMat[2][1])+' 0 '
			fstr +=                    str(RotMat[0][2])+' '+str(RotMat[1][2])+' '+str(RotMat[2][2])+' 0 '
			fstr +=                   '0 0 0 1 "/>'
			fstr += '<ViewSettings NearPlane="0.909327" TrackScale="1.73205" FarPlane="8.65447"/>'
			fstr += '</project>\n\n'
		if 'Blender' in params['output']:							#  Pastable code for Blender's Python console
			if params['verbose']:
				print('Writing Blender code to file')
			rotMat, _ = cv2.Rodrigues(champion[0])
			t = -RotMat.dot(champion[1])							#  Camera position
																	#  Set rotation in Quaternions
			RotMat = RotMat.T										#  Transpose rotation matrix
			tr = RotMat.trace()										#  Rotation matrix trace
			if tr > 0:
				sq = np.sqrt(1.0 + tr) * 2.0
				qw = 0.25 * sq
				qx = (RotMat[2][1] - RotMat[1][2]) / sq
				qy = (RotMat[0][2] - RotMat[2][0]) / sq
				qz = (RotMat[1][0] - RotMat[0][1]) / sq
			elif RotMat[0][0] > RotMat[1][1] and RotMat[0][0] > RotMat[2][2]:
				sq = np.sqrt(1.0 + RotMat[0][0] - RotMat[1][1] - RotMat[2][2]) * 2.0
				qw = (RotMat[2][1] - RotMat[1][2]) / sq
				qx = 0.25 * sq
				qy = (RotMat[0][1] + RotMat[1][0]) / sq
				qz = (RotMat[0][2] + RotMat[2][0]) / sq
			elif RotMat[1][1] > RotMat[2][2]:
				sq = np.sqrt(1.0 + RotMat[1][1] - RotMat[0][0] - RotMat[2][2]) * 2.0
				qw = (RotMat[0][2] - RotMat[2][0]) / sq
				qx = (RotMat[0][1] + RotMat[1][0]) / sq
				qy = 0.25 * sq
				qz = (RotMat[1][2] + RotMat[2][1]) / sq
			else:
				sq = np.sqrt(1.0 + RotMat[2][2] - RotMat[0][0] - RotMat[1][1]) * 2.0
				qw = (RotMat[1][0] - RotMat[0][1]) / sq
				qx = (RotMat[0][2] + RotMat[2][0]) / sq
				qy = (RotMat[1][2] + RotMat[2][1]) / sq
				qz = 0.25 * sq
  			fstr += 'import bpy\n'
			fstr += 'pi = 3.14159265\n'
			fstr += 'scene = bpy.data.scenes["Scene"]\n'
			fstr += 'scene.camera.location.x = '+str(t[0][0])+'\n'
			fstr += 'scene.camera.location.y = '+str(t[1][0])+'\n'
			fstr += 'scene.camera.location.z = '+str(t[2][0])+'\n'
			fstr += 'scene.camera.rotation_mode = "QUATERNION"\n'	#  Blender quaternions read W, X, Y, Z
																	#  Quirks of the Blender coordinate system:
																	#  Swap W and Y; negate Y.
			fstr += 'scene.camera.rotation_quaternion[0] = '+str(-qy)+'\n'
			fstr += 'scene.camera.rotation_quaternion[1] = '+str(qz)+'\n'
																	#  Swap X and Z; negate X.
			fstr += 'scene.camera.rotation_quaternion[2] = '+str(qw)+'\n'
			fstr += 'scene.camera.rotation_quaternion[3] = '+str(-qx)+'\n'
			fstr += '\n'
		'''

		fh = open('find.log', 'w')									#  Write to file
		fh.write(fstr)
		fh.close()

		if params['verbose']:
			print('Writing camera pyramid PLY file')
		plystr  = 'ply\n'
		plystr += 'format ascii 1.0\n'
		plystr += 'comment https://github.com/EricCJoyce\n'
		plystr += 'element vertex 5\n'
		plystr += 'property float x\n'
		plystr += 'property float y\n'
		plystr += 'property float z\n'
		plystr += 'element edge 8\n'
		plystr += 'property int vertex1\n'
		plystr += 'property int vertex2\n'
		plystr += 'property uchar red\n'
		plystr += 'property uchar green\n'
		plystr += 'property uchar blue\n'
		plystr += 'end_header\n'
		campoints = []
		campoints.append( np.array([0.0, 0.0, 0.0]) )				#  Push camera center
																	#  Push upper-left image plane corner
		campoints.append( np.array([ -params['Kmat'][0][2],  params['Kmat'][1][2], params['Kmat'][0][0] ]) )
																	#  Push upper-right image plane corner
		campoints.append( np.array([  params['Kmat'][0][2],  params['Kmat'][1][2], params['Kmat'][0][0] ]) )
																	#  Push lower-right image plane corner
		campoints.append( np.array([  params['Kmat'][0][2], -params['Kmat'][1][2], params['Kmat'][0][0] ]) )
																	#  Push lower-left image plane corner
		campoints.append( np.array([ -params['Kmat'][0][2], -params['Kmat'][1][2], params['Kmat'][0][0] ]) )

		R, _ = cv2.Rodrigues(champion[0])
		t = -R.T.dot(champion[1])
		R = R.T

		for point in campoints:
			p = point / params['Kmat'][0][0]
			p = p * 0.1
			T = np.array([[R[0][0], R[0][1], R[0][2], t[0][0]], \
			              [R[1][0], R[1][1], R[1][2], t[1][0]], \
			              [R[2][0], R[2][1], R[2][2], t[2][0]], \
			              [0.0,     0.0,     0.0,     1.0]])
			p = T.dot( np.array([p[0], p[1], p[2], 1.0]))
			plystr += str(p[0]) + ' ' + str(p[1]) + ' ' + str(p[2]) + '\n'

		plystr += '0 1 0 255 0\n'
		plystr += '0 2 0 255 0\n'
		plystr += '0 3 0 0 0\n'
		plystr += '0 4 0 0 0\n'
		plystr += '1 2 0 255 0\n'
		plystr += '2 3 0 0 0\n'
		plystr += '3 4 0 0 0\n'
		plystr += '4 1 0 0 0\n'

		fh = open('camerapose.ply', 'w')
		fh.write(plystr)
		fh.close()

	return

#  Build a projection matrix from the given rotation 3-vector 'r',
#  the given translation 3-vector 't', and the given intrinsic matrix 'K'
def buildProjectionMatrix(r, t, K):
	R, _ = cv2.Rodrigues(r)
	Ext = np.array([ [ R[0][0], R[0][1], R[0][2], t[0][0] ], \
	                 [ R[1][0], R[1][1], R[1][2], t[1][0] ], \
	                 [ R[2][0], R[2][1], R[2][2], t[2][0] ] ], dtype=np.float64)
	return K.dot(Ext)

#  Create a copy of the 2D source and mark where inliers using the same color palette
#  used for the 3D vertices.
def writeInlierImage(colors, inliers, correspondences2_3, photo):
	img = np.copy(photo)
	img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)						#  Reverse default color arrangement
	img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)						#  Dump color data by forcing grayscale
	img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)						#  Image is now gray, but points will be drawn in color
	for i in range(0, len(inliers)):
		c = [int(x) for x in colors[i][0]]							#  Note: reverse the individual color-list to match
		c.reverse()													#  OpenCV function's BGR-order
		cv2.circle(img, (int(np.round(correspondences2_3[ inliers[i] ][0][0])), \
		                 int(np.round(correspondences2_3[ inliers[i] ][0][1]))), 3, \
		                tuple(c), thickness=2)
	cv2.imwrite('inliers.jpg', img)
	return

#  Create a PLY file with colored points at each inlier vertex.
#  correspondences2_3 is a list of tuples of two tuples, each: ((image-x, image-y), (space-x, space-y, space-z))
def writeInlierPly(colors, inliers, correspondences2_3):
	fstr  = 'ply\n'
	fstr += 'format ascii 1.0\n'
	fstr += 'comment https://github.com/EricCJoyce\n'
	fstr += 'element vertex ' + str(len(inliers)) + '\n'
	fstr += 'property float x\n'
	fstr += 'property float y\n'
	fstr += 'property float z\n'
	fstr += 'property uchar red\n'
	fstr += 'property uchar green\n'
	fstr += 'property uchar blue\n'
	fstr += 'end_header\n'
	for i in range(0, len(inliers)):
		fstr += str(correspondences2_3[ inliers[i] ][1][0]) + ' '	#  Write the X component
		fstr += str(correspondences2_3[ inliers[i] ][1][1]) + ' '	#  Write the Y component
		fstr += str(correspondences2_3[ inliers[i] ][1][2]) + ' '	#  Write the Z component
		fstr += str(colors[i][0][0]) + ' '							#  Write the R component
		fstr += str(colors[i][0][1]) + ' '							#  Write the G component
		fstr += str(colors[i][0][2]) + '\n'							#  Write the B component--DONE!
	fh = open('inliers.ply', 'w')
	fh.write(fstr)
	fh.close()
	return

#  'p' is a tuple: (x, y) in texmap image coordinates ([0, Width], [0, Height])
#  't' is a list of the 3 points (x, y) in ([0, Width], [0, Height]) making up the triangle
#  'featuretexmap' is the name of the texture-map containing the feature at 'p'
#  'facetexmap' is the name of the texture-map containing the triangle 't'
def in2DTriangle(p, t, featuretexmap, facetexmap):
	if featuretexmap == facetexmap:									#  Do the obvious test first: if they're not even
																	#  in the same texture map, how could they be the same?
		v0 = np.array(t[2]) - np.array(t[0])						#  v0 = C - A
		v1 = np.array(t[1]) - np.array(t[0])						#  v1 = B - A
		v2 = np.array(p) - np.array(t[0])							#  v2 = P - A

		dot00 = v0.dot(v0)
		dot01 = v0.dot(v1)
		dot02 = v0.dot(v2)
		dot11 = v1.dot(v1)
		dot12 = v1.dot(v2)

		if (dot00 * dot11 - dot01 * dot01) > 0.0:					#  Preclude zero-division
			invDenom = 1.0 / (dot00 * dot11 - dot01 * dot01)
			u = (dot11 * dot02 - dot01 * dot12) * invDenom
			v = (dot00 * dot12 - dot01 * dot02) * invDenom

			return u >= 0.0 and v >= 0.0 and (u + v) < 1.0

	return False

#  Run SIFT feature-detection, find matches, return list of tuples.
#  Each tuple has the form:  ((photo_x, photo_y), texmapfilename, (texmap_x, texmap_y), cost)
#  Input 'photo' is an OpenCV Mat containing the query image
#  Input 'photoKP' is a list of key-point coordinates in the image
#  Input 'photoDesc' is a corresponding list of descriptors for each key-point
#  Input 'texmapfilename' is a string, the name of the texture-map file
#  Input 'texmap' is an OpenCV Mat containing the texture map image
#  Input 'params' is our script's parameters, collected into a dictionary
def SIFT(photo, photoKP, photoDesc, texmapfilename, texmap, params):
	correspondences = []

	ready = False													#  First find out whether we need to
	if os.path.exists('SIFT.txt'):									#  compute SIFT at all
		fh = open('SIFT.txt', 'r')
		lines = fh.readlines()
		fh.close()
		for line in lines:
			arr = line.strip().split('\t')
			if arr[0] == params['photo'] and arr[1] == texmapfilename:
				kPhoto = (float(arr[2]), float(arr[3]))
				kMap = (float(arr[4]), float(arr[5]))
				d = float(arr[6])
				correspondences.append( (kPhoto, texmapfilename, kMap, d) )
				ready = True										#  This counts as true as long as we find
																	#  at least one correspondence
	if not ready:													#  Compute SIFT correspondences
		fh = open('SIFT.txt', 'a+')									#  Prepare never to compute this correspondence again
		sift = cv2.xfeatures2d.SIFT_create(nOctaveLayers=7)			#  Initialize SIFT detector

		mask = texmap.copy()										#  Clone the texture map
		for y in range(0, len(mask)):								#  Iterate over all its pixels
			for x in range(0, len(mask[y])):
				masked = False										#  Unmasked until proven guilty
				for color in params['maskedcolors']:				#  Try all mask colors
					if mask[y][x][0] == color[0] and mask[y][x][1] == color[1] and mask[y][x][2] == color[2]:
						masked = True
				if masked:
					mask[y][x] = [0, 0, 0]
				else:
					mask[y][x] = [255, 255, 255]
																	#  Dilate and then erode
		mask = cv2.dilate(mask, np.ones((3, 3), dtype=np.uint8), iterations=1)
		mask = cv2.erode(mask, np.ones((3, 3), dtype=np.uint8), iterations=params['erosion'])
		if params['showMask']:
			plt.imshow(mask)
			plt.show()

		kp, desc = sift.detectAndCompute(texmap, None)				#  Find key points and descriptors in texmap
		unmaskedKp = []												#  List of unmasked key points
		unmaskedDesc = []											#  List of unmasked key point descriptors
		for i in range(0, len(kp)):									#  For every key point, does it fall on the mask?
			if mask[ int(kp[i].pt[1]) ][ int(kp[i].pt[0]) ][0] == 255:
				unmaskedKp.append(kp[i])
				unmaskedDesc.append(list(desc[i]))

		unmaskedDesc = np.array(unmaskedDesc)						#  Convert to NumPy array
		bf = cv2.BFMatcher(cv2.NORM_L1, crossCheck=False)			#  Create BFMatcher object
		matches = []
																	#  Mastch descriptors
		for knn in bf.knnMatch(photoDesc, unmaskedDesc, params['knn']):
			matches += knn

		matches = sorted(matches, key=lambda x:x.distance)			#  Sort them by their distance
																	#  (Most probable matches first)
		if params['showSIFT']:										#  Draw first n matches.
			img = cv2.drawMatches(photo, photoKP, texmap, unmaskedKp, matches[:params['showSIFTn']], None, flags=2)

			plt.imshow(img)
			plt.show()

		for m in matches:
			kPhoto = (photoKP[m.queryIdx].pt[0], photoKP[m.queryIdx].pt[1])
			kMap = (unmaskedKp[m.trainIdx].pt[0], unmaskedKp[m.trainIdx].pt[1])
			d = m.distance
			correspondences.append( (kPhoto, texmapfilename, kMap, d) )
																	#  Add file content to accumulator string
			fh.write(params['photo'] + '\t' + texmapfilename + '\t' + \
			         str(kPhoto[0]) + '\t' + str(kPhoto[1]) + '\t' + \
			         str(kMap[0]) + '\t' + str(kMap[1]) + '\t' + str(d) + '\n')
		fh.close()

	return correspondences

#  Run ORB-detection, find matches, return list of tuples.
#  Each in list is of the form: ((photo_x, photo_y), texmapfilename, (texmap_x, texmap_y), cost)
#  Input 'photo' is an OpenCV Mat containing the query image
#  Input 'photoKP' is a list of key-point coordinates in the image
#  Input 'photoDesc' is a corresponding list of descriptors for each key-point
#  Input 'texmapfilename' is a string, the name of the texture-map file
#  Input 'texmap' is an OpenCV Mat containing the texture map image
#  Input 'params' is our script's parameters, collected into a dictionary
def ORB(photo, photoKP, photoDesc, texmapfilename, texmap, params):
	correspondences = []

	ready = False													#  First find out whether we need to
	if os.path.exists('ORB.txt'):									#  compute ORB at all
		fh = open('ORB.txt', 'r')
		lines = fh.readlines()
		fh.close()
		for line in lines:
			arr = line.strip().split('\t')
			if arr[0] == params['photo'] and arr[1] == texmapfilename:
				kPhoto = (float(arr[2]), float(arr[3]))
				kMap = (float(arr[4]), float(arr[5]))
				d = float(arr[6])
				correspondences.append( (kPhoto, texmapfilename, kMap, d) )
				ready = True										#  This counts as true as long as we find
																	#  at least one correspondence
	if not ready:													#  Compute ORB correspondences
		fh = open('ORB.txt', 'a+')									#  Prepare never to compute this correspondence again
		orb = cv2.ORB_create()										#  Initialize ORB detector

		mask = texmap.copy()										#  Clone the texture map
		for y in range(0, len(mask)):								#  Iterate over all its pixels
			for x in range(0, len(mask[y])):
				masked = False										#  Unmasked until proven guilty
				for color in params['maskedcolors']:				#  Try all mask colors
					if mask[y][x][0] == color[0] and mask[y][x][1] == color[1] and mask[y][x][2] == color[2]:
						masked = True
				if masked:
					mask[y][x] = [0, 0, 0]
				else:
					mask[y][x] = [255, 255, 255]
																	#  Dilate and then erode
		mask = cv2.dilate(mask, np.ones((3, 3), dtype=np.uint8), iterations=1)
		mask = cv2.erode(mask, np.ones((3, 3), dtype=np.uint8), iterations=params['erosion'])
		if params['showMask']:
			plt.imshow(mask)
			plt.show()

		kp, desc = orb.detectAndCompute(texmap, None)				#  Find key points and descriptors in texmap
		unmaskedKp = []												#  List of unmasked key points
		unmaskedDesc = []											#  List of unmasked key point descriptors
		for i in range(0, len(kp)):									#  For every key point, does it fall on the mask?
			if mask[ int(kp[i].pt[1]) ][ int(kp[i].pt[0]) ][0] == 255:
				unmaskedKp.append(kp[i])
				unmaskedDesc.append(list(desc[i]))

		unmaskedDesc = np.array(unmaskedDesc)						#  Convert to Numpy array
		bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)		#  Create BFMatcher object
																	#  They say you should use NORM_HAMMING for BRISK and ORB
		matches = []
																	#  Match descriptors
		for knn in bf.knnMatch(photoDesc, unmaskedDesc, params['knn']):
			matches += knn

		matches = sorted(matches, key=lambda x:x.distance)			#  Sort them by their distance
																	#  (Most probable matches first)
		if params['showORB']:										#  Draw first n matches.
			img = cv2.drawMatches(photo, photoKP, texmap, unmaskedKp, matches[:params['showORBn']], None, flags=2)

			plt.imshow(img)
			plt.show()

		for m in matches:
			kPhoto = (photoKP[m.queryIdx].pt[0], photoKP[m.queryIdx].pt[1])
			kMap = (unmaskedKp[m.trainIdx].pt[0], unmaskedKp[m.trainIdx].pt[1])
			d = m.distance
			correspondences.append( (kPhoto, texmapfilename, kMap, d) )
																	#  Add file content to accumulator string
			fh.write(params['photo'] + '\t' + texmapfilename + '\t' + \
			         str(kPhoto[0]) + '\t' + str(kPhoto[1]) + '\t' + \
			         str(kMap[0]) + '\t' + str(kMap[1]) + '\t' + str(d) + '\n')
		fh.close()

	return correspondences

#  Run BRISK-detection, find matches, return list of tuples.
#  Each in list is of the form: ((photo_x, photo_y), texmapfilename, (texmap_x, texmap_y), cost)
#  Input 'photo' is an OpenCV Mat containing the query image
#  Input 'photoKP' is a list of key-point coordinates in the image
#  Input 'photoDesc' is a corresponding list of descriptors for each key-point
#  Input 'texmapfilename' is a string, the name of the texture-map file
#  Input 'texmap' is an OpenCV Mat containing the texture map image
#  Input 'params' is our script's parameters, collected into a dictionary
def BRISK(photo, photoKP, photoDesc, texmapfilename, texmap, params):
	correspondences = []

	ready = False													#  First find out whether we need to
	if os.path.exists('BRISK.txt'):									#  compute BRISK at all
		fh = open('BRISK.txt', 'r')
		lines = fh.readlines()
		fh.close()
		for line in lines:
			arr = line.strip().split('\t')
			if arr[0] == params['photo'] and arr[1] == texmapfilename:
				kPhoto = (float(arr[2]), float(arr[3]))
				kMap = (float(arr[4]), float(arr[5]))
				d = float(arr[6])
				correspondences.append( (kPhoto, texmapfilename, kMap, d) )
				ready = True										#  This counts as true as long as we find
																	#  at least one correspondence
	if not ready:													#  Compute BRISK correspondences
		fh = open('BRISK.txt', 'a+')								#  Prepare never to compute this correspondence again

		brisk = cv2.BRISK_create()									#  Initialize BRISK detector

		mask = texmap.copy()										#  Clone the texture map
		for y in range(0, len(mask)):								#  Iterate over all its pixels
			for x in range(0, len(mask[y])):
				masked = False										#  Unmasked until proven guilty
				for color in params['maskedcolors']:				#  Try all mask colors
					if mask[y][x][0] == color[0] and mask[y][x][1] == color[1] and mask[y][x][2] == color[2]:
						masked = True
				if masked:
					mask[y][x] = [0, 0, 0]
				else:
					mask[y][x] = [255, 255, 255]
																	#  Dilate and then erode
		mask = cv2.dilate(mask, np.ones((3, 3), dtype=np.uint8), iterations=1)
		mask = cv2.erode(mask, np.ones((3, 3), dtype=np.uint8), iterations=params['erosion'])
		if params['showMask']:
			plt.imshow(mask)
			plt.show()

		kp, desc = brisk.detectAndCompute(texmap, None)				#  Find key points and descriptors in texmap
		unmaskedKp = []												#  List of unmasked key points
		unmaskedDesc = []											#  List of unmasked key point descriptors
		for i in range(0, len(kp)):									#  For every key point, does it fall on the mask?
			if mask[ int(kp[i].pt[1]) ][ int(kp[i].pt[0]) ][0] == 255:
				unmaskedKp.append(kp[i])
				unmaskedDesc.append(list(desc[i]))

		unmaskedDesc = np.array(unmaskedDesc)						#  Convert to Numpy array
		bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)		#  Create BFMatcher object
																	#  They say you should use NORM_HAMMING for BRISK and ORB
		matches = []
																	#  Match descriptors
		for knn in bf.knnMatch(photoDesc, unmaskedDesc, params['knn']):
			matches += knn

		matches = sorted(matches, key=lambda x:x.distance)			#  Sort them by their distance
																	#  (Most probable matches first)
		if params['showBRISK']:										#  Draw first n matches.
			img = cv2.drawMatches(photo, photoKP, texmap, unmaskedKp, matches[:params['showBRISKn']], None, flags=2)

			plt.imshow(img)
			plt.show()

		correspondences = []
		for m in matches:
			kPhoto = (photoKP[m.queryIdx].pt[0], photoKP[m.queryIdx].pt[1])
			kMap = (unmaskedKp[m.trainIdx].pt[0], unmaskedKp[m.trainIdx].pt[1])
			d = m.distance
			correspondences.append( (kPhoto, texmapfilename, kMap, d) )
																	#  Add file content to accumulator string
			fh.write(params['photo'] + '\t' + texmapfilename + '\t' + \
			         str(kPhoto[0]) + '\t' + str(kPhoto[1]) + '\t' + \
			         str(kMap[0]) + '\t' + str(kMap[1]) + '\t' + str(d) + '\n')

	return correspondences

#  Read the given intrinsic matrix file and construct a K matrix accordingly.
#  The file we expect should have the following format:
#  Comments begin with a pound character, #
#  fx, separated by whitespace, followed by a real number, sets K[0][0]
#  fy, separated by whitespace, followed by a real number, sets K[1][1]
#  cx, separated by whitespace, followed by a real number, sets K[0][2]
#  cy, separated by whitespace, followed by a real number, sets K[1][2]
#  These can appear in any order, but values not set in this way will leave the initial values
#  in place: [[0, 0, 0],
#             [0, 0, 0],
#             [0, 0, 1]]
#  Also notice that the image has the final say. We allow that the target image may have been scaled
#  for memory constraints, so we take our lead from the calibration file, but amend that according
#  to the image we must work with.
def loadKFromFile(Kfilename, photo):
	fh = open(Kfilename, 'r')
	lines = fh.readlines()
	fh.close()
	w = None														#  These may not have been included
	h = None
	height, width, channels = photo.shape
	K = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
	for line in lines:
		arr = line.strip().split()
		if arr[0] != '#':											#  Ignore comments
			if arr[0] == 'fx':										#  Set fx
				K[0][0] = float(arr[1])
			elif arr[0] == 'fy':									#  Set fy
				K[1][1] = float(arr[1])
			elif arr[0] == 'cx':									#  Set cx
				K[0][2] = float(arr[1])
			elif arr[0] == 'cy':									#  Set cy
				K[1][2] = float(arr[1])
			elif arr[0] == 'w':										#  Save w
				w = float(arr[1])
			elif arr[0] == 'h':										#  Save h
				h = float(arr[1])

	a = width / w													#  Derive the scaling factor (arbitrarily) from width

	K[0][0] *= a													#  Scale fx
	K[1][1] *= a													#  Scale fy
	K[0][2] *= a													#  Scale cx
	K[1][2] *= a													#  Scale cy

	return K

#  Parse the command line and set variables accordingly
def parseRunParameters():
	Kmat = 'K.mat'													#  Default camera intrinsic matrix file
	knn = 3															#  Number of nearest neighbors to collect
	excludeList = []												#  List of files to exclude from consideration
	useSIFT = True													#  Whether to use SIFT detection
	useORB = True													#  Whether to use ORB detection
	useBRISK = True													#  Whether to use BRISK detection
	maskcolors = []													#  List of RGB tuples: colors to treat as "nothing"
	erosion = 5
	showSIFT = False												#  Whether to show the SIFT matches
	showORB = False													#  Whether to show the ORB matches
	showBRISK = False												#  Whether to show the BRISK matches
	showMask = False
	showSIFTnum = 40												#  How many SIFT matches to show
	showORBnum = 40													#  How many ORB matches to show
	showBRISKnum = 40												#  How many BRISK matches to show
	RansacIter = 0													#  Iterations to run RANSAC solver (zero means adapt)
	outlierRatio = 0.9												#  Worst-case assumption: this portion of points are outliers
	confidence = 0.95												#  How confident do we wish to be in adaptive RANSAC

	showFeatures = False											#  Whether to create 2D and 3D reference files for features
	showInliers = False												#  Write a PLY file containing the 3D points solvePnPRansac
																	#  considers to be inliers, and also render a new image from
																	#  the corresponding 2D points.
	reprojErr = 5.0													#  RANSAC reprojection error
	verbose = False
	output = []														#  All formats for output
	txorigin = 'ul'													#  Default assumption: texture map origin is upper-left
	epsilon = 0.0													#  Distance within which vertices are considered identical
	helpme = False

	argtarget = None												#  Current argument to be set
																	#  Permissible setting flags
	flags = ['-K', '-x', '-v', '-knn', \
	         '-SIFT', '-ORB', '-BRISK', '-m', '-e', '-o', '-reprojErr', '-iter', '-outlier', '-conf', \
	         '-showSIFT', '-showORB', '-showBRISK', '-showMask', '-showInliers', '-showFeatures', \
	         '-showSIFTn', '-showORBn', '-showBRISKn', '-epsilon', \
	         '-ll', '-ul', '-lr', '-ur', \
	         '-?', '-help', '--help']
	for i in range(3, len(sys.argv)):
		if sys.argv[i] in flags:
			argtarget = sys.argv[i]
			if argtarget == '-v':
				verbose = True
			elif argtarget == '-showSIFT':
				showSIFT = True
			elif argtarget == '-showORB':
				showORB = True
			elif argtarget == '-showBRISK':
				showBRISK = True
			elif argtarget == '-showMask':
				showMask = True
			elif argtarget == '-showInliers':
				showInliers = True
			elif argtarget == '-showFeatures':
				showFeatures = True
			elif argtarget == '-ll':
				txorigin = 'll'
			elif argtarget == '-ul':
				txorigin = 'ul'
			elif argtarget == '-ur':
				txorigin = 'ur'
			elif argtarget == '-lr':
				txorigin = 'lr'
			elif argtarget == '-?' or argtarget == '-help' or argtarget == '--help':
				helpme = True
		else:
			argval = sys.argv[i]

			if argtarget is not None:
				if argtarget == '-K':								#  Following argument sets the intrinsicmatrix file
					Kmat = argval
				elif argtarget == '-knn':							#  Following argument sets number of nearest
					knn = int(argval)								#  neighbors per interest point to collect
				elif argtarget == '-x':								#  Following argument is a file to exclude
					excludeList.append(argval)						#  (may or may not have file path)
				elif argtarget == '-SIFT':							#  Following argument toggles SIFT
					if argval[0].upper() == 'Y':
						useSIFT = True
					else:
						useSIFT = False
				elif argtarget == '-ORB':							#  Following argument toggles ORB
					if argval[0].upper() == 'Y':
						useORB = True
					else:
						useORB = False
				elif argtarget == '-BRISK':							#  Following argument toggles BRISK
					if argval[0].upper() == 'Y':
						useBRISK = True
					else:
						useBRISK = False
				elif argtarget == '-o':								#  Following string is an output format
					if argval not in output:
						output.append(argval)
				elif argtarget == '-m':								#  Following THREE arguments are R, G, B of a mask color
					if len(maskcolors) == 0 or len(maskcolors[-1]) == 3:
						maskcolors.append( [] )
					clr = int(argval)
					if clr > 255:
						clr = 255
					if clr < 0:
						clr = 0
					maskcolors[-1].append(clr)
				elif argtarget == '-e':								#  Following argument sets the number of pixels
					erosion = int(argval)							#  to erode in the mask
				elif argtarget == '-reprojErr':						#  Following argument sets the RANSAC reprojection error
					reprojErr = np.fabs(float(argval))
				elif argtarget == '-iter':							#  Following argument sets the number of RANSAC iterations
					RansacIter = abs(int(argval))
				elif argtarget == '-outlier':						#  Following argument sets outlier ration
					outlierRatio = np.fabs(float(argval))			#  (pessimistic assumption)
					if outlierRatio > 1.0:
						outlierRatio = 1.0
				elif argtarget == '-conf':							#  Following argument sets adaptive RANSAC confidence
					confidence = np.fabs(float(argval))
					if confidence > 1.0:
						confidence = 1.0
				elif argtarget == '-showSIFTn':						#  Following argument sets number of matches to show
					showSIFTnum = int(argval)
				elif argtarget == '-showORBn':						#  Following argument sets number of matches to show
					showORBnum = int(argval)
				elif argtarget == '-showBRISKn':					#  Following argument sets number of matches to show
					showBRISKnum = int(argval)
				elif argtarget == '-epsilon':						#  Following argument sets epsilon value
					epsilon = np.fabs(float(argval))
																	#  Use default values where necessary.
	params = {}
	params['Kmat'] = Kmat
	params['knn'] = knn
	params['exclude'] = excludeList
	params['verbose'] = verbose
	params['SIFT'] = useSIFT
	params['ORB'] = useORB
	params['BRISK'] = useBRISK
	params['output'] = output
	params['maskedcolors'] = maskcolors
	params['erosion'] = erosion
	params['confidence'] = confidence
	params['outlierRatio'] = outlierRatio
	params['reprojErr'] = reprojErr
	params['showSIFT'] = showSIFT
	params['showORB'] = showORB
	params['showBRISK'] = showBRISK
	params['showMask'] = showMask
	params['showSIFTn'] = showSIFTnum
	params['showORBn'] = showORBnum
	params['showBRISKn'] = showBRISKnum
	params['showInliers'] = showInliers
	params['showFeatures'] = showFeatures
	params['iter'] = RansacIter
	params['txorigin'] = txorigin
	params['epsilon'] = epsilon
	params['helpme'] = helpme

	return params

#  Explain usage of this script and its options to the user.
def usage():
	print('Usage:  python find.py image-filename mesh-filename <options, each preceded by a flag>')
	print(' e.g.:  python find.py image.jpg mesh.obj -K SONY-DSLR-A580-P30mm.mat -v -iter 100000 -o P')
	print('Flags:  -K            following argument is the file containing camera intrinsic data.')
	print('                      The structure and necessary details in a camera file are outlined in README.md')
	print('        -knn          following argument is the number of matches to find in the texture map')
	print('                      per feature in the target image. The default is 3.')
	print('        -x            exclude the following file from treatment as a texture map.')
	print('        -v            enable verbosity')
	print('        -SIFT         following argument is Y or N, enabling and disabling SIFT detection respectively.')
	print('        -ORB          following argument is Y or N, enabling and disabling ORB detection respectively.')
	print('        -BRISK        following argument is Y or N, enabling and disabling BRISK detection respectively.')
	print('        -o            add an output format. Please see the Readme file for recognized formats.')
	print('        -m            following three int arguments (R, G, B) describe a color to key out.')
	print('        -e            following int argument sets the number of pixels by which to erode the texmap edge.')
	print('                      (Readme explains when this would be appropriate.)')
	print('        -reprojErr    following real argument sets the tolerable reprojection error for RANSAC.')
	print('        -iter         following int argument sets the number of iterations for RANSAC.')
	print('                      Note: when this argument is omitted or explicitly set to zero, then adaptive')
	print('                      RANSAC will be used.')
	print('        -outlier      following real number in [0.0, 1.0] is assumed outlier ratio.')
	print('                      This only applies to adaptive RANSAC.')
	print('        -conf         following real number in [0.0, 1.0] is target inlier confidence.')
	print('                      This only applies to adaptive RANSAC.')
	print('        -showSIFT     will display pop-up windows illustrating SIFT correspondences.')
	print('        -showORB      will display pop-up windows illustrating ORB correspondences.')
	print('        -showBRISK    will display pop-up windows illustrating BRISK correspondences.')
	print('        -showSIFTn    following argument sets the number of SIFT correspondences to show.')
	print('        -showORBn     following argument sets the number of ORB correspondences to show.')
	print('        -showBRISKn   following argument sets the number of BRISK correspondences to show.')
	print('        -showMask     will display pop-up windows illustrating the texturemap mask.')
	print('        -showInliers  will generate an image and a point cloud of RANSAC inliers.')
	print('        -showFeatures will generate an image and a point cloud of detected features.')
	print('        -epsilon      following argument sets distance within which vertices are considered the same.')
	print('        -ul           tells the script to assume texture map origins are upper-left.')
	print('                      (This is the default assumption.)')
	print('        -ll           tells the script to assume texture map origins are lower-left.')
	print('        -lr           tells the script to assume texture map origins are lower-right.')
	print('        -ur           tells the script to assume texture map origins are upper-right.')
	print('        -?')
	print('        -help')
	print('        --help        displays this message.')
	return

if __name__ == '__main__':
	main()
