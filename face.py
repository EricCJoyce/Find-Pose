#  Eric Joyce, Stevens Institute of Technology, 2019

import os
import sys
import numpy as np
import cv2															#  Really only used here to get texture-map dimensions
from sklearn.neighbors import KDTree								#  This data structure is the weapon-of-choice for
																	#  multi-dimensional nearest-neighbor problems
class Mesh:
	def __init__(self, filename, texmappath, exclude=[], epsilon=0.0, verbose=False):
		self.faces = {}												#  Hold Face objects
		self.v = 0													#  Number of LISTED vertices in the whole mesh
																	#  (This number may be inflated by duplicates)
		self.vt = 0													#  Number of LISTED texture vertices in the whole mesh
																	#  (This number may be inflated by duplicates)
		self.barycenters2d = None									#  To become a query-able KDTree
		self.barycenters3d = None									#  To become a query-able KDTree
		self.filename = filename									#  Name of the mesh obj
		self.imgformats = ['png', 'jpg', 'jpeg']					#  Acceptable texture map file formats
		self.texmappath = texmappath								#  Path from script to matterport materials
		self.texmaporigin = 'ul'									#  Indication of which corner is texture map origin:
																	#  {'ul', 'll', 'ur', 'lr'} respectively for
																	#  upper-left, lower-left, upper-right, lower-right
		self.epsilon = epsilon										#  Acceptable discrepancies between points
																	#  to be considered the same
		self.reconcile = True										#  Whether we should bother reconciling triangle soup
																	#  with epsilon-distances
		self.verbose = verbose										#  Loading these and reconciling triangle soup can
																	#  take a while; show signs of life
		self.filesizes = {}											#  Save time by looking these up once
		for imgfile in os.listdir(texmappath):						#  For every texmap...
			imgfilename = imgfile.split('.')						#  if it's an known format and not omitted...
			if imgfilename[-1].lower() in self.imgformats:
				if imgfile not in exclude:
					texmap = cv2.imread(texmappath + '/' + imgfile, cv2.IMREAD_COLOR)
																	#  Save to lookup table of width and height
					self.filesizes[imgfile] = (len(texmap[0]), len(texmap))
					if self.verbose:								#  Show the texture map dimensions
						print('  ' + imgfile + ': ' + str(len(texmap[0])) + ' x ' + str(len(texmap)))
				elif self.verbose:									#  Show that we're omitting a file by request
					print('  Excluding ' + imgfile)
		self.vertexLookup = {}										#  Look up a vertex index to find a list of all faces
																	#  to which it contributes.
		self.sames = {}												#  Look up a vertex index to find a list of vertices
																	#  we consider "equal to it within epsilon."

	#  Read the OBJ file line by line. Accumulate 3D vertex and texmap (2D) vertex information, and build
	#  an instance of the Face class once we have enough information for a face.
	def load(self):
		v = {}														#  Vertices: dictionary of 3-tuples: (float, float, float)
		vctr = 1													#  Vertex index counter

		vt = {}														#  Texture coordinates
		vtctr = 1													#  Texture coordinates' index counter

		fctr = 0													#  Face index counter is free to start with zero
																	#  because it is never referred to by other data types
																	#  in the OBJ format
		currentMaterial = None										#  Track which material is currently applied

		if self.verbose:
			print('\n  Loading mesh from ' + self.filename)

		fh = open(self.filename, 'r')								#  Read entire file
		lines = fh.readlines()
		fh.close()

		if self.verbose:	#########################################  3D VERTICES
			print('  Reading vertices...')
		for line in lines:											#  Make one initial pass during which we only
			arr = line.strip().split()								#  care about the vertices.
			if len(arr) > 0:
				if arr[0] == 'v':
					x = float(arr[1])
					y = float(arr[2])
					z = float(arr[3])
					v[vctr] = (x, y, z)								#  Add the vctr-th vertex to the hash table
					self.vertexLookup[vctr] = []					#  Prepare a running list of every face
					vctr += 1										#  that uses this vertex
		if self.verbose:
			print('    ' + str(vctr - 1) + ' vertices')

		allV = [v[x] for x in range(1, vctr)]						#  Build complete list by vertex index
		redundancyTree = KDTree(allV)								#  Turn it into a tree
																	#  Find all vertices within epsilon of each other:
																	#  we're going to call them "The Same," but only so
																	#  we can use them to find neighbors in triangle soup.
		if self.reconcile:											#  So... IGNORE this step if our application doesn't
			if self.verbose:										#  care about triangle adjacency!
				print('  Reconciling triangle soup with epsilon '+str(self.epsilon)+'...')
			samectr = 0
			for vnum in range(1, vctr):								#  Perform test for every vertex.
				ind = redundancyTree.query_radius(np.array( [ v[vnum] ] ), self.epsilon)
				ind = [x + 1 for x in ind[0] if x + 1 != vnum]
				if len(ind) > 0:
					self.sames[vnum] = ind
					samectr += 1
					if self.verbose:
						sys.stdout.write('    %d epsilon-equivalent vertices found\r' % samectr)
						sys.stdout.flush()

			if self.verbose:
				print('')

		if self.verbose:	#########################################  2D (TEXMAP) VERTEX
			print('  Reading texture map vertices...')
		for line in lines:
			arr = line.strip().split()
			if len(arr) > 0:										#  Make sure line actually had content
				if arr[0] == 'vt':
					u = float(arr[1])
					w = float(arr[2])
					vt[vtctr] = (u, w)								#  Add the vtctr-th vertex to the hash table
					vtctr += 1

		if self.verbose:	#########################################  FACE
			print('  Reading faces...')
		for line in lines:
			arr = line.strip().split()
			if len(arr) > 0:										#  Make sure line actually had content
				if arr[0] == 'f':

					subarr = arr[1].split('/')						#  Split v/vt pair
					a1 = int(subarr[0])								#  Save v index
					a2 = int(subarr[1])								#  Save vt index

					subarr = arr[2].split('/')						#  Split v/vt pair
					b1 = int(subarr[0])								#  Save v index
					b2 = int(subarr[1])								#  Save vt index

					subarr = arr[3].split('/')						#  Split v/vt pair
					c1 = int(subarr[0])								#  Save v index
					c2 = int(subarr[1])								#  Save vt index

					texmapW = self.filesizes[currentMaterial][0]	#  Retrieve actual dimensions of this texmap
					texmapH = self.filesizes[currentMaterial][1]	#  so we can get actual pixel locations

					self.faces[fctr] = Face()						#  New face...
																	#  made of these three 3D vertices...
					self.faces[fctr].set3DTriangle(v[a1], v[b1], v[c1])
																	#  which have these three OBJ indices...
					self.faces[fctr].set3DTriangleIndices(a1, b1, c1)
																	#  skinned with this 2D triangle...
					if self.texmaporigin == 'ul':					#  (Origin in upper-left corner)
						self.faces[fctr].set2DTriangle( (vt[a2][0] * texmapW, vt[a2][1] * texmapH), \
						                                (vt[b2][0] * texmapW, vt[b2][1] * texmapH), \
						                                (vt[c2][0] * texmapW, vt[c2][1] * texmapH) )
					elif self.texmaporigin == 'll':					#  (Origin in lower-left corner)
						self.faces[fctr].set2DTriangle( (vt[a2][0] * texmapW, texmapH - vt[a2][1] * texmapH), \
						                                (vt[b2][0] * texmapW, texmapH - vt[b2][1] * texmapH), \
						                                (vt[c2][0] * texmapW, texmapH - vt[c2][1] * texmapH) )
					elif self.texmaporigin == 'lr':					#  (Origin in lower-right corner)
						self.faces[fctr].set2DTriangle( (texmapW - vt[a2][0] * texmapW, texmapH - vt[a2][1] * texmapH), \
						                                (texmapW - vt[b2][0] * texmapW, texmapH - vt[b2][1] * texmapH), \
						                                (texmapW - vt[c2][0] * texmapW, texmapH - vt[c2][1] * texmapH) )
					else:											#  (Origin in upper-right corner)
						self.faces[fctr].set2DTriangle( (texmapW - vt[a2][0] * texmapW, vt[a2][1] * texmapH), \
						                                (texmapW - vt[b2][0] * texmapW, vt[b2][1] * texmapH), \
						                                (texmapW - vt[c2][0] * texmapW, vt[c2][1] * texmapH) )
					self.faces[fctr].set2DTriangleIndices(a2, b2, c2)
																	#  ...which has these three OBJ indices...
					self.faces[fctr].texmap = currentMaterial		#  ...and which comes from this texture map

					self.vertexLookup[a1].append(fctr)				#  Keep a running list of faces touching this vertex
					self.vertexLookup[b1].append(fctr)				#  Keep a running list of faces touching this vertex
					self.vertexLookup[c1].append(fctr)				#  Keep a running list of faces touching this vertex

					fctr += 1

				elif arr[0] == 'usemtl':							#  Change the currently applied material
					currentMaterial = arr[1]
		if self.verbose:
			print('    ' + str(fctr) + ' faces')

		tree2d = [list(self.faces[x].barycenter2D) for x in range(0, fctr)]
		tree3d = [list(self.faces[x].barycenter3D) for x in range(0, fctr)]

		self.barycenters2d = KDTree(tree2d)
		self.barycenters3d = KDTree(tree3d)

		self.v = vctr - 1											#  Save for reference
		self.vt = vtctr - 1

		return

	def query2d(self, pt, a, b):
		_, ind = self.barycenters2d.query(np.array([list(pt)]), k=b)
		return list(ind[0])[a:b + 1]

	def computeFaceNeighbors(self):
		for i in range(0, len(self.faces)):							#  For each face in the mesh
			n = []													#  prepare a list of all neighbor faces.
			for v in self.faces[i].t3Dindices:						#  Look up each vertex in each face
				n += [x for x in self.vertexLookup[v] if x != i]	#  and add as neighbor-faces all faces formed by this vertex.

			s = []
			for v in self.faces[i].t3Dindices:
				if v in self.sames:
					for same in self.sames[v]:
						s += [x for x in self.vertexLookup[same] if x != i]

			n += s
			self.faces[i].neighbors = list(dict.fromkeys(n))		#  Remove duplicate entries and store in Face class

		return

class Face:
	def __init__(self):
		self.texmap = None											#  Will be a string
		self.t2D = []												#  Will be a list of 3 2-tuples
		self.barycenter2D = None									#  Will be a 2-tuple
		self.t3D = []												#  Will be a list of 3 3-tuples
		self.t3Dindices = None										#  Will be a 3-tuple
		self.t2Dindices = None										#  Will be a 3-tuple
		self.barycenter3D = None									#  Will be a 3-tuple
		self.norm = None											#  Will be a 3-tuple
		self.area = None
		self.neighbors = []											#  Will be list of face indices

	def set3DTriangle(self, a, b, c):
		self.t3D = [a, b, c]										#  [ (x, y, z), (x, y, z), (x, y, z) ]
		self.compute3DBarycenter()
		self.computeNorm()
		self.computeArea()
		return

	def set2DTriangle(self, a, b, c):
		self.t2D = [a, b, c]
		self.compute2DBarycenter()
		return

	def set2DTriangleIndices(self, a, b, c):
		self.t2Dindices = (a, b, c)
		return

	def set3DTriangleIndices(self, a, b, c):
		self.t3Dindices = (a, b, c)
		return

	def compute3DBarycenter(self):
		self.barycenter3D = (np.sum( [x[0] for x in self.t3D] ) / 3.0, \
		                     np.sum( [x[1] for x in self.t3D] ) / 3.0, \
		                     np.sum( [x[2] for x in self.t3D] ) / 3.0)
		return

	def compute2DBarycenter(self):
		self.barycenter2D = (np.sum( [x[0] for x in self.t2D] ) / 3.0, \
		                     np.sum( [x[1] for x in self.t2D] ) / 3.0)
		return

	#  The surface norm is the cross product of two triangle edges.
	#  An edge is defined as the difference between two points.
	def computeNorm(self):
		U = [ self.t3D[1][0] - self.t3D[0][0], self.t3D[1][1] - self.t3D[0][1], self.t3D[1][2] - self.t3D[0][2] ]
		V = [ self.t3D[2][0] - self.t3D[0][0], self.t3D[2][1] - self.t3D[0][1], self.t3D[2][2] - self.t3D[0][2] ]
		self.norm = ( U[1] * V[2] - U[2] * V[1], U[2] * V[0] - U[0] * V[2], U[0] * V[1] - U[1] * V[0] )
		self.norm = self.norm / np.linalg.norm(self.norm)
		return

	#  Compute the area of the triangle
	def computeArea(self):
		AB = np.array( [self.t3D[1][0] - self.t3D[0][0], self.t3D[1][1] - self.t3D[0][1], self.t3D[1][2] - self.t3D[0][2]] )
		AC = np.array( [self.t3D[2][0] - self.t3D[0][0], self.t3D[2][1] - self.t3D[0][1], self.t3D[2][2] - self.t3D[0][2]] )
		cosTheta = AB.dot(AC)
		lenAB = np.linalg.norm(AB)
		lenAC = np.linalg.norm(AC)
		cosTheta /= lenAB * lenAC
		sinTheta = np.sqrt(1.0 - cosTheta**2)
		self.area = sinTheta * lenAB * lenAC * 0.5
		return
