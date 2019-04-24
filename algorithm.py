import cdd
import numpy as np
from scipy.spatial import Voronoi
import csv

from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection, Line3DCollection
# from cvxopt import matrix
# from cvxopt import solvers
# from cvxopt import spdiag
import math

ERR = 1e-6

# point is a tuple of d+1 numbers representing its coordinates
# hyperplane is a tuple of d+2 numbers (b,a_1,...a_d+1), representing the equation b+a_1x_1+...+a_d+1x_d+1=0
# return whether a point is on a hyperplane
def onHyperplane(point, hyperplane):
    value = hyperplane[0]
    value += np.dot(point, hyperplane[1:])
    if abs(value) < ERR:
        return True
    return False

# determines if a point is contained in a halfspace (see halfspace = not contained in halfspace)
# point is a tuple of d+1 numbers representing its coordinates
# halfspace is a tuple of d+2 numbers (b,a_1,...a_d+1), representing the equation b+a_1x_1+...+a_d+1x_d+1>=0
def seeHalfspace(point, halfspace):
    value = halfspace[0]
    value += np.dot(point, halfspace[1:])
    if value <= ERR:
        return True
    return False

# poly is a list of hyperplanes that constitute the boundary of the polytope
# determines whether a point is in the polytope
def inOrOnPoly(point, poly, eq_indices):
    for i in range(len(poly)):
        halfspace = poly[i]
        value = halfspace[0]+np.dot(point, halfspace[1:])
        # for hyperplanes, check if dot = 0
        if i in eq_indices and abs(value) > ERR:
            return False
        # for halfspaces, check if dot >= 0
        elif i not in eq_indices and value < -ERR:
            return False
    return True

def interiorToPoly(point, poly, eq_indices):
    for i in range(len(poly)):
        halfspace = poly[i]
        value = halfspace[0]+np.dot(point, halfspace[1:])
        # for hyperplanes, check if dot = 0
        if i in eq_indices and abs(value) > ERR:
            return False
        # for halfspaces, check if dot > 0
        elif i not in eq_indices and value < ERR:
            return False
    return True

# convert the V-representation of a polytope to the H-representation
# a halfspace is a list of d+2 numbers (b,a_1,...a_d+1), representing the inequality b+a_1x_1+...+a_d+1x_d+1>=0
# input a list of vertices; returns a list of halfspaces and indices of equations among the list
def v2hRep(vertices):
    mat = cdd.Matrix([[1] + list(v) for v in vertices], number_type = 'float')
    mat.rep_type = cdd.RepType.GENERATOR
    poly = cdd.Polyhedron(mat)
    h = poly.get_inequalities()
    halfspaces = []
    for i in range(0, h.row_size):
        halfspaces.append(h.__getitem__(i))
    eq_indices = list(h.lin_set)
    return halfspaces, eq_indices

# convert the H-representation of a polytope to the V-representation
# input a list of halfspaces and a list of indices of equations among the list; returns a list of vertices
def h2vRep(halfspaces, eq_indices):
    mat = cdd.Matrix(halfspaces, number_type = 'float')
    mat.rep_type = cdd.RepType.INEQUALITY
    mat.lin_set = frozenset(eq_indices)
    poly = cdd.Polyhedron(mat)
    v = poly.get_generators()
    vertices = []
    for i in range(0, v.row_size):
        # vertices returned by get_generators start with an extra 1
        # need to remove the first to get actual coordinates of vertices
        vertices.append(v.__getitem__(i)[1:])
    return vertices

def cone_v2h(vertices, rays):
    matrix = [[1] + list(v) for v in vertices] + [[0] + list(r) for r in rays]
    mat = cdd.Matrix(matrix, number_type = 'float')
    mat.rep_type = cdd.RepType.GENERATOR
    poly = cdd.Polyhedron(mat)
    h = poly.get_inequalities()
    halfspaces = []
    for i in range(0, h.row_size):
        halfspaces.append(h.__getitem__(i))
    eq_indices = list(h.lin_set)
    return halfspaces, eq_indices

def cone_h2v(halfspaces, eq_indices):
    mat = cdd.Matrix(halfspaces, number_type = 'float')
    mat.rep_type = cdd.RepType.INEQUALITY
    mat.lin_set = frozenset(eq_indices)
    poly = cdd.Polyhedron(mat)
    v = poly.get_generators()
    vertices = []
    rays = []
    for i in range(0, v.row_size):
        # vertices returned by get_generators start with an extra 1
        # rays returned by get_generators start with an extra 0
        # need to remove the first to get actual coordinates of vertices and rays
        vector = v.__getitem__(i)
        if (vector[0] == 1):
            vertices.append(vector[1:])
        if (vector[0] == 0):
            rays.append(vector[1:])
    return vertices, rays

# Given vertices of a polytope, find its dimension
# Dimension given by the rank of the matrix whose rows are differences of all other vertex coordinates and one vertex
def findDimension(vertices):
    if len(vertices) == 0:
        return -1
    if len(vertices) == 1:
        return 0
    vectors = []
    for v in vertices[1:]:
        vectors.append(np.subtract(v,vertices[0]))
    dimension = np.linalg.matrix_rank(vectors)
    return dimension

# Given V- and H- representation of a polyhedron, compute all of its facets
# A facet is represented by a list containing the index of the hyperplane it lies in and the indices of vertices of the polyhedron on the hyperplane
# Input a list of vertices, a list of halfspaces, a list of indices that specify equations among halfspace inequalities
def computeFacets(vertices, halfspaces, eq_indices):
    facets = []
    for i in range(len(halfspaces)):
        # find all halfspace inequalities excluding hyperplane equations
        if i in eq_indices:
            continue
        facets.append([i])
        for j in range(len(vertices)):
            if onHyperplane(vertices[j], halfspaces[i]):
                facets[i].append(j)
    return facets

# Find all ridges of a polyhedron of certain dimension, each ridge represented by the index of the two facets followed by a set containing index of equations and the list of halfspaces that encloses the ridge
def computeRidges(vertices, facets):
    dim = findDimension(vertices)
    ridges = []
    for i in range(len(facets)):
        fi_vertices = facets[i][1:]
        for j in range(i+1,len(facets)):
            fj_vertices = facets[j][1:]
            # find indices of points lying on both facet fi and fj
            point_indices = [p for p in fi_vertices if p in fj_vertices]
            # store coordinates of vertices in ridgepoints
            # compute dimension of potential ridge by subtracting vertex coordinates and computing rank of vector matrix
            ridgepoints = [vertices[p] for p in point_indices]
            dimension = findDimension(ridgepoints)
            # the dimension of a ridge is 2 less than the dimension of the polytope
            if dimension == dim - 2:
                halfspaces,eq_indices = v2hRep(ridgepoints)
                ridge = [i,j]
                ridge.append(eq_indices)
                ridge += halfspaces
                ridges.append(ridge)
    return ridges

# Check if a ridge is in a facet
def ridgeInFacet(ridge, facet_index):
    if ridge[0] == facet_index or ridge[1] == facet_index:
        return True
    return False

# Given a pointset on a hyperplane, find the voronoi cell in the voronoi diagram that contains the point at a specified index in pointset
# Input a point set, index of a specific point and the hyperplane the point set is on
# Return the list of halfspaces and equation indices that define the Voronoi cell
def findVCell(index, pointSet, hyperplane):
    points = np.array(pointSet)
    normal_vectors = []
    constants = []
    # find halfspaces that bounds the cell by finding pairs of points that give a normal vector pointing inward to the halfspaces
    hrep = []
    hrep.append(hyperplane)
    for j in range(len(pointSet)):
        if index != j:
            # compute the vector pointing from each other point to the given point
            point = np.array(pointSet[index])
            point2 = np.array(pointSet[j])
            vector = list(point - point2)
            midpoint = (point + point2)/2.0
            constant = 0.0 - np.dot(vector, midpoint)
            # each dividing halfspace is determined by normal vector and midpoint
            hrep.append(tuple([constant] + vector))
    # hrep starts with a equation that defines the hyperplane, followed by a list of halfspaces
    return hrep, [0]

# Return if a voronoi cell contains some point interior to a ridge
# a ridge is a list containing the following info:
# The first two elements are indices of facets that give the ridge
# The third element is the index of equations among list of halfspaces
# Starting from the fourth elements is the list of halfspaces that bounds the ridge
# vcell is a list of halfspaces that bounds the voronoic cell; note that the first element is an equation representing the hyperplane vcell lies on
def findPointIn(ridge, vcell_h, vcell_e):
    # find halfspaces including info about equations that define the intersection of ridge and vcell
    halfspaces = []
    ridge_eq_indices = ridge[2]
    ridge_halfspaces = ridge[3:]
    # only one equation in vcell h-rep, index equal to number of ridge halfspaces because vcell is appended after ridge
    vcell_eq_indices = [len(ridge_halfspaces)]
    halfspaces += ridge_halfspaces + vcell_h
    eq_indices = ridge_eq_indices+vcell_eq_indices
    # compute the vertices of the intersection of vcell and ridge
    try:
        vertices, rays = cone_h2v(halfspaces, eq_indices)
        interior = True
        # loop through all facets of ridge and see if there is a facet that contains the intersection of ridge and vcell
        for i in range(len(ridge_halfspaces)):
            # skip equations (hyperplanes)
            if i in ridge_eq_indices:
                continue
            containedInFacet = True
            h = ridge_halfspaces[i]
            # loop through vertices of the intersection of ridge and vcell
            for v in vertices:
                # =0 implies on facet (hyperplane); >0 implies in halfspace but not on facet
                if h[0]+np.dot(v, h[1:]) > ERR:
                    containedInFacet = False
                    break
            # if all vertices of the intersection of ridge and cell are contained in a facet of ridge, then there's no point interior to the ridge
            if containedInFacet:
                interior = False
                break
        return interior
    except RuntimeError:
        # ridge and vcell do not intersect
        return False
    # inserting an extra variable to handle strict inequalities
    # vcell_lp = [ halfspace + [0] for halfspace in vcell]
    # ridge_lp = [ halfspace + [-1] for halfspace in ridge[2:]]
    # inequalities.extend(vcell_lp)
    # inequalities.extend(ridge_lp)
    # mat = cdd.Matrix(inequalities, number_type = 'float')
    # maximize this extra variable in lp
    # mat.obj_type = cdd.LPObjectType.MAX
    # mat.obj_func = [0 for i in range(len(vcell[0])-1)] + [1]
    # lp = cdd.LinProg(mat)
    # lp.solve()
    # lp feasible if max>0
    # if lp.status == cdd.LPStatusType.Optimal and lp.obj_value > 0:
    #     return True
    # else:
    #     return False

def seeThrough(index, pointSet, ridge, findex, facets, halfspaces):
    hyperplane = halfspaces[facets[findex][0]]
    vcell_h, vcell_e = findVCell(index, pointSet, hyperplane)
    # return false if cannot find a point in the voronoic cell interior to the ridge
    if not findPointIn(ridge, vcell_h, vcell_e):
        return False
    # find the other facet that the ridge lies on
    if ridge[0] == findex:
        f2 = ridge[1]
    elif ridge[1] == findex:
        f2 = ridge[0]
    else:
        return False
    facet2 = facets[f2]
    halfspace2 = halfspaces[facet2[0]]
    point = pointSet[index]
    # return false if the point is not on the same side of the ridge as facet1
    # any point p on facet1 satisfies p.f2>=0 where f2 is the coefficients in the inequality of the other facet of the ridge (facet2)
    # the point x is on the other side if x.f2<0
    if halfspace2[0] + np.dot(point, halfspace2[1:]) < -ERR:
        return False
    return True

# Given a point, find the closest point on a polytope and compute distance
# Input coordinates of a point and h-rep of a polytope defined by a list of halfspaces and indices of equations among the list
# def computeDistance(point, halfspaces, eq_indices):
#     I = spdiag([1.0 for i in range(len(point))])
#     q = matrix([ -1.0 * val for val in point])
#     inequalities = []
#     equations = []
#     for h in range(len(halfspaces)):
#         if h in eq_indices:
#             equations.append(halfspaces[h])
#         else:
#             inequalities.append(halfspaces[h])
#     G = matrix([[-1.0 * val for val in ineq[1:]] for ineq in inequalities], (len(inequalities), len(point)))
#     h = matrix([ineq[0] for ineq in inequalities])
#     A = matrix([[-1.0 * val for val in eq[1:]] for eq in equations], (len(equations), len(point)))
#     b = matrix([eq[0] for eq in equations])
#     # solve the quadratic program
#     sol = solvers.qp(I, q, G, h, A, b)
#     # get coordinates of the closest point
#     closest_point = []
#     for val in sol['x']:
#         closest_point.append(float(val))
#     # compute distance from point to closest point
#     d_squared = 0.0
#     for i in range(len(point)):
#         d_squared += (closest_point[i] - point[i])*(closest_point[i] - point[i])
#     distance = math.sqrt(d_squared)
#     return closest_point, distance

# find projection of a given point onto a hyperplane
# point denoted by a list x = (x_1,x_2,...x_d+1); hyperplane denoted by a list (b, a_1,...a_d+1)
# using the known normal vector v = (a_1,...a_d+1), solve the equation b+v.(x-kv)=0
def findProjection(point, hyperplane):
    normal_vector = hyperplane[1:]
    constant = hyperplane[0]
    scale = (constant + np.dot(point, normal_vector))/sum(a*a for a in normal_vector )
    projection = [point[i] - scale * normal_vector[i] for i in range(len(point))]
    return projection

def projectToAffineSubspace(point, hyperplanes):
    projection = np.zeros(len(point))
    # normal vectors and b (constant of hyperplanes) are taken directly from hyperplane equations
    normal_vectors = [h[1:] for h in hyperplanes]
    b = [h[0] for h in hyperplanes]
    # compute coefficients for linear equations to solve for projection vector
    # compute what linear combination of normal vectors gives the projection vector
    coefficient_matrix = [[np.dot(n, m) for m in normal_vectors] for n in normal_vectors]
    constants = [-b[i]-np.dot(normal_vectors[i], point) for i in range(len(normal_vectors))]
    solution = np.linalg.solve(coefficient_matrix, constants)
    projection += np.array(point)
    for i in range(len(solution)):
        projection += solution[i] * np.array(normal_vectors[i])
    return tuple(projection)

def distanceBetween(point1, point2):
    return np.linalg.norm(np.array(point1) - np.array(point2))

# Given a point, find the closest point on a polytope of dimension dim and compute distance
# Find the closest point by iterating through closest point on each facet the point can see
def computeClosest(point, vertices, halfspaces, eq_indices, dim):
    # base case of induction: 0d - vertex
    if dim == 0:
        closest_point = vertices[0]
        min_distance = distanceBetween(closest_point, point)
        return closest_point, min_distance
    # if polytope itself has a lower dimension, first check if the projection of the point onto poly lies in interior and satisfy automatically
    # declare projection outside because we need it later for checking seen facets
    projection = []
    if dim < len(point):
        hyperplanes = [halfspaces[e] for e in eq_indices]
        projection = projectToAffineSubspace(point, hyperplanes)
        if interiorToPoly(projection, halfspaces, eq_indices):
            distance = distanceBetween(point, projection)
            return projection, distance
    # if not, check all facets the point can see and find closest for each facet iteratively
    # filter facets of the polyhedron that the given point can see
    facets = computeFacets(vertices, halfspaces, eq_indices)
    # degenerate case: each facet is a just vertex
    if len(facets[0]) == 2:
        min_distance = float('inf')
        closest_point = []
        for vertex in vertices:
            distance = distanceBetween(vertex, point)
            if distance < min_distance:
                closest_point = vertex
                min_distance = distance
        return closest_point, distance
    # general case: each facet has >=2 vertices
    seenFacets = []
    for f in facets:
        halfspace = halfspaces[f[0]]
        # for lower dimension, check which side projectoin falls
        if dim < len(point):
            if seeHalfspace(projection, halfspace):
                seenFacets.append(f)
        else:
            if seeHalfspace(point, halfspace):
                seenFacets.append(f)
    # inductive step
    min_distance = float('inf')
    closest_point = []
    for sf in seenFacets:
        facet_vertices = [vertices[i] for i in sf[1:]]
        facet_halfspaces, facet_eq = v2hRep(facet_vertices)
        p, d = computeClosest(point, facet_vertices, facet_halfspaces, facet_eq, dim - 1)
        if d < min_distance:
            min_distance = d
            closest_point = p
    return closest_point, min_distance

def intersectPoly(p1_h, p1_eq, p2_h, p2_eq):
    h = p1_h + p2_h
    eq = p1_eq + [eq+len(p1_h) for eq in p2_eq]
    return h, eq

def computeTangentCone(point, halfspaces, eq_indices):
    hrep = []
    eq = []
    for i in range(len(halfspaces)):
        halfspace = halfspaces[i]
        if onHyperplane(point, halfspace):
            hrep.append(halfspace)
            if i in eq_indices:
                eq.append(len(hrep)-1)
    return hrep, eq

def computeUnitJet(jet_frame, value):
    sum = sum([math.sqrt(value ** (2 * i)) for i in range(len(jet_frame))])
    jet = [sum([(value ** i) * jet_frame[i][j]/sum for i in range(len(jet_frame))]) for j in range(len(jet_frame[0]))]
    return jet

def computeJet(jet_frame):
    jet_frame_array = np.array(jet_frame)
    jet = np.sum(jet_frame_array, axis = 0)
    return jet

def computeIteratedTagentCone(point, partial_jet_frame, halfspaces, eq_indices):
    if len(partial_jet_frame) == 0:
        return computeTangentCone(point, halfspaces, eq_indices)
    #jet = computeUnitJet(partial_jet_frame, 5e-2)
    cone_hrep, cone_eq = computeTangentCone(point, halfspaces, eq_indices)
    cone_v, cone_ray = cone_h2v(cone_hrep, cone_eq)
    icone_hrep = []
    icone_eq = []
    for vector in partial_jet_frame:
        # append hyperplane orthogonal to each vector in partial jet frame at the given point
        hyperplane = [-np.dot(vector, point)] + vector
        icone_hrep.append(tuple(hyperplane))
        icone_eq.append(len(icone_hrep) - 1)
    # append every hyperplane that contains every vector in the partial jet_frame
    # in other words, omit the one whose normal vector has positive dot product with some vector
    for i in range(len(cone_hrep)):
        omit = False
        h = cone_hrep[i]
        for vector in partial_jet_frame:
            if np.dot(vector, h[1:]) > ERR:
                omit = True
                break
        if not omit:
            icone_hrep.append(h)
            if i in cone_eq:
                icone_eq.append(len(icone_hrep)-1)
    return icone_hrep, icone_eq

def unitVectorOf(vector):
    length = math.sqrt(sum(v * v for v in vector))
    unitVector = [v/length for v in vector]
    return unitVector

def findJetFrameHelper(point, partial_jet_frame, vector, halfspaces, eq_indices):
    if (len(partial_jet_frame) == findDimension(h2vRep(halfspaces, eq_indices))):
        return [partial_jet_frame]
    jet_frames= []
    cone_hrep, cone_eq = computeIteratedTagentCone(point, partial_jet_frame, halfspaces, eq_indices)
    cone_ver, cone_ray = cone_h2v(cone_hrep, cone_eq)
    findOrthogonalVector = False
    orthogonal = []
    for ray in cone_ray:
        if (abs(np.dot(vector, ray)) < ERR):
            findOrthogonalVector = True
            orthogonal = list(ray)
    if (findOrthogonalVector):
        new_partial_jet_frame = partial_jet_frame + [unitVectorOf(orthogonal)]
        jet_frames.extend(findJetFrameHelper(point, new_partial_jet_frame, vector, halfspaces, eq_indices))
    else:
        if len(cone_ray) == 0:
            jet_frames.append(partial_jet_frame)
        else:
            max_cosine = max([np.dot(unitVectorOf(ray), unitVectorOf(vector)) for ray in cone_ray])
            for ray in cone_ray:
                cosine = np.dot(unitVectorOf(ray), unitVectorOf(vector))
                if cosine == max_cosine:
                    new_partial_jet_frame = partial_jet_frame + [unitVectorOf(ray)]
                    jet_frames.extend(findJetFrameHelper(point, new_partial_jet_frame, vector, halfspaces, eq_indices))
    return jet_frames

# def findJetFrames(point, vector, halfspaces, eq_indices):
#     jet_frames = []
#     partial_jet_frames = []
#     while (len(partial_jet_frames)!=0):
#         partial_jet_frame = partial_jet_frames.pop()
#         new_pjfs, terminate = findJetFrameHelper(point, partial_jet_frame, vector, halfspaces, eq_indices)
#         if terminate:
#             jet_frames.extend(new_pjfs)
#         else:
#             partial_jet_frames.extend(new_pjfs)
#     return jet_frames

def findAngleSequence(point, vector, halfspaces, eq_indices):
    jet_frames = findJetFrameHelper(point, [], vector, halfspaces, eq_indices)
    if len(jet_frames) == 0:
        return []
    # initialize min angle sequence with max - vector dot product with unit vector of itself
    min_sequence = [np.dot(vector, unitVectorOf(vector))] * len(jet_frames[0])
    for jf in jet_frames:
        sequence = [-np.dot(n, vector) for n in jf]
        if compareAngleSequence(sequence, min_sequence) < 0:
            min_sequence = [s for s in sequence]
    return min_sequence

def compareAngleSequence(s1, s2):
    for i in range(len(s1)):
        if len(s2) <= i:
            return s1[i]
        if s1[i] < s2[i]:
            return -1
        elif s1[i] > s2[i]:
            return 1
    return 0

def chooseEvent(events, source_images, facets, halfspaces):
    min_radius = -1
    min_sequence = []
    closest_event = []
    for event in events:
        point_index = event[0]
        facet_index = event[1]
        hyperplane = halfspaces[facets[facet_index][0]]
        ridge = event[2]
        ridge_halfspaces = ridge[3:]
        ridge_eq_indices = ridge[2]
        # compute vcell, intersection of ridge and vcell
        vcell_h, vcell_eq = findVCell(point_index, source_images[facet_index], hyperplane)
        intersect_h, intersect_eq = intersectPoly(ridge_halfspaces, ridge_eq_indices, vcell_h, vcell_eq)
        try:
            ver = h2vRep(intersect_h, intersect_eq)
            dim = findDimension(ver)
            h, eq = v2hRep(ver)
        except RuntimeError:
            continue
        # find closest point on ridge to the given point
        point = source_images[facet_index][point_index]
        closest_point, radius = computeClosest(point, ver, h, eq, dim)
        # find the min angle sequence for this event
        vector = [point[i]-closest_point[i] for i in range(len(point))]
        angleSequence = findAngleSequence(closest_point, vector, intersect_h, intersect_eq)
        if min_radius == -1 or radius < min_radius:
            min_radius = radius
            min_sequence = angleSequence
            closest_event = [e for e in event]
        elif radius == min_radius and compareAngleSequence(angleSequence, min_sequence) < 0:
            min_sequence = angleSequence
            closest_event = [e for e in event]
    return closest_event

# Given a source point and a polytope, intialize facets, source points, potential events
def initializeProcess(point, vertices, halfspaces, eq_indices):
    facets = computeFacets(vertices, halfspaces, eq_indices)
    sourceImageSets = []
    orderedFacetLists = {}
    for i in range(len(facets)):
        hyperplane = halfspaces[facets[i][0]]
        if onHyperplane(point, hyperplane):
            sourceImageSets.append([point])
        else:
            sourceImageSets.append([])
    return sourceImageSets, orderedFacetLists, events

def plotVector(point, vector, ax):
    X,Y,Z=zip(point)
    U,V,W=zip(vector)
    ax.quiver(X,Y,Z,U,V,W,arrow_length_ratio=0.1)

def foldingMap(point, f_index1, f_index2, facets, halfspaces, eq_indices):
    f1 = facets[f_index1]
    f2 = facets[f_index2]
    h1 = halfspaces[f1[0]]
    h2 = halfspaces[f2[0]]
    projection = projectToAffineSubspace(point, [h1, h2])
    # linear combo of normal vectors to the two hyperplane gives the vector from projection to point after rotation
    n1 = -np.array(h1[1:])
    n2 = -np.array(h2[1:])
    v = np.array(point)-np.array(projection)
    # plotVector(projection, v, ax)

    # if point is on the shared ridge of f1 and f2, it folds to itself
    if np.linalg.norm(v) < ERR:
        return tuple(projection)

    if np.dot(n1, n2) == 0:
        # Then the folded vector will be paralell to n1 and thus a multiple of n1, cannot be solve through linear equations
        ratio = np.linalg.norm(v)/np.linalg.norm(n1)
        mappedPoint = np.array(projection) + ratio * n1
        # plotVector(projection, ratio * n1, ax)
        return tuple(mappedPoint)
    # solving linear combo of normal vectors v'=an1 + bn2
    # equations are (an1 + bn2)n2=0 (v' is on f2), and (an1+bn2)v=|v|^2*cos(v, v') = |v|^2*cos(n1, n2) (v and v' are of same length)
    coefficient_matrix = [[np.dot(n1, n2), np.dot(n2, n2)], [np.dot(n1, v), np.dot(n2, v)]]
    cosine = np.dot(n1, n2) / (np.linalg.norm(np.array(n1)) * np.linalg.norm(np.array(n2)))
    constants = [0, (np.linalg.norm(v) ** 2) * cosine]
    solution = np.linalg.solve(coefficient_matrix, constants)
    folded_vector = solution[0] * n1 + solution[1] * n2
    # find correct direction of folded_vector
    if np.dot(folded_vector, n1) < 0:
        folded_vector = -folded_vector
    mappedPoint = np.array(projection) + folded_vector
    # plotVector(projection, folded_vector, ax)
    return tuple(mappedPoint)

# fold a point along a sequence of adjacent facets
def compositeFolding(point, f_indices, facets, halfspaces, eq_indices, vertices):
    mappedPoint = point
    originalPoint = point
    for i in range(len(f_indices)-1):
        originalPoint = mappedPoint
        mappedPoint = foldingMap(originalPoint, f_indices[i], f_indices[i+1], facets, halfspaces, eq_indices)
    return mappedPoint

def findNeighborFacet(ridge, facet_index):
    neighbor_facet_index = -1
    if ridge[0] == facet_index:
        neighbor_facet_index = ridge[1]
    if ridge[1] == facet_index:
        neighbor_facet_index = ridge[0]
    return neighbor_facet_index

def computeSourceImages(vertices, v, facets, halfspaces, ridges, eq_indices):
    # before looping through events
    # for each facet, initialize a set of points (source images)
    # for each pair of a source image and its facet, initialize an ordered list of facets (ordered by folding)
    # for each facet, initialize a set of potential events
    # compute folding map along all ridges
    source_images = [[] for i in range(len(facets))]
    list_of_facets = [[] for i in range(len(facets))]
    events = [[] for i in range(len(facets))]
    all_events = []

    # initialize source images, list of facets, events with respect to source point v
    for facet_index in range(len(facets)):
        if onHyperplane(v, halfspaces[facets[facet_index][0]]):
            source_images[facet_index].append(v)
            # the index of a list in facet lists associated with facet is the index of source image
            # because when a point is added to source image list, a facet list is defined and added to corresponding list in lists of facets
            list_of_facets[facet_index].append([facet_index])
            for ridge in ridges:
                if ridgeInFacet(ridge, facet_index):
                    # use index for source image and facet to make vcell and chooseEvent convenient
                    events[facet_index].append((0, facet_index, ridge))
            all_events.extend(events[facet_index])
            break

    eventCounter = 0
    while len(all_events) != 0:
        # while events isn't empty:
        # call chooseEvent to pick the event with the min radius and min angle sequence
        # update potential events of F'
        # update potential events of F
        # update potential events in total
        event = chooseEvent(all_events, source_images, facets, halfspaces)
        point_index = event[0]
        facet_index = event[1]
        ridge = event[2]
        # find the neighboring facet that intesect facet at ridge
        # Set F' as the facet adjacent to the facet in the event along the ridge in the event
        facet_2_index = findNeighborFacet(ridge, facet_index)
        point = source_images[facet_index][point_index]
        # Set v' as the point on span of F' that the point in the event is folded to
        mappedPoint = foldingMap(point, facet_index, facet_2_index, facets, halfspaces, eq_indices)
        # add v' to set of source images of F'
        source_images[facet_2_index].append(mappedPoint)
        new_point_index = len(source_images[facet_2_index]) - 1
        # list of facets L_(v',F') = L_(v,F).append(F')
        list_of_facets[facet_2_index].append(list_of_facets[facet_index][point_index] + [facet_2_index])
        #update potential events of F
        events[facet_index] = [e for e in events[facet_index] if not (e[0] == event[0] and e[1] == event[1] and e[2] == event[2])]
        # update potential events of F'
        events[facet_2_index] = []
        for i in range(len(source_images[facet_2_index])):
            w = source_images[facet_2_index][i]
            for r in ridges:
                # check ridge is on facet and source image sees ridge through the facet
                if ridgeInFacet(r, facet_2_index) and seeThrough(i, source_images[facet_2_index], r, facet_2_index, facets, halfspaces):
                    # find neighbor facet with respect to the ridge
                    neighbor_facet_index = findNeighborFacet(r, facet_2_index)
                    # check that w after folding is not in source images of the neighbor facet
                    w_folded = foldingMap(w, facet_2_index, neighbor_facet_index, facets, halfspaces, eq_indices)
                    w_folded_in_source_images = False
                    for s in source_images[neighbor_facet_index]:
                        if np.linalg.norm(np.array(w_folded) - np.array(s))<1e-6:
                            w_folded_in_source_images = True
                            break
                    if not w_folded_in_source_images:
                        events[facet_2_index].append((i, facet_2_index, r))
        #update all events
        all_events = []
        for fi in range(len(facets)):
            all_events.extend(events[fi])
        eventCounter += 1
        print('event', eventCounter, 'complete')

    # for i in range(len(facets)):
    #     print('list_of_facets for facet', i, ':\n', list_of_facets[i])
    #     print('source images for facet', i, ':\n', source_images[i])
    return list_of_facets, source_images

# for all facets and all their source images
# compute the aggregate folding map along the maintained list of facets
# compute a part of unfolding that fold intersection of vcell and facet along inverse map of the aggregate folding map in the step above
# union parts of unfloding across all facets and source images to get the whole unfolding
def unfold(source_images, list_of_facets, facets, halfspaces, eq_indices, vertices):
    unfolded_parts = []
    for fi in range(len(facets)):
        for pi in range(len(source_images[fi])):
            facet_list = list_of_facets[fi][pi]
            facet = facets[fi]
            hyperplane = halfspaces[facet[0]]
            vcell_h, vcell_eq = findVCell(pi, source_images[fi], hyperplane)
            facet_h, facet_eq = v2hRep([vertices[i] for i in facet[1:]])
            intersect_h, intersect_eq = intersectPoly(vcell_h, vcell_eq, facet_h, facet_eq)
            try:
                intersect_v = h2vRep(intersect_h, intersect_eq)
            except RuntimeError:
                continue
            unfolded_intersect_v = []
            for v in intersect_v:
                unfolded_intersect_v.append(compositeFolding(v, facet_list[::-1], facets, halfspaces, eq_indices, vertices))
            unfolded_parts.append(unfolded_intersect_v)
    return unfolded_parts

# convert coordinates of unfolding to one dimension lower by
# setting source point as the origin
# and rotate the normal vector of the hyperplane to the axis of the last dimension
def convertCoordinates(unfolded_parts, source_point, hyperplane):
    converted_unfoldings = []
    normal_vector = np.array(hyperplane[1:])
    axis_vector = np.zeros(normal_vector.size)
    axis_vector[axis_vector.size-1]=1
    if abs(abs(np.dot(normal_vector, axis_vector)) - np.linalg.norm(normal_vector)) <= ERR:
        #normal vector paralell to axis_vector - no need of rotation
        for poly in unfolded_parts:
            converted_poly = []
            for point in poly:
                #compute coordinates with source as origin
                point_from_source =  np.array(point)-np.array(source_point)
                #throw out last dimension
                converted_poly.append(tuple(point_from_source[:len(point)-1]))
            converted_unfoldings.append(converted_poly)
    else:
        #normal vector not paralell to axis vector - need rotation to last axis
        #two cases: normal vector orthogonal to axis vector or not
        if np.dot(normal_vector, axis_vector)/np.linalg.norm(normal_vector) < ERR:
            # normal_vector orthogonal to axis_vector
            unit_normal_vector = normal_vector/np.linalg.norm(normal_vector)
            for poly in unfolded_parts:
                converted_poly = []
                for point in poly:
                    #compute coordinates with source as origin
                    point_from_source =  np.array(point)-np.array(source_point)
                    coordinate_along_axis = np.dot(point_from_source, axis_vector)
                    vector_to_rotate = coordinate_along_axis*axis_vector
                    unchanged_vector = point_from_source - vector_to_rotate
                    rotated_vector = np.linalg.norm(vector_to_rotate) * unit_normal_vector
                    if coordinate_along_axis > 0:
                        rotated_vector = -rotated_vector
                    new_point = (unchanged_vector+rotated_vector)[:rotated_vector.size-1]
                    converted_poly.append(tuple(new_point))
                converted_unfoldings.append(converted_poly)
            return converted_unfoldings
        #non-orthogonal case
        perp_vector = normal_vector - np.dot(normal_vector, axis_vector)*axis_vector
        unit_perp_vector = perp_vector/np.linalg.norm(perp_vector)
        for poly in unfolded_parts:
            converted_poly = []
            for point in poly:
                #compute coordinates with source as origin
                point_from_source =  np.array(point)-np.array(source_point)
                coordinate_along_axis = np.dot(point_from_source, axis_vector)
                coordinate_along_perp = np.dot(point_from_source, unit_perp_vector)
                vector_to_rotate = coordinate_along_axis*axis_vector + coordinate_along_perp*unit_perp_vector
                unchanged_vector = point_from_source - vector_to_rotate
                rotated_vector = np.linalg.norm(vector_to_rotate) * unit_perp_vector
                if coordinate_along_axis < 0:
                    rotated_vector = -rotated_vector
                new_point = (unchanged_vector+rotated_vector)[:rotated_vector.size-1]
                converted_poly.append(tuple(new_point))
            converted_unfoldings.append(converted_poly)
    return converted_unfoldings

def cleanClosePoints(polytope):
    cleaned = []
    for vertex in polytope:
        add = True
        for point in cleaned:
            p = np.array(point)
            v = np.array(vertex)
            if np.linalg.norm(v-p)<ERR:
                # points too close to each other, merge to avoid numerical issue
                add = False
        if add:
            cleaned.append(vertex)
    return cleaned

def cleanUnfoldedParts(unfolded_parts):
    cleaned_unfolded_parts = []
    for poly in unfolded_parts:
        corrected_poly = []
        for point in poly:
            corrected_point = point
            cleaned = False
            for cleaned_poly in cleaned_unfolded_parts:
                if not cleaned:
                    for cleaned_point in cleaned_poly:
                        p = np.array(point)
                        cp = np.array(cleaned_point)
                        if np.linalg.norm(p-cp)<ERR:
                            corrected_point = cleaned_point
                            cleaned = True
                            break
            corrected_poly.append(corrected_point)
        cleaned_unfolded_parts.append(corrected_poly)
        print('corrected part ', len(cleaned_unfolded_parts), 'with ', len(corrected_poly), 'points')
    return cleaned_unfolded_parts

def throwLowerDimension(unfolded_parts):
    cleaned_parts = []
    for poly in unfolded_parts:
        if findDimension(poly) == 3:
            cleaned_parts.append(poly)
    return cleaned_parts

# ===========Below is for testing=================

#compute vertices for 600-cell
v_600_cell = [(0,0,0,2),(0,0,0,-2),(0,0,2,0),(0,0,-2,0),(0,2,0,0),(0,-2,0,0),(2,0,0,0),(-2,0,0,0)]
v_600_cell.extend([(1,1,1,1),(1,1,1,-1),(1,1,-1,1),(1,1,-1,-1),(1,-1,1,1),(1,-1,1,-1),(1,-1,-1,1),(1,-1,-1,-1),(-1,1,1,1),(-1,1,1,-1),(-1,1,-1,1),(-1,1,-1,-1),(-1,-1,1,1),(-1,-1,1,-1),(-1,-1,-1,1),(-1,-1,-1,-1)])
signs = [[1,1,1],[1,1,-1],[1,-1,1],[1,-1,-1],[-1,1,1],[-1,1,-1],[-1,-1,1],[-1,-1,-1]]
a=(1+math.sqrt(5))/2.0
b=1.0
c=(math.sqrt(5)-1)/2.0
d=0

for i in range(8):
    x=a*signs[i][0]
    y=b*signs[i][1]
    z=c*signs[i][2]
    v_600_cell.extend([(x,y,z,d),(y,x,d,z),(z,d,x,y),(d,z,y,x),(y,z,x,d),(z,x,y,d),(y,d,z,x),(d,x,z,y),(z,y,d,x),(d,y,x,z),(x,z,d,y),(x,d,y,z)])

#compute vertices for 120-cell
v_120_cell=[(0,0,2,2),(0,0,2,-2),(0,0,-2,2),(0,0,-2,-2)]
v_120_cell.extend([(0,2,0,2),(0,2,0,-2),(0,-2,0,2),(0,-2,0,-2)])
v_120_cell.extend([(0,2,2,0),(0,2,-2,0),(0,-2,2,0),(0,-2,-2,0)])
v_120_cell.extend([(2,0,0,2),(2,0,0,-2),(-2,0,0,2),(-2,0,0,-2)])
v_120_cell.extend([(2,0,2,0),(2,0,-2,0),(-2,0,2,0),(-2,0,-2,0)])
v_120_cell.extend([(2,2,0,0),(2,-2,0,0),(-2,2,0,0),(-2,-2,0,0)])
signs = [[1,1,1,1],[1,1,1,-1],[1,1,-1,1],[1,1,-1,-1],[1,-1,1,1],[1,-1,1,-1],[1,-1,-1,1],[1,-1,-1,-1],[-1,1,1,1],[-1,1,1,-1],[-1,1,-1,1],[-1,1,-1,-1],[-1,-1,1,1],[-1,-1,1,-1],[-1,-1,-1,1],[-1,-1,-1,-1]]
for i in range(16):
    v1 = (1*signs[i][0],1*signs[i][1],1*signs[i][2],math.sqrt(5)*signs[i][3])
    v2 = ((3-math.sqrt(5))/2.0*signs[i][0],(1+math.sqrt(5))/2.0*signs[i][1],(1+math.sqrt(5))/2.0*signs[i][2],(1+math.sqrt(5))/2.0*signs[i][3])
    v3 = ((math.sqrt(5)-1)/2.0*signs[i][0],(math.sqrt(5)-1)/2.0*signs[i][1],(math.sqrt(5)-1)/2.0*signs[i][2],(3+math.sqrt(5))/2.0*signs[i][3])
    v_120_cell.append(v1)
    v_120_cell.append((v1[0],v1[1],v1[3],v1[2]))
    v_120_cell.append((v1[0],v1[3],v1[2],v1[1]))
    v_120_cell.append((v1[3],v1[1],v1[2],v1[0]))
    v_120_cell.append(v2)
    v_120_cell.append((v2[1],v2[0],v2[2],v2[3]))
    v_120_cell.append((v2[2],v2[1],v2[0],v2[3]))
    v_120_cell.append((v2[3],v2[1],v2[2],v2[0]))
    v_120_cell.append(v3)
    v_120_cell.append((v3[0],v3[1],v3[3],v3[2]))
    v_120_cell.append((v3[0],v3[3],v3[2],v3[1]))
    v_120_cell.append((v3[3],v3[1],v3[2],v3[0]))
    v4 = ((math.sqrt(5)-1)/2.0*signs[i][0],1*signs[i][1],(1+math.sqrt(5))/2.0*signs[i][2],2*signs[i][3])
    v_120_cell.append(v4)
    v_120_cell.extend([(v4[1],v4[0],v4[3],v4[2]),(v4[2],v4[3],v4[0],v4[1]),(v4[3],v4[2],v4[1],v4[0])])
    v_120_cell.extend([(v4[1],v4[2],v4[0],v4[3]),(v4[1],v4[3],v4[2],v4[0]),(v4[2],v4[1],v4[3],v4[0]),(v4[0],v4[2],v4[3],v4[1])])
    v_120_cell.extend([(v4[2],v4[0],v4[1],v4[3]),(v4[3],v4[0],v4[2],v4[1]),(v4[3],v4[1],v4[0],v4[2]),(v4[0],v4[3],v4[1],v4[2])])
signs = [[1,1,1],[1,1,-1],[1,-1,1],[1,-1,-1],[-1,1,1],[-1,1,-1],[-1,-1,1],[-1,-1,-1]]
for i in range(8):
    v5 = (0,(3-math.sqrt(5))/2.0*signs[i][0],1*signs[i][1],(3+math.sqrt(5))/2.0*signs[i][2])
    v_120_cell.append(v5)
    v_120_cell.extend([(v5[1],v5[0],v5[3],v5[2]),(v5[2],v5[3],v5[0],v5[1]),(v5[3],v5[2],v5[1],v5[0])])
    v_120_cell.extend([(v5[1],v5[2],v5[0],v5[3]),(v5[1],v5[3],v5[2],v5[0]),(v5[2],v5[1],v5[3],v5[0]),(v5[0],v5[2],v5[3],v5[1])])
    v_120_cell.extend([(v5[2],v5[0],v5[1],v5[3]),(v5[3],v5[0],v5[2],v5[1]),(v5[3],v5[1],v5[0],v5[2]),(v5[0],v5[3],v5[1],v5[2])])
    v6 = (0,(math.sqrt(5)-1)/2.0*signs[i][0],(1+math.sqrt(5))/2.0*signs[i][1],math.sqrt(5)*signs[i][2])
    v_120_cell.append(v6)
    v_120_cell.extend([(v6[1],v6[0],v6[3],v6[2]),(v6[2],v6[3],v6[0],v6[1]),(v6[3],v6[2],v6[1],v6[0])])
    v_120_cell.extend([(v6[1],v6[2],v6[0],v6[3]),(v6[1],v6[3],v6[2],v6[0]),(v6[2],v6[1],v6[3],v6[0]),(v6[0],v6[2],v6[3],v6[1])])
    v_120_cell.extend([(v6[2],v6[0],v6[1],v6[3]),(v6[3],v6[0],v6[2],v6[1]),(v6[3],v6[1],v6[0],v6[2]),(v6[0],v6[3],v6[1],v6[2])])

# cube
# v = [(0, 0, 0, 0), (1, 0, 0, 0), (1, 1, 0, 0), (0, 1, 0, 0), (0, 0, 1, 0), (1, 0, 1, 0), (1, 1, 1, 0), (0, 1, 1, 0), (0, 0, 0, 1), (1, 0, 0, 1), (1, 1, 0, 1), (0, 1, 0, 1), (0, 0, 1, 1), (1, 0, 1, 1), (1, 1, 1, 1), (0, 1, 1, 1)]
# pyramid
# v = [(0, 0, 0, 0), (1, 0, 0, 0), (1, 1, 0, 0), (0, 1, 0, 0), (0, 0, 1, 0), (1, 0, 1, 0), (1, 1, 1, 0), (0, 1, 1, 0), (0.5, 0.5, 0.5, 1)]
# 24-cell
v = [(1,1,0,0),(1,-1,0,0),(-1,-1,0,0),(-1,1,0,0),(1,0,1,0),(1,0,-1,0),(-1,0,-1,0),(-1,0,1,0),(1,0,0,1),(1,0,0,-1),(-1,0,0,-1),(-1,0,0,1),(0,1,1,0),(0,1,-1,0),(0,-1,-1,0),(0,-1,1,0),(0,1,0,1),(0,1,0,-1),(0,-1,0,-1),(0,-1,0,1),(0,0,1,1),(0,0,1,-1),(0,0,-1,-1),(0,0,-1,1)]
# cross-polytope
# v = [(2,0,0,0),(-2,0,0,0),(0,2,0,0),(0,-2,0,0),(0,0,2,0),(0,0,-2,0),(0,0,0,2),(0,0,0,-2)]
# v = [(0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0), (0, 0, 1), (1, 0, 1), (1, 1, 1), (0, 1, 1)]
# v = [(0, 0, 0), (1, 1, 0), (1, 0, 1), (0, 1, 1)]
# v = [(1, 0, 1), (1, 1, 1), (0, 0, 1), (0, 1, 1), (0.5, 0.5, 0)]
# 600-cell
# v = v_600_cell
# 120-cell
# v = v_120_cell

h, e = v2hRep(v)
# print('h', h)
# print('e', e)
f = computeFacets(v, h, e)
# print('f', f)
r = computeRidges(v, f)

fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
ax = fig.gca(projection='3d')
# # vertices of a pyramid
# v_arr = np.array(v)
# ax.scatter3D(v_arr[:, 0], v_arr[:, 1], v_arr[:, 2])
# # generate list of sides' polygons of our pyramid
# # verts = [ [v[2],v[3],v[4]], [v[0],v[2],v[4]],[v[0],v[1],v[4]], [v[1],v[3],v[4]], [v[0],v[1],v[3],v[2]]]
# # verts = [ [v[2],v[3],v[1]], [v[0],v[1],v[2]],[v[0],v[1],v[3]], [v[0],v[2],v[3]]]
# verts = [ [v[0],v[1],v[2],v[3]], [v[4],v[5],v[6],v[7]],[v[0],v[3],v[7],v[4]], [v[1],v[2],v[6],v[5]], [v[0],v[1],v[5],v[4]], [v[2], v[3],v[7],v[6]]]
# # plot sides
# collection = Poly3DCollection(verts, facecolors='cyan', linewidths=1, edgecolors='r', alpha=.25)
# face_color = [0.5, 0.5, 1]
# collection.set_facecolor(face_color)
# ax.add_collection3d(collection)
# ax.set_xlim(-3,3)
# ax.set_ylim(-3,3)
# ax.set_zlim(-3,3)
# ax.axis('equal')
# limits for 24-cell
ax.axes.set_xlim3d(-3,3)
ax.axes.set_ylim3d(-3,3)
ax.axes.set_zlim3d(-3,3)
# ax.axis('equal')
# limits for 120-cell
# ax.axes.set_xlim3d(-8,8)
# ax.axes.set_ylim3d(-8,8)
# ax.axes.set_zlim3d(-8,8)
# limits for 600-cell
# ax.axes.set_xlim3d(-6,6)
# ax.axes.set_ylim3d(-6,6)
# ax.axes.set_zlim3d(-6,6)

#source point for cube
# source_point = (0.5, 0.5, 0.5, 0)
# source_point = (0, 0, 0.5, 0)
#source point for pyramid
# source_point = (0.5, 0.5, 0.5, 0)
#source point for 24-cell
source_point = (1, 0, 0, 0)
#source point for cross-polytope
# source_point = (0.5, 0.5, 0.5, 0.5)
# source_point = (0.85,0.3,1)
#source point for 600-cell
# source_point = tuple(0.25*(np.array(v[24])+np.array(v[25])+np.array(v[34])+np.array(v[36])))
# source point for 120-cell
# ver_sum = np.zeros(len(v[0]))
# for i in f[0][1:]:
#     ver_sum += np.array(v[i])
# source_point = tuple(0.05*ver_sum)


print('source', source_point)

facetLists, sourceImages = computeSourceImages(v, source_point, f, h, r, e)

# for i in range(len(f)):
#     si_arr = np.array(sourceImages[i])
#     ax.scatter3D(si_arr[:, 0], si_arr[:, 1], si_arr[:, 2])

unfolded_parts = unfold(sourceImages, facetLists, f, h, e, v)
print('number of unfolded parts:',len(unfolded_parts))

#process numerical errors - combine points extremely close to each other within each part
for i in range(len(unfolded_parts)):
    unfolded_parts[i] = cleanClosePoints(unfolded_parts[i])
    print('cleaned part ', i, 'with ', len(unfolded_parts[i]), 'points')

#process numerical errors - combine points extremely close to each other across different parts
unfolded_parts = cleanUnfoldedParts(unfolded_parts)

#convert coordinates to one dimension lower
hyperplane=[]
for halfspace in h:
    if onHyperplane(source_point, halfspace):
        hyperplane = halfspace
        break
flattened_unfoldings = convertCoordinates(unfolded_parts, source_point, hyperplane)
print('number of unfolded parts:',len(unfolded_parts))

for i in range(len(flattened_unfoldings)):
    flattened_unfoldings[i] = cleanClosePoints(flattened_unfoldings[i])
    print('cleaned part ', i, 'with ', len(unfolded_parts[i]), 'points')

flattened_unfoldings = cleanUnfoldedParts(flattened_unfoldings)

flattened_unfoldings = throwLowerDimension(flattened_unfoldings)

# print('flattened', flattened_unfoldings)

#write each unfolded part to a csv file
# for i in range(len(flattened_unfoldings)):
#     filepath = 'pyramid/part'+str(i)+'.csv'
#     with open(filepath, 'w') as csvfile:
#         writer = csv.writer(csvfile, delimiter=',')
#         for point in flattened_unfoldings[i]:
#             writer.writerow([point[0], point[1], point[2]])

for j in range(len(flattened_unfoldings)):
    flat_poly = flattened_unfoldings[j]
    # v_arr = np.array(flat_poly)
    # ax.scatter3D(v_arr[:, 0], v_arr[:, 1], v_arr[:, 2])
    try:
        flat_poly_h, flat_poly_e=v2hRep(flat_poly)
    except RuntimeError:
        continue
    flat_poly_f = computeFacets(flat_poly, flat_poly_h, flat_poly_e)
    reordered_vertices = []
    for face in flat_poly_f:
        #reorder face index to draw 2d face correctly
        face_v = [flat_poly[k] for k in face[1:]]
        # skip if face_v is empty
        if len(face_v) == 0:
            continue
        face_h, face_e = v2hRep(face_v)
        #compute facets of each facet -- 2d edge to get adjacent vertices
        face_f = computeFacets(face_v, face_h, face_e)
        # if each edge has <=1 vertices, skip drawing this since it's of lower dimension
        if len(face_f[0]) < 3:
            continue

        reordered_face_ver = [0]
        temp = 0
        for i in range(1, len(face_v)):
            for edge in face_f:
                if temp in edge[1:]:
                    new_ver_index = -1
                    if edge[1] == temp:
                        new_ver_index = edge[2]
                    else:
                        new_ver_index = edge[1]
                    # add newly found adjacent vertex if it's the first iteration or if it hasn't been added immediately before current
                    if len(reordered_face_ver) == 1 or reordered_face_ver[len(reordered_face_ver)-2]!=new_ver_index:
                            reordered_face_ver.append(new_ver_index)
                            temp = new_ver_index
                            break
        reordered_vertices.append([face_v[reordered_face_ver[k]] for k in range(len(reordered_face_ver))])
    flat_poly_arr = np.array(flat_poly)
    ax.scatter3D(flat_poly_arr[:, 0],flat_poly_arr[:, 1],flat_poly_arr[:, 2])
    collection = Poly3DCollection(reordered_vertices, alpha=0.6)
    face_color = [0.5, 1, 1]
    collection.set_facecolor('cyan')
    edge_color = [0.5, 0.5, 1.0]
    collection.set_edgecolor(edge_color)
    ax.add_collection3d(collection)

# ax.set_autoscale_on(False)
# ax.axis('equal')
plt.show()

# h_120, e_120 = v2hRep(v_120_cell)
# f_120 = computeFacets(v_120_cell, h_120, e_120)
# print('v120', len(v_120_cell))
# print('f0', f_120[0])
# print('f0_ver', f_120[0][1:])
# print('h0', h_120[0])
