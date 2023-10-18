import numpy as np
import scipy


#############################################################################
# Make the temperature matrix (discrete Laplacian operator) in 2 dimensions #
#############################################################################


#############################################################################
# Define the boundary condition                                             #
#############################################################################

def boundary_condition(x, y):
    return np.cos(2 * np.pi * x) + np.sin(2 * np.pi * y)


# end of boundary_condition

#############################################################################
# Define the Source term
#############################################################################

def S(x, y):
    return 4 * np.pi ** 2 * (np.cos(2 * np.pi * x) + np.sin(2 * np.pi * y))


# end of S

#############################################################################
# Make the temperature matrix and right-hand side in 2 dimensions           #
#############################################################################

def make_linear_system(xmin, xmax, ymin, ymax, mX, mY):
    """Create the matrix of the discrete Laplacian operator in two dimensions on a mX-by-mY grid.
    Parameters:
      :param xmin: interval [xmin, xmax]
      :param xmax: interval [xmin, xmax]
      :param ymin: interval [ymin, ymax]
      :param ymax: interval [ymin, ymax]
      :param mX  : number of grid points in the x-direction.
      :param mY  : number of grid points in the y-direction.
    Outputs:
      A  : the sparse mX*mY-by-mX*mY matrix representing the finite difference approximation to Laplace's equation.
      rhs: the right-hand side of the linear system

    """
    # Define the grid
    x = np.linspace(xmin, xmax, mX)
    y = np.linspace(ymin, ymax, mY)

    dx = x[1] - x[0]
    dy = y[1] - y[0]

    one_over_dx_square = 1 / dx / dx
    one_over_dy_square = 1 / dy / dy

    # Start with a vector of zeros
    ndim = mX * mY
    rhs = np.zeros(shape=ndim)

    # First make a list with one triple (row, column, value) for each nonzero element of A
    triples = []

    # Treat the interior points:
    for i in range(1, mX - 1):
        for j in range(1, mY - 1):
            # what row of the matrix is grid point (i, j)?
            row = j*mX + i
            # fill out the row in the matrix:
            triples.append((row, row, 2.0 * one_over_dx_square + 2.0 * one_over_dy_square))
            triples.append((row, row - 1, -1.0 * one_over_dx_square))
            triples.append((row, row + 1, -1.0 * one_over_dx_square))
            triples.append((row, row - mX, -1.0 * one_over_dy_square))
            triples.append((row, row + mX, -1.0 * one_over_dy_square))
            rhs[row] = S(x[i], y[j])

    # Treat the boundary points:
    for i in range(mX):
        j = 0  # bottom wall
        row = j*mX + i
        triples.append((row, row, 1.0))
        rhs[row] = boundary_condition(x[i], y[j])
        j = mY - 1  # top wall
        row = j*mX + i
        triples.append((row, row, 1.0))
        rhs[row] = boundary_condition(x[i], y[j])

    for j in range(1, mY - 1):  # Need to avoid 0 and mY-1 because duplicate entries are summed up with csr_matrix
        i = 0  # left wall
        row = j*mX + i
        triples.append((row, row, 1.0))
        rhs[row] = boundary_condition(x[i], y[j])
        i = mX - 1  # right wall
        row = j*mX + i
        triples.append((row, row, 1.0))
        rhs[row] = boundary_condition(x[i], y[j])

    # Finally convert the list of triples to a scipy sparse matrix
    rownum = [t[0] for t in triples]
    colnum = [t[1] for t in triples]
    values = [t[2] for t in triples]
    A = scipy.sparse.csr_matrix((values, (rownum, colnum)), shape=(ndim, ndim))

    return A, rhs


# end of make__linear_system


#############################################################################
# Make a 16-by-16 version of the temperature matrix for demos               #
#############################################################################

def make_A_small():
    """Make a small 4x4 version of the temperature matrix, as a dense array"""
    A = make_A(4)
    return A.toarray()


# end of make_A_small


#############################################################################
# Make a right-hand side vector for the 2D Laplacian / temperature matrix   #
#############################################################################

def make_b(k, top=0, bottom=0, left=0, right=0):
    """Create the right-hand side for the temperature problem on a k-by-k grid.
    Parameters: 
      k: number of grid points in each dimension.
      top: list of k values for top boundary (optional, defaults to 0)
      bottom: list of k values for bottom boundary (optional, defaults to 0)
      left: list of k values for top boundary (optional, defaults to 0)
      right: list of k values for top boundary (optional, defaults to 0)
    Outputs:
      b: the k**2 element vector (as a numpy array) for the rhs of the Poisson equation with given boundary conditions
    """
    # Start with a vector of zeros
    ndim = k * k
    rhs = np.zeros(shape=ndim)

    # Fill in the four boundaries as appropriate
    rhs[0: k] += top
    rhs[ndim - k: ndim] += bottom
    rhs[0: ndim: k] += left
    rhs[k - 1: ndim: k] += right

    return rhs


# end of make_rhs


#############################################################################
# Make a size-16 version of the right-hand side for demos                   #
#############################################################################

def make_b_small():
    """Make a small k=4 version of the right-hand side vector"""
    return make_b(4, top=radiator(4))


# End of make_rhs_small


#############################################################################
# Make one wall with a radiator                                             #
#############################################################################

def radiator(k, width=.3, temperature=100., default=0):
    """
	Create one wall with a radiator
	Parameters:
	k: number of grid points in each dimension; length of the wall
	width: width of the radiator as a fraction of length of the wall
	temperature: temperature of the radiator
	default: default temperature for the rest of the elements along the wall
	Outputs:
	wall: the k element vector (as a numpy array) for the boundary conditions at the wall
	"""
    rad_start = int(k * (0.5 - width / 2))
    rad_end = int(k * (0.5 + width / 2))
    wall = np.zeros(k) + default
    wall[rad_start: rad_end] = temperature

    return wall
# End of radiator

#%%
