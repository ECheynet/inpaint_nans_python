import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

def inpaint_nans(A, method=0):
    """
    Replicates John D’Errico’s inpaint_nans MATLAB function in Python.
    
    Parameters
    ----------
    A : 2D array_like
        Input array with NaNs to be filled.
    method : int, optional
        Which PDE/finite-difference scheme to use (0 to 5). Default is 0.
    
    Returns
    -------
    B : 2D ndarray
        Array with NaNs replaced/inpainted.
    """
    # Convert input A to a float array in Fortran (column-major) order
    A = np.array(A, dtype=float, order='F')
    n, m = A.shape
    nm = n * m
    
    # Flatten (column-major) so linear indexing matches MATLAB's
    A_flat = A.ravel(order='F')
    
    # Find NaNs
    isnan = np.isnan(A_flat)
    nan_list = np.where(isnan)[0]   # these are 0-based linear indices
    known_list = np.where(~isnan)[0]
    nan_count = len(nan_list)
    if nan_count == 0:
        # No NaNs, nothing to do
        return A  # already a copy
    
    # Convert those nan indices to row/col, 1-based analog to MATLAB
    #   In MATLAB: [nr, nc] = ind2sub([n,m], nan_list)
    #   We store them as 1-based to match the original boundary logic,
    #   then keep them in a combined array: [lin_idx, row, col]
    #   where row,col in [1..n], [1..m].
    nr = (nan_list % n) + 1
    nc = (nan_list // n) + 1
    nan_list_info = np.column_stack([nan_list, nr, nc])
    
    # --- Helper: build a function to do the solve or least-squares:
    def sparse_solve(M, rhs):
        """
        Solve M x = rhs.
        Uses direct solve if M is square and has full rank,
        otherwise use lsqr for least-squares.
        """
        # M should be shape (R, C). If R==C, try spsolve:
        R, C = M.shape
        if R == C:
            return spla.spsolve(M.tocsc(), rhs)
        else:
            # Use lsqr for least squares
            sol = spla.lsqr(M, rhs, atol=1e-12, btol=1e-12, iter_lim=10_000)
            return sol[0]
    
    # We’ll need a function to eliminate known values from RHS, similarly to MATLAB
    def eliminate_knowns(fda, known_idx, A_known):
        """
        Return the adjusted rhs after applying -fda[:, known_idx] * A_known,
        i.e. the part that remains for the unknown columns.
        """
        rhs = -fda[:, known_idx].dot(A_known)
        return rhs
    
    # Switch on 'method' just like the original code
    if method not in [0,1,2,3,4,5]:
        raise ValueError("method must be one of {0,1,2,3,4,5}.")
    
    # We will build 'fda' (the finite-difference operator) as a sparse matrix
    # in each method, then solve or least-squares for the unknowns.
    B = A_flat.copy()
    
    # ----- Subfunction from the original code, in Python:
    def identify_neighbors(n, m, nan_list_3col, talks_to):
        """
        Identify neighbors of the NaN pixels, not including the NaNs themselves.
        
        nan_list_3col: array of shape (N,3): [lin_index, row, col]
                       with row,col in 1-based coords.
        talks_to: array of shape (p,2) of row/col offsets
        """
        if nan_list_3col.shape[0] == 0:
            return np.empty((0,3), dtype=int)
        
        nan_count_local = nan_list_3col.shape[0]
        talk_count = talks_to.shape[0]
        
        # For each offset in talks_to, add to row,col
        # shape => (nan_count_local * talk_count, 2)
        repeated = np.repeat(nan_list_3col[:,1:3], talk_count, axis=0)  # row,col repeated
        offsets = np.tile(talks_to, (nan_count_local, 1))
        nn = repeated + offsets  # neighbor row,col (1-based)
        
        # Filter out-of-bounds neighbors
        in_bounds = (nn[:,0] >= 1) & (nn[:,0] <= n) & \
                    (nn[:,1] >= 1) & (nn[:,1] <= m)
        nn = nn[in_bounds]
        
        # Convert (row,col) back to linear index in 0-based
        #  MATLAB 1-based: lin = row + (col-1)*n
        #  Python 0-based: lin = (row-1) + (col-1)*n
        # so if row,col is 1-based, we do:
        lin_idx = (nn[:,0] - 1) + (nn[:,1] - 1)*n
        
        neighbors_list = np.column_stack([lin_idx, nn])
        
        # Unique rows and remove those that are in the NaN list
        neighbors_list = np.unique(neighbors_list, axis=0)
        
        # Build a set of all nan linear indices for easy filter
        nan_lin_set = set(nan_list_3col[:,0].tolist())
        
        # Keep only those not themselves in the NaN set
        mask_not_nan = np.array([ (x[0] not in nan_lin_set)
                                  for x in neighbors_list ], dtype=bool)
        neighbors_list = neighbors_list[mask_not_nan]
        return neighbors_list
    
    # Because the methods differ widely, we implement them in turn:
    if method == 0 or method == 3:
        # Methods 0 and 3 are similar to method 1 but build the matrix only
        # around the nans and their neighbors, then do a del^2 or del^4 solve.
        
        # If method=0 => del^2, if method=3 => del^4
        if method == 0:
            # del^2 using neighbors: up/down/left/right
            # We only build around the nans + their immediate neighbors
            #  for 2D or 1D.  If 1D, treat specially.
            
            if n == 1 or m == 1:
                # 1D case
                # Identify the "work_list" as nan +/- 1
                work_list = np.concatenate([nan_list,
                                            nan_list - 1,
                                            nan_list + 1])
                work_list = work_list[(work_list >= 0) & (work_list < nm)]
                work_list = np.unique(work_list)
                
                # Build fda
                # For each i in work_list, we want i-1, i, i+1 with [1, -2, 1]
                # We'll do that in a lil_matrix
                fda = sp.lil_matrix((len(work_list), nm), dtype=float)
                # Fill row-by-row
                for row_idx, i in enumerate(work_list):
                    # center
                    fda[row_idx, i] = -2.0
                    if i - 1 >= 0:
                        fda[row_idx, i - 1] = 1.0
                    if i + 1 < nm:
                        fda[row_idx, i + 1] = 1.0
                        
                # Eliminate knowns
                rhs = eliminate_knowns(fda, known_list, A_flat[known_list])
                
                # We only solve for columns in nan_list
                unknown_idx = nan_list
                # We only keep rows that reference those columns
                # i.e. any row with a non-zero in unknown columns
                mask_rows = (fda[:, unknown_idx].sum(axis=1).A.ravel() != 0)
                row_sel = np.where(mask_rows)[0]
                
                fda_sub = fda[row_sel,:][:,unknown_idx]
                rhs_sub = rhs[row_sel]
                
                sol = sparse_solve(fda_sub, rhs_sub)
                
                # Place solution
                B[unknown_idx] = sol
                
            else:
                # 2D case
                # Horizontal and vertical neighbors only
                talks_to = np.array([[-1,0],[1,0],[0,-1],[0,1]])
                neighbors_list = identify_neighbors(n, m, nan_list_info, talks_to)
                all_list = np.vstack([nan_list_info, neighbors_list])
                
                # Build fda
                fda = sp.lil_matrix((nm, nm), dtype=float)
                
                # second partials row-wise: (row > 1 & row < n)
                L = np.where((all_list[:,1] > 1) & (all_list[:,1] < n))[0]
                for i in L:
                    idx = all_list[i,0]
                    fda[idx, idx] += -2.0
                    fda[idx, idx-1] += 1.0
                    fda[idx, idx+1] += 1.0
                
                # second partials col-wise: (col > 1 & col < m)
                L = np.where((all_list[:,2] > 1) & (all_list[:,2] < m))[0]
                for i in L:
                    idx = all_list[i,0]
                    fda[idx, idx] += -2.0
                    fda[idx, idx - n] += 1.0
                    fda[idx, idx + n] += 1.0
                
                # Eliminate knowns
                rhs = eliminate_knowns(fda, known_list, A_flat[known_list])
                
                # Solve only for relevant rows & columns
                unknown_idx = nan_list
                mask_rows = (fda[:, unknown_idx].sum(axis=1).A.ravel() != 0)
                row_sel = np.where(mask_rows)[0]
                
                fda_sub = fda[row_sel,:][:,unknown_idx]
                rhs_sub = rhs[row_sel]
                sol = sparse_solve(fda_sub, rhs_sub)
                
                B[unknown_idx] = sol
        
        else:
            # method == 3 => "better plate" using del^4
            # We use bigger stencils
            #  The code is quite extensive, we replicate the logic:
            
            # neighbors for the center region
            talks_to = np.array([[-2,0],[-1,-1],[-1,0],[-1,1],
                                 [0,-2],[0,-1],[0,1],[0,2],
                                 [1,-1],[1,0],[1,1],[2,0]])
            neighbors_list = identify_neighbors(n, m, nan_list_info, talks_to)
            all_list = np.vstack([nan_list_info, neighbors_list])
            
            fda = sp.lil_matrix((nm, nm), dtype=float)
            
            # main interior: row>=3 & row<=n-2 & col>=3 & col<=m-2
            L = np.where((all_list[:,1] >= 3) & (all_list[:,1] <= n-2) &
                         (all_list[:,2] >= 3) & (all_list[:,2] <= m-2))[0]
            # fill with the big 13-point stencil
            # Coeffs: [1 2 -8 2 1 -8 20 -8 1 2 -8 2 1], 
            # Offsets in linear indices: [-2n, -(n+1), -n, -(n-1), -2, -1, 0, +1, +2, (n-1), +n, (n+1), +2n]
            base_offsets = np.array([-2*n, -n-1, -n, -n+1,
                                     -2,    -1,    0,   1,   2,
                                      n-1,   n,   n+1,  2*n])
            base_coeffs = np.array([1,2,-8,2, 1,-8,20,-8,1, 2,-8,2,1], dtype=float)
            for i in L:
                idx = all_list[i,0]
                for off, coef in zip(base_offsets, base_coeffs):
                    fda[idx, idx+off] += coef
            
            # boundaries near row=2 or row=n-1 or col=2 or col=m-1
            # do a simpler 5-point Laplacian: [1 -4 1], etc.
            # the original code lumps all boundary expansions.  For brevity, replicate:
            
            # row=2 or row=n-1 or col=2 or col=m-1
            L = np.where((((all_list[:,1] == 2) | (all_list[:,1] == n-1)) &
                          (all_list[:,2] >= 2) & (all_list[:,2] <= m-1)) |
                         (((all_list[:,2] == 2) | (all_list[:,2] == m-1)) &
                          (all_list[:,1] >= 2) & (all_list[:,1] <= n-1)))[0]
            # 5-point: offsets = [-n, -1, 0, +1, +n], coeff = [1,1,-4,1,1]
            offsets_5 = np.array([-n, -1, 0, 1, n])
            coeffs_5 = np.array([1,1,-4,1,1], dtype=float)
            for i in L:
                idx = all_list[i,0]
                for off, c in zip(offsets_5, coeffs_5):
                    fda[idx, idx+off] += c
            
            # row=1 or row=n, col in [2..m-1]
            L = np.where(((all_list[:,1] == 1) | (all_list[:,1] == n)) &
                         (all_list[:,2] >= 2) & (all_list[:,2] <= m-1))[0]
            # 3-point vertical second derivative: offsets = [-n, 0, +n], coeffs = [1, -2, 1]
            offsets_3v = np.array([-n, 0, n])
            coeffs_3v = np.array([1, -2, 1], dtype=float)
            for i in L:
                idx = all_list[i,0]
                for off, c in zip(offsets_3v, coeffs_3v):
                    fda[idx, idx+off] += c
            
            # col=1 or col=m, row in [2..n-1]
            L = np.where(((all_list[:,2] == 1) | (all_list[:,2] == m)) &
                         (all_list[:,1] >= 2) & (all_list[:,1] <= n-1))[0]
            # 3-point horizontal second derivative: offsets = [-1, 0, +1], coeffs = [1, -2, 1]
            offsets_3h = np.array([-1, 0, 1])
            coeffs_3h = np.array([1, -2, 1], dtype=float)
            for i in L:
                idx = all_list[i,0]
                for off, c in zip(offsets_3h, coeffs_3h):
                    fda[idx, idx+off] += c
            
            # Eliminate knowns
            rhs = eliminate_knowns(fda, known_list, A_flat[known_list])
            
            # Solve
            unknown_idx = nan_list
            mask_rows = (fda[:, unknown_idx].sum(axis=1).A.ravel() != 0)
            row_sel = np.where(mask_rows)[0]
            fda_sub = fda[row_sel,:][:,unknown_idx]
            rhs_sub = rhs[row_sel]
            sol = sparse_solve(fda_sub, rhs_sub)
            
            B[unknown_idx] = sol
    
    elif method == 1:
        # Least squares with del^2 on the entire array
        # Build the Laplacian operator for all points
        if n == 1 or m == 1:
            # 1D
            # second difference for interior points
            # row i => i=1..(nm-2) in 0-based => fill [i, i+1, i+2]
            # but we'll do it more systematically:
            fda = sp.lil_matrix((nm-2, nm), dtype=float)
            for i in range(nm-2):
                fda[i, i  ] =  1.0
                fda[i, i+1] = -2.0
                fda[i, i+2] =  1.0
            
        else:
            # 2D
            fda = sp.lil_matrix((nm, nm), dtype=float)
            # Row-second-derivatives for i=2..n-1 => index = i+(j-1)*n
            # We'll just loop or systematically fill them:
            for j in range(m):
                for i in range(1, n-1):
                    idx = i + j*n
                    # i => row, j => col in 0-based, so fda[idx, idx +/- 1]
                    fda[idx, idx]   += -2.0
                    fda[idx, idx-1] +=  1.0
                    fda[idx, idx+1] +=  1.0
            
            # Column-second-derivatives for j=2..m-1 => index i+(j-1)*n
            for j in range(1, m-1):
                for i in range(n):
                    idx = i + j*n
                    fda[idx, idx]     += -2.0
                    fda[idx, idx - n] +=  1.0
                    fda[idx, idx + n] +=  1.0
        
        # Eliminate knowns
        rhs = eliminate_knowns(fda, known_list, A_flat[known_list])
        
        # Solve
        unknown_idx = nan_list
        mask_rows = (fda[:, unknown_idx].sum(axis=1).A.ravel() != 0)
        row_sel = np.where(mask_rows)[0]
        fda_sub = fda[row_sel,:][:,unknown_idx]
        rhs_sub = rhs[row_sel]
        sol = sparse_solve(fda_sub, rhs_sub)
        
        B[unknown_idx] = sol
    
    elif method == 2:
        # Direct solve for del^2 BVP across holes only
        if n == 1 or m == 1:
            raise ValueError("Method 2 has problems for 1D input. Use another method.")
        else:
            # 2D
            fda = sp.lil_matrix((nm, nm), dtype=float)
            
            # second partials on row index
            L = np.where((nan_list_info[:,1] > 1) & (nan_list_info[:,1] < n))[0]
            for i in L:
                idx = nan_list_info[i,0]
                fda[idx, idx]   += -2.0
                fda[idx, idx-1] +=  1.0
                fda[idx, idx+1] +=  1.0
            
            # second partials on column index
            L = np.where((nan_list_info[:,2] > 1) & (nan_list_info[:,2] < m))[0]
            for i in L:
                idx = nan_list_info[i,0]
                fda[idx, idx]     += -2.0
                fda[idx, idx - n] +=  1.0
                fda[idx, idx + n] +=  1.0
            
            # fix boundary corners if they are NaN
            corners = [0, n-1, nm-n, nm-1]  # 0-based corners in Fortran?  
            # Actually, in column-major: top-left = 0, bottom-left = n-1,
            # top-right = (m-1)*n, bottom-right = nm-1
            # The original code forces certain patterns if those corners are in nan_list
            for c in corners:
                if c in nan_list:
                    # replicate the code: fda(c, [c c+...]) = ...
                    # corner examples from the original
                    if c == 0:
                        fda[c, c]   = -2.0
                        fda[c, c+1] += 1.0
                        fda[c, c+n] += 1.0
                    elif c == n-1:
                        fda[c, c]     = -2.0
                        fda[c, c-1]   += 1.0
                        fda[c, c+n]   += 1.0
                    elif c == (m-1)*n:
                        fda[c, c]     = -2.0
                        fda[c, c+1]   += 1.0
                        fda[c, c-n]   += 1.0
                    elif c == nm-1:
                        fda[c, c]     = -2.0
                        fda[c, c-1]   += 1.0
                        fda[c, c-n]   += 1.0
            
            # Eliminate knowns
            rhs = eliminate_knowns(fda, known_list, A_flat[known_list])
            
            # Solve directly on the nan_list subset
            unknown_idx = nan_list
            fda_sub = fda[unknown_idx,:][:, unknown_idx]
            rhs_sub = rhs[unknown_idx]
            sol = sparse_solve(fda_sub, rhs_sub)
            
            B[unknown_idx] = sol
    
    elif method == 4:
        # Spring analogy, only horizontal + vertical neighbors
        # Diagonals in the original code are not used or are they? Actually,
        # code has "hv_list=[-1  -1 0; 1 1 0; -n 0 -1; n 0 1]", but that’s
        # for HV.  (No diagonal in the original code method 4.)
        
        # We'll build a matrix of "springs"
        hv_list = np.array([[-1, 0], [1, 0], [0, -1], [0, 1]])  # row/col offsets
        # But in the original code, it actually used:
        #   hv_list = [ -1 -1 0
        #               1  1  0
        #              -n  0 -1
        #               n  0  1 ]
        # That was building pairs of (index, index+...). It's simpler to replicate logic:
        
        # Let’s explicitly gather pairs of neighbors among the nan_list:
        springs = []
        for i in range(nan_count):
            idx   = nan_list_info[i,0]
            row_i = nan_list_info[i,1]
            col_i = nan_list_info[i,2]
            # up/down/left/right neighbors
            # up => if row_i>1 => idx-n
            if row_i > 1:
                springs.append([idx, idx - 1])     # in col-major, up is -1
            if row_i < n:
                springs.append([idx, idx + 1])     # down is +1
            if col_i > 1:
                springs.append([idx, idx - n])     # left is -n
            if col_i < m:
                springs.append([idx, idx + n])     # right is +n
        
        # Unique + sort each pair
        springs = np.array(springs)
        # Sort rows so that [min,max]
        springs.sort(axis=1)
        # Unique
        springs = np.unique(springs, axis=0)
        
        # Build the sparse system: each spring => row in the system, [1 -1] for those two columns
        n_springs = springs.shape[0]
        S = sp.lil_matrix((n_springs, nm), dtype=float)
        for i in range(n_springs):
            i1, i2 = springs[i]
            S[i, i1] =  1.0
            S[i, i2] = -1.0
        
        # Right side after eliminating known
        rhs = - S[:, known_list].dot(B[known_list])
        
        # Solve only for nan columns
        unknown_idx = nan_list
        S_sub = S[:, unknown_idx]
        sol = sparse_solve(S_sub, rhs)
        
        B[unknown_idx] = sol
    
    elif method == 5:
        # Average of 8 nearest neighbors
        # The code builds an operator that enforces B(i) = average(8 neighbors).
        # That translates to sum_of_neighbors - 8*B(i) = 0, or equivalently
        # for each i in nan_list, sum_{neighbors} (B(neighbor)) - 8 B(i) = 0.
        
        fda = sp.lil_matrix((nm, nm), dtype=float)
        
        def add_avg_equation(center_idx, neighbor_idx):
            # for eq: B(neighbor_idx) - B(center_idx)
            fda[center_idx, neighbor_idx] += 1.0
            fda[center_idx, center_idx]   += -1.0
        
        # We replicate the "if" blocks from MATLAB for the 8 neighbors
        # We only do it for i in nan_list
        for i in nan_list:
            # row,col in 1-based
            r = (i % n) + 1
            c = (i // n) + 1
            # top-left => if r>1,c>1 => i-(n+1)
            if (r > 1) and (c > 1):
                add_avg_equation(i, i - n - 1)
            # top => if c>1 => i-n
            if (c > 1):
                add_avg_equation(i, i - n)
            # top-right => if r<n,c>1 => i-(n-1)
            if (r < n) and (c > 1):
                add_avg_equation(i, i - n + 1)
            # left => if r>1 => i-1
            if (r > 1):
                add_avg_equation(i, i - 1)
            # right => if r<n => i+1
            if (r < n):
                add_avg_equation(i, i + 1)
            # bottom-left => if r>1,c<m => i+(n-1)
            if (r > 1) and (c < m):
                add_avg_equation(i, i + n - 1)
            # bottom => if c<m => i+n
            if (c < m):
                add_avg_equation(i, i + n)
            # bottom-right => if r<n,c<m => i+(n+1)
            if (r < n) and (c < m):
                add_avg_equation(i, i + n + 1)
        
        # Eliminate known
        rhs = eliminate_knowns(fda, known_list, B[known_list])
        
        # Solve for unknown
        unknown_idx = nan_list
        fda_sub = fda[unknown_idx,:][:, unknown_idx]
        rhs_sub = rhs[unknown_idx]
        sol = sparse_solve(fda_sub, rhs_sub)
        
        B[unknown_idx] = sol
    
    # Reshape back to (n,m) in Fortran order
    B = np.reshape(B, (n, m), order='F')
    return B

#%%

