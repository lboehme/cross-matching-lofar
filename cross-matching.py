import numpy as np
from astropy.coordinates import SkyCoord, search_around_sky
from astropy import units as u


def f_gauss(x,y, A,x0,y0,sx,sy,theta):
    a = np.cos(theta)**2/(2*sx**2) + np.sin(theta)**2/(2*sy**2)
    b = -np.sin(2*theta)/(4*sx**2) + np.sin(2*theta)/(4*sy**2)
    c = np.sin(theta)**2/(2*sx**2) + np.cos(theta)**2/(2*sy**2)
    return A*np.exp(-(a*(x-x0)**2 + 2*b*(x-x0)*(y-y0) + c*(y-y0)**2))
    
def leng(x):
    if type(x) == int or type(x) == np.int64 or type(x) == float:
        return 1
    else:
        return len(x)
    
def cross_match(idLX_L, idLX_X, RA_L,DEC_L,MAJ_L,MIN_L,PA_L, RA_X,DEC_X,MAJ_X,MIN_X,PA_X, v_thr):
    # This is the main cross-matching function. It compares the Gaussians of potential matches
    # and according to the threshold value returns a list of accepted matches
    # with indices into L and X

    MY_LX_mid = []
    MY_LX_max = []

    for i,v in enumerate(idLX_L): #v = Index of L source
        v1 = idLX_X[i] #v1 = Index of X source
        
        #Case 1: Calculate Gaussian at potential counterpart
        m_val = f_gauss(RA_X[v1] ,DEC_X[v1], 1, RA_L[v], DEC_L[v], MAJ_L[v]/2.355, MIN_L[v]/2.355, PA_L[v]+np.pi/2) # + Pi/2 = 90° as angle from x-axis
        
        #Case 2: Draw line between both source center
        x = np.arange(RA_L[v]-MAJ_L[v], RA_L[v]+MAJ_L[v], 0.0001)
        m = (DEC_L[v]-DEC_X[v1])/(RA_L[v]-RA_X[v1])
        bl = (RA_X[v1]*DEC_L[v] - RA_L[v]*DEC_X[v1])/(RA_X[v1]-RA_L[v])
        fline = m*x+bl
        
        # And calculate Gaussian at intersection of line and FWHM of pot. counterpart
        val2 = f_gauss(x, fline, 1, RA_X[v1], DEC_X[v1], MAJ_X[v1]/2.355, MIN_X[v1]/2.355, PA_X[v1]+np.pi/2) #Gaussian values (from X) along line to find FWHM
        ind, = np.where(np.logical_and(val2 < 0.51, val2>0.49)) #Intersection
        val_gauss2 = f_gauss(x[ind], fline[ind], 1, RA_L[v], DEC_L[v], MAJ_L[v]/2.355, MIN_L[v]/2.355, PA_L[v]+np.pi/2) #Gaussian (of L) at FWHM
        
        try:
            max_LR = np.max(val_gauss2) #Intersection ind can be more than one value
        except:
            max_LR = m_val
            
        MY_LX_mid.append(m_val) #Append values to list
        MY_LX_max.append(max_LR)
        
        
    MY_LX_mid = np.array(MY_LX_mid)
    MY_LX_max = np.array(MY_LX_max)

    accLX_L = idLX_L[np.logical_or(MY_LX_mid>v_thr,MY_LX_max>v_thr*1.1)] #Accept if value is above threshold or at FWHM 1.1 times above threshold
    accLX_X = idLX_X[np.logical_or(MY_LX_mid>v_thr,MY_LX_max>v_thr*1.1)]

    return accLX_L, accLX_X
    
   

def cross_id(accLX_L, accLX_X):
    # Returns one array - mat_X - which describes all cross-matches
    # between catalogue L and X.
    # The index describes the source from catalogue L and the entry
    # the matches in catalogue X, either as a single number
    # or as an array 

    mat_X = np.arange(0,len(RA_L),dtype=object)

    for l,n in zip(accLX_L,accLX_X): #l,n describe the matched sources, l in L, n in X
        if type(mat_X[l]) == np.ndarray: #if entry is already array, append new match to array
            mat_X[l] = np.append(mat_X[l],n)
        elif mat_X[l] == l: #mat_X is initialized as arange, so if entry at l is unset (=l), set it to the match n
            mat_X[l] = n
        elif type(mat_X[l]) in [int, np.int64]: #If entry is already set, convert it to an array
            _ = mat_X[l]
            mat_X[l] = np.array([_,n])
        else:
            print('error')

    for i,v in enumerate(mat_X): # Go through again and change all unset entries (entry==index) to np.NaN
        if type(v) == np.ndarray:
            continue
        elif v == i:
            mat_X[i] = np.nan

    return mat_X





# Example use
cat_L = SkyCoord(ra=RA_L*u.degree, dec=DEC_L*u.degree)
cat_T2 = SkyCoord(ra=RA_T2*u.degree, dec=DEC_T2*u.degree)

idLT2_L, idLT2_T2,d2d,d3d = search_around_sky(cat_L,cat_T2,205*u.arcsecond)

accLT2_L, accLT2_T2 = cross_match(idLT2_L, idLT2_T2, RA_L,DEC_L,MAJ_L,MIN_L,PA_L, RA_T2,DEC_T2,MAJ_T2,MIN_T2,PA_T2,0.5)

mat_T2 = cross_id(accLT2_L, accLT2_T2)
