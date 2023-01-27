import sys
import getopt
from mpmath import *
import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import eigh_tridiagonal

# Usage: python3.10 main_higherprecision.py --nmax==25 --prec=1000 filename

def stieltjes(x,pbd,nmax):

    nmin = 5
    qovmin = mpf(10e-50)
    overmax = mpf(1.0)

#initiate the recursive computation of the a,b coefficients and the orthogonal
#polynomials according to (3.3.20-23) of Mueller-Plathe & Dierksen (1990)
    acoef = []
    bcoef = []
    qpol = []

    b = mpf(0.0)
    a = mpf(0.0)
    q = []
    acoef.append(mpf(0.0))

    for i in range(len(x)):
        b+=pbd[i]
        a+=pbd[i]/x[i]
        q.append(mpf(1.0))

    bcoef.append(b)
    acoef.append(a / bcoef[0])
    qpol.append(q)

    b = mpf(0.0)
    a = mpf(0.0)
    q = []
    for i in range(len(x)):
        q.append(1/x[i]-acoef[1])
    qpol.append(q)

    for i in range(len(x)):
        b+=qpol[1][i]*pbd[i]/x[i]
        a+=qpol[1][i]*pbd[i]/(x[i]**2)
    bcoef.append(b/bcoef[0])
    acoef.append(a/(bcoef[0]*bcoef[1])-acoef[1])

#calculate the higher-order coefficients and polynomials recursively
#up to the (nmax-1)th order (total of nmax polynomials)

    asum = acoef[1]

    for i in range(3,nmax):

        asum += acoef[i-1]
        q = []
        for j in range(len(x)):
            q.append( ( mpf(1.0)/x[j] - acoef[i-1])*qpol[i-2][j]-bcoef[i-2]*qpol[i-3][j])
        qpol.append(q)

        bprod = bcoef[0]
        for j in range(1,i-1):
            bprod *= bcoef[j]

        b = mpf(0.0)
        for j in range(len(x)):
            b+=qpol[i-1][j]*pbd[j]/(x[j]**(i-1))
        bcoef.append(b/bprod)

        bprod *= bcoef[i-1]
        a = mpf(0.0)
        for j in range(len(x)):
            a+=qpol[i - 1][j]*pbd[j]/(x[j]**i)
        acoef.append(a/ bprod - asum)

#calculate the nmax-th order polynomial just for the orthogonality check
    q = []
    for j in range(len(x)):
        q = []
        q.append((1.0 / x[j] - acoef[nmax-1]) * qpol[nmax-2][j] - bcoef[nmax - 2] * qpol[nmax-3][j])
    qpol.append(q)

#check the orthogonality of the polynomials to define the maximal approximation order
#if the orthogonality is preserved for all orders, maxord is set to nmax

    maxord = nmax
    for i in range(1,nmax-1):
        qnorm = mpf(0.0)
        qoverlap = mpf(0.0)
        for j in range(len(x)):
            qnorm += qpol[i][j]**2*pbd[j]
            qoverlap += qpol[i][j]*qpol[i-1][j]*pbd[j]
        if(fabs(qoverlap)<qovmin):
            qoverlap = qovmin
        if(qnorm/fabs(qoverlap)<overmax):
            maxord=i-1
            break

#look how many Stieltjes orders are available
    if maxord < 5:
        print("*** Warning ***")
        print("only very low-order approximation is available")
        sys.exit()
    print("MaxOrder=",maxord)

#perform the gamma calculation using the successive approximations
# n=5,...,nmax

    min=5
    max=maxord

    xorder = np.zeros((max - 5, max - 5), dtype=np.longdouble)
    pbdorder = np.zeros((max - 5, max - 5), dtype=np.longdouble)
    for iord in range(min,max):
        print("Performs Stieltjes at order",iord)
        diag = np.zeros(iord,dtype=np.longdouble)
        diag[:iord] = acoef[1:iord+1]

        offdiag = np.zeros(iord-1,dtype=np.longdouble)
        offdiag[:] = -np.sqrt(bcoef[1:iord])

        w, v = eigh_tridiagonal(diag, offdiag, eigvals_only=False, lapack_driver='stebz')
        xnew = 1.0/w
        pbdnew = bcoef[0]*np.power(v[0,:],2)

#calculate the gamma's by simple numerical differentiation at the middle
# point of each [XNEW(I),ENEW(I+1)] interval
        for i in range(1,iord-min+2):
             xorder[iord-min,i-1] = 0.5*(xnew[i]+xnew[i+1])
             pbdorder[iord-min,i-1] = 0.5*(pbdnew[i]+pbdnew[i+1])/(xnew[i]-xnew[i+1])
    return xorder, pbdorder

# Press the green button in the gutter to run the script.
if __name__ == '__main__':

# Stieltjes Order by default
    nmax = 25
# Precision by default
    mp.dps = 1000

    print('ARGV      :', sys.argv[1:])
    options, remainder = getopt.getopt(sys.argv[1:], ':',   ['nmax=', 
                                                             'prec=',
                                                            ])

    print('OPTIONS   :', options)
    print('FILENAME  :', remainder[0])
    fname = remainder[0]

    for opt, arg in options:
        if opt == '--nmax':
           nmax = int(arg)
        elif opt == '--prec':
           prec = int(arg)
           mp.dps = prec

    print('PRECISION  :', prec)
    dat = np.loadtxt(open(fname,"r"))
    x = dat[:,0]
    xshift=np.min(x)-0.2
    x-=xshift
    pbd = np.power(dat[:,1],1)
    sort=np.argsort(x)
    x=x[sort]
    pbd=pbd[sort]

    xmp = []
    pbdmp = []
    for i in range(len(x)):
       xmp.append(mpf(x[i]))
       pbdmp.append(mpf(pbd[i]))

    print("")
    xst, pbdst = stieltjes(xmp,pbdmp,nmax)
    print("")
    for i in range(0,len(xst)):
        xst[i] += xshift
        xi = xst[i][0:i+1].astype(np.double)
        pbdi = pbdst[i][0:i+1].astype(np.double)
        sort = np.argsort(xi)
        xi = xi[sort]
        pbdi = pbdi[sort]
        print("Order= ",i," G= ",np.interp(0.0, xi, pbdi)*2.0*np.pi*27211," meV")
