# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import sys
import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import eigh_tridiagonal

class ProbaDistrib:
    def __init__(self,x,pbd):
        self.x = x
        self.pbd = pbd

    def __len__(self):
        return len(pbd)

    def print_pbd(self):
        for i in range(len(self)):
            print(self.x[i],self.pbd[i])

    def moment(self,n):
        return np.sum(np.power(x,n)*pbd)

    def entropy(self):
        return -np.sum(np.log(pbd)*pbd)

    def plot_pbd(self):
        plt.plot(self.x,self.pbd)
        plt.show()

    def stieltjes(self):

# Stieltjes Order
        nmin = 11
        nmax = 33
        qovmin = np.longdouble(10e-50)
        overmax = np.longdouble(1.0)

#initiate the recursive computation of the a,b coefficients and the orthogonal
#polynomials according to (3.3.20-23) of Mueller-Plathe & Dierksen (1990)
        acoef = np.zeros(nmax + 1, dtype=np.longdouble)
        bcoef = np.zeros(nmax + 1, dtype=np.longdouble)
        qpol = np.zeros((nmax + 1,len(self)), dtype=np.longdouble)

        bcoef[0] = np.sum(self.pbd)
        acoef[1] = np.sum(self.pbd/self.x)/bcoef[0]
        qpol[0][:] = 1.0
        qpol[1][:] = 1.0/self.x-acoef[1]

        bcoef[1] = np.sum(qpol[1][:]*self.pbd/self.x)/bcoef[0]
        acoef[2] = np.sum(qpol[1][:]*self.pbd/np.power(self.x,2))/(bcoef[0]*bcoef[1])-acoef[1]

#calculate the higher-order coefficients and polynomials recursively
#up to the (nmax-1)th order (total of nmax polynomials)

        asum = np.copy(acoef[1])
        for i in range(3,nmax):

            asum += acoef[i-1]
            qpol[i-1][:] = (1.0/self.x - acoef[i-1])*qpol[i-2][:]-bcoef[i-2]*qpol[i-3][:]

            bprod = bcoef[0]
            for j in range(1,i-1):
                bprod *= bcoef[j]

            bcoef[i-1] = np.sum(qpol[i-1][:]*self.pbd/np.power(self.x,i-1))/bprod

            bprod *= bcoef[i-1]
            acoef[i] = np.sum(qpol[i - 1][:] * self.pbd / np.power(self.x, i)) / bprod - asum

#calculate the nmax-th order polynomial just for the orthogonality check
        qpol[nmax][:] = (1.0 / self.x - acoef[nmax]) * qpol[nmax-1][:] - bcoef[nmax - 1] * qpol[nmax-2][:]

#check the orthogonality of the polynomials to define the maximal approximation order
#if the orthogonality is preserved for all orders, maxord is set to nmax

        maxord = nmax
        for i in range(1,nmax):
            qnorm = np.sum(np.power(qpol[i][:],2)*self.pbd)
            qoverlap = np.sum(qpol[i][:]*qpol[i-1][:]*self.pbd)
            if(np.abs(qoverlap)<qovmin):
                qoverlap = qovmin
            if(qnorm/np.abs(qoverlap)<overmax):
                maxord=i-1
                break
            if (not(len([x for x in bcoef[0:i] if x<0])==0)):
                maxord = i - 1
                break
#look how many Stieltjes orders are available
        if maxord < 5:
            print("*** Warning ***")
            print("only very low-order approximation is available")
            sys.exit()
#perform the gamma calculation using the successive approximations
# n=5,...,nmax

        min=5
        max=maxord
        xorder = np.zeros((max-5,max-5),dtype=np.longdouble)
        pbdorder = np.zeros((max-5, max-5), dtype=np.longdouble)

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
#                 print(i,iord-min,0.5*(xnew[i]+xnew[i+1]))
                 xorder[iord-min,i-1] = 0.5*(xnew[i]+xnew[i+1])
                 pbdorder[iord-min,i-1] = 0.5*(pbdnew[i]+pbdnew[i+1])/(xnew[i]-xnew[i+1])
        return xorder, pbdorder
# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    dat = np.loadtxt(open("fanotot0.txt","r"))
    x = dat[:,0]
    xshift=-np.min(x)*1.01
    x-=xshift
    pbd = np.power(dat[:,1],2)
    sort=np.argsort(x)
    x=x[sort]
    pbd=pbd[sort]
    pbd1= ProbaDistrib(x,pbd)
    # print(len(pbd1))
    # pbd1.print_pbd()
#    print(pbd1.moment(2))
#    print(pbd1.entropy())
#    pbd1.plot_pbd()
    xst, pbdst = pbd1.stieltjes()
    for i in range(0,len(xst)):
        xst[i] += xshift
        pbd_new = ProbaDistrib(xst[i][0:i+1],pbdst[i][0:i+1])
#        print(xst[i][0:i+1])
        xi = xst[i][0:i+1].astype(np.double)
        pbdi = pbdst[i][0:i+1].astype(np.double)
        sort = np.argsort(xi)
        xi = xi[sort]
        pbdi = pbdi[sort]
#        print(i)
#         pbd_new.print_pbd()
#         print("")
# #        print(i,pbd_new.moment(2))
# #        print(i,pbd_new.entropy())
        print(i,np.interp(0.0, xi, pbdi)*2.0*np.pi*27211)
# See PyCharm help at https://www.jetbrains.com/help/pycharm/
