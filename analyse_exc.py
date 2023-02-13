import numpy
import pandas
import matplotlib
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
import os.path
import matplotlib.gridspec as gridspec
from scipy.interpolate import LinearNDInterpolator
from scipy.interpolate import NearestNDInterpolator

eV=1/0.0367493
angstrom=0.529177

#TO DO: treat correctly sum over degenerate k-vectors

def main():
    #INPUT#
    nb=0 #define an exciton 1
    nd=0 #define an exciton 2
    pathBSdbs='../bands.out.dbs'
    pathBSagr='../bands.out.agr'
    pathEXC= 'analysisexc_q1'
    nbnd=80  # define number of bands for band structure
    nkpt=71 # define number of k-pointr for band structure
    nval=55  # max valence band 
    lbnd=30   # lowest band to print
    ntrans=2000 # in analysisexc_ file per exciton
    nexc=30     # number of excitons in analysisexc_ file
    scisor=0 #2.64 #scisor in eV for a gap 
    BZ=numpy.asarray([[0.0459674, 0.0000000,   0.000],
                      [0.0000,  0.1484785, 0.000],
                      [0.0000,   0.0000, 0.1211486]])*2*numpy.pi # define a BZ
    
   #END OF INPUT#
   #EXC=pandas.read_csv('analysisexc_q1',header=None,comment='#',skiprows=lambda x: x not in numpy.arange(1,31000,1003),delim_whitespace=True)
    
    EXCanalys=pandas.read_csv(pathEXC,header=None,comment='#',skiprows=numpy.arange(1,(ntrans+3)*nexc,ntrans+3),delim_whitespace=True) #shape(1,31000 (maximum number of rows), number of lines to go over = number of transitions+3)    
    EXCanalys=numpy.reshape(EXCanalys.values,(nexc,ntrans,18)) # (shape =number exciton, number of transitions, number of columns in the q1 file)
    
    A=EXCanalys[:,:,7]+1j*EXCanalys[:,:,8]
    rho=EXCanalys[:,:,9]+1j*EXCanalys[:,:,10]
     
    #to plot a BS with the transition's amplitudes  
    for i in range(1,2):
        plotBS(EXCanalys,A,rho,[nd,nb],lbnd,nbnd,pathBSdbs,pathBSagr,nkpt,scisor,nval,BZ,plot=i)
    #to plot a cumulative sum of the transition's amplitudes
    #plotS(EXCanalys,A,rho,631,nd) 

def loadBS(pathBSdbs,pathBSagr,nkpt,nbnd):
    'load DFT band structure'
    kpt=numpy.zeros((nkpt,3))
    bnds=numpy.zeros((nkpt,nbnd))
    fp=open(pathBSdbs)
    j=0
    jj=0
    ii=0
    kk=0
    a=False
    b=False
    c=False
    spcl_k=[]
    for i, line in enumerate(fp):
        "load bands and k-vectors"
        if line[:8]=='DATABASE':
            a=True
        if line[:12]=='DATABASE KEY':
            a=False
        if a:
            if line[:3]=='kpt':
                ii=i
            if i==ii+1:
                kpt[j]=numpy.fromstring(line,sep=' ')
                j+=1
            if i==ii+2:
                bnds[jj]=numpy.fromstring(line,sep=' ')
                jj+=1

        "load special points"
        "TO DO: need to work with dictionary to plot the path"
        if line[:21]=='BAND STRUCTURE SCHEME':
            b=False
        if b:
            if len(line)>1:
                spcl_k.append(numpy.fromstring(line[5:],sep=' '))
        if line[:16]=='SPECIAL K-POINTS':
            b=True
        "load special points labels"
        if kk==1:
            spcl_lbls=line[2::5]
            kk+=1
        if line[:21]=='BAND STRUCTURE SCHEME':
            kk+=1

    #kpt=(numpy.insert(kpt,40,kpt[0],axis=0))
    #bnds=(numpy.insert(bnds,40,bnds[0],axis=0))

    spcl_k = numpy.asarray(spcl_k)
    #spcl_k = numpy.insert(spcl_k,3,spcl_k[0],axis=0)
    spcl_ind = numpy.zeros(spcl_k.shape[0],dtype='int')
    "find special k-vec indices" 
    "need to also load the name of path"
    for i in range(spcl_k.shape[0]):
        spcl_ind[i]=(spcl_k[i]==kpt).all(-1).argmax()

    spcl_ind[4]=40 
    kptx=numpy.loadtxt(pathBSagr,skiprows=59,max_rows=nkpt)[:,0]
    
    return kpt,bnds,kptx,spcl_ind,spcl_lbls

def getS(EXCanalys,A,rho,n):
    ind=EXCanalys[n,:,6].argsort()
    S=numpy.zeros(1000)
    for i in range(1000):
        S[i]=numpy.abs(numpy.sum(A[n,ind[:i]]*rho[n,ind[:i]]))
    return ind,S

def mapAtoBS(EXCanalys,A,rho,kpt,n,BZ):
    #EXCanalys_unique[0] - The sorted unique values
    #EXCanalys_unique[1] - The indices of the first occurrences of the unique values in the original array
    #EXCanalys_unique[2] - The indices to reconstruct the original array from the unique array
    EXCanalys_unique=numpy.unique(EXCanalys[n,:,6],axis=0,return_inverse=True,return_index=True)
    S_Arho=numpy.zeros(EXCanalys_unique[1].shape[0],dtype=complex)
    S_A   =numpy.zeros(EXCanalys_unique[1].shape[0],dtype=complex)
    S_rho =numpy.zeros(EXCanalys_unique[1].shape[0],dtype=complex)
    
    for i,ii in enumerate(EXCanalys_unique[1]):
        "ind - getting the indeces of unique values in EXC"
        ind=numpy.where(EXCanalys_unique[2]==i)[0]


        S_Arho[i]=numpy.sum(A[n,ind]*rho[n,ind])
        S_A[i]=numpy.sum(A[n,ind])
        S_rho[i]=numpy.sum(rho[n,ind])
    return S_Arho,S_A,S_rho,EXCanalys_unique

def interpolateK(data,kvec_exc,kvec_BS):
    my_interpolating_function = LinearNDInterpolator(kvec_exc, data,fill_value=0)
    return my_interpolating_function(kvec_BS-numpy.round(kvec_BS))

def interp_unique_bands(data,band_ind,kvec_exc,kvec_BS):
    #data - data to interpolate
    #band_ind - c and v band indeces in EXC
    #kvec_exc - kvec in EXC
    #kvec_BS - kvec in band structure
    inter_data_cv=[]
    #for j in range(2): #over c and v
    ind_unique=numpy.unique(band_ind[:,:],axis=0,return_inverse=True,return_index=True)
    inter_data=numpy.zeros((ind_unique[1].shape[0],kvec_BS.shape[0],3))
    for  i,ii in enumerate(ind_unique[1]):
        ind=numpy.where(ind_unique[2]==i)[0]
        try:
            inter_data[i,:,0]=interpolateK(data[ind],kvec_exc[ind],kvec_BS)
        except:
            inter_data[i,:,0]=0
        inter_data[i,:,1]=band_ind[ii,0]
        inter_data[i,:,2]=band_ind[ii,1]
    #inter_data_cv.append(inter_data)
    return inter_data


def plotBS(EXCanalys,A,rho,nexc,lbnd,nbnd,pathBSdbs,pathBSagr,nkpt,scisor,nval,BZ,plot):
    kpt,bnds,kptx,spcl_k,spcl_lbls = loadBS(pathBSdbs,pathBSagr,nkpt,nbnd)
    Sb_Arho,Sb_A,Sb_rho,EXCanalysb_unique=mapAtoBS(EXCanalys,A,rho,kpt,nexc[0],BZ)
    Sd_Arho,Sd_A,Sd_rho,EXCanalysd_unique=mapAtoBS(EXCanalys,A,rho,kpt,nexc[1],BZ)
    

    plt.figure(figsize=(18, 16))

    plt.xticks(fontsize=34,fontweight='bold')
    plt.yticks(fontsize=34,fontweight='bold')
    plt.ylabel('$\epsilon$, (eV)', fontsize=50)
    print(kptx[spcl_k])
    print(spcl_lbls)
    plt.xticks(kptx[spcl_k],spcl_lbls) # define your BZ path
    plt.grid(axis='x')

    #if plot==0: # Vitaly: sum over degenerate k-vector doesn't work for a moment
    #    Sb=numpy.absolute(Sb_Arho)
    #    Sd=numpy.absolute(Sd_Arho)
    #    plt.text(0.25,2,'$|\\sum_{\\mathbf{k}=equiv}A_{\\lambda}^{vc,\\mathbf{k}}\\tilde{\\rho}^{vc,\\mathbf{k}}|$',fontsize=40)
    Sb=[]
    if plot==1: #plot |Al*rho|
        Sb.append(numpy.abs(A[nexc[0]]*rho[nexc[0]])*1e2)#[EXCanalysb_unique[1]]
        Sb.append(numpy.abs(A[nexc[1]]*rho[nexc[1]])*1e2)#[EXCanalysd_unique[1]]
        plt.text(0.35,2,'$|A_{\\lambda}^{vc,\\mathbf{k}}\\tilde{\\rho}^{vc,\\mathbf{k}}|$',fontsize=40)
    elif plot==2: #plot |Al|
        Sb.append((A[nexc[0]]).real*1e3)#[EXCanalysb_unique[1]]
        Sb.append((A[nexc[1]]).real*1e3)#[EXCanalysd_unique[1]]
        plt.text(0.35,2,'$|A_{\\lambda}^{vc,\\mathbf{k}}|$',fontsize=40)#\\tilde{\\rho}^{vc,\\mathbf{k}}|$',fontsize=50)
    elif plot==3: #plot |rho|
        Sb.append((rho[nexc[0]]).real*1e1)#[EXCanalysb_unique[1]]
        Sb.append((rho[nexc[1]]).real*1e1)#[EXCanalysd_unique[1]]
        plt.text(0.35,2,'$|\\tilde{\\rho}^{vc,\\mathbf{k}}|$',fontsize=40)#\\tilde{\\rho}^{vc,\\mathbf{k}}|$',fontsize=50)



    #plt.scatter(kptx[mappingEXCb[14]],bnds[mappingEXCb[14],int(EXCanalysb_unique[0][14,0]-1)]-bnds[-1,nval],s=Sb[14]*1e5,c='C1',alpha=0.35)
    #plt.scatter(kptx[mappingEXCd[3]],bnds[mappingEXCd[3],int(EXCanalysd_unique[0][3,0]-1)]-bnds[-1,nval],s=Sd[3]*1e5,c='C0',alpha=0.35)
    #plt.legend([
    #            'bright 3.36 eV',
    #            'dark 2.77 eV',
    #           ],loc='center left',fontsize=30)
    
    cols=['C0','C1','C0']    
    "interpolating k-vectors"
    for i,ii in enumerate(nexc):
        print(i)
        "translating k-vectors to reduced coordinates"
        kvec_exc=numpy.zeros((EXCanalys.shape[1],3))
        for j in range(EXCanalys.shape[1]):
            kvec_exc[j]=numpy.around(numpy.dot(numpy.linalg.inv(BZ.T),EXCanalys[ii,j,15:18]),decimals=3)
        
        interpol_bands=interp_unique_bands(Sb[i],EXCanalys[ii,:,4:6],kvec_exc,kpt[])
        numpy.save('interpol_bands_exc%d' % (ii+1),interpol_bands)
        for j in range(interpol_bands.shape[0]):
            colors=[]
            for jj,jjj in enumerate((numpy.sign(interpol_bands[j,:,0]))):
                colors.append(cols[int(jjj)])

            "plotting only valence bands"
        #for j,ii in enumerate(interpol_bands[i,:,0]):
            if i==0: 
            #    col='C0'
                alpha=0.35
                marker='o'
            else: 
            #    col='C1'
                alpha=0.0
                marker='o'
            plt.scatter(kptx[:],bnds[:,int(interpol_bands[j,0,1]-1)]-bnds[-1,nval],s=abs(interpol_bands[j,:,0]),c=colors,alpha=alpha,marker=marker)
            "plotting only conductionbands"
            plt.scatter(kptx[:],bnds[:,int(interpol_bands[j,0,2]-1)]-bnds[-1,nval]+scisor,s=abs(interpol_bands[j,:,0]),c=colors,alpha=alpha,marker=marker)
            
            
   #cols[int(numpy.sign(interpol_bands[j,:,0])+1)] 
    for i in range(lbnd,nval+1):
        plt.plot(kptx,bnds[:,i]-bnds[-1,nval],'-',color='red',linewidth=1)
    for i in range(nval+1,nbnd): # one can use a scisor to plot the band structure
        plt.plot(kptx,bnds[:,i]-bnds[-1,nval]+scisor,'-',color='red',linewidth=1)




    plt.plot([0,1],[0,0],'--',color='gray')

    #plt.xlim(0,1)
    plt.ylim(-3.8,7)
    plt.tight_layout()
    plt.show()
    #plt.savefig('BS_sumArhotw.pdf',dpi=300)

def plotS(EXCanalys,A,rho,nb,nd):
    'plot a cummulative sum'
    plt.figure(figsize=(10, 8))
    plt.xticks(fontsize=26,fontweight='bold')
    plt.yticks(fontsize=26,fontweight='bold')
    plt.xlabel('$\\omega$, (eV)', fontsize=50)
    plt.ylabel('$S_{\\lambda}$', fontsize=50)

    col=['black','C1']
    ii=0
    for i in [nd,nb]:
        plt.plot(EXCanalys[i,getS(EXCanalys,A,rho,i)[0],6],getS(EXCanalys,A,rho,i)[1],'-',color=col[ii])
        ii+=1
    plt.legend([' dark',
                'bright',
               ],loc='lower right',fontsize=22)
    plt.tight_layout()
    plt.show()
    #plt.savefig('EXC_Sl.pdf',dpi=300)


if __name__ == "__main__":
    main()
