#!/usr/bin/python3
import re
import pickle
import scipy
import numpy as np
from scipy.sparse import csr_matrix
import sklearn.manifold
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d import proj3d

fn = "usersha1-artmbid-artname-plays.tsv"
dissimilarityMatrixFilename = 'dMat.p'

verbose = True

dim3d = False

def buildDataStructures(filename):
    """ Build Data Strctures is specifically designed to parse the dataset lastfm-dataset-360K
        collected by Oscar Celma (oscar.celma@upf.edu) provided by Last.fm
        it returns a tuple containing:
        1.A sparse int matrix Artist x User containing how often a user listned to a song by artist
        2.A Map mapping a user's hashed id to his matrix index
        3.A map mapping the artistname :: String to the matrix index :: Int
        4.An array Containing the Artistnames :: String in the order of the matrix indices
    """
    patt = re.compile("([0-9A-Fa-f-]*)\s*\t\s*([0-9A-Fa-f-]*)\s*\t\s*([^\t]*)\t\s*(\d*)")
    interpretMap = {}
    userCounter = 0
    lastUserId = None
    userIdToIndex = {}
    with open(filename,'r') as f:
        for line in f:
            match = patt.match(line)
            try:
                (userId,artistId,artistName, listenCounter) = match.groups()
                if userId != lastUserId:
                    userIdToIndex[userId] = userCounter
                    userCounter+=1
                    lastUserId = userId
            except AttributeError:
                if verbose:
                    print("the line :\n%s\n could not be parsed" % line)
                continue
            if artistName in interpretMap:
                interpretMap[artistName].append((userId,int(listenCounter)))
            else:
                interpretMap[artistName] = [(userId,int(listenCounter))]
    n = len(interpretMap)
    
    if verbose:
        print("the interprets dict now contains %d elements" % n)

    for key in list(interpretMap):
        if len(interpretMap[key]) < 200:
            del interpretMap[key]
    m = len(interpretMap)
    if verbose:
        print("%d elements have been removed for a total of %d elements" % (n-m,m))
    
    artistNameToIndex = {key : value for (key, value) in zip(list(interpretMap), range(m))}
    indexToArtistName = np.array(list(interpretMap))
    row,col,data = ([],[],[])
    for key in interpretMap.keys():
        for (userId, listens) in interpretMap[key]:
            col.append( artistNameToIndex[key] )
            row.append( userIdToIndex[userId] )
            data.append( listens )
    
    listensMatrix = csr_matrix((data,(col,row)),shape = (m, userCounter), dtype=np.int64)
    return (listensMatrix, userIdToIndex, artistNameToIndex, indexToArtistName)


def genDisMat(simMat,eps=1e-7):
    """ generates a dissimilarity Matrix based on the expectation value of the
    conditional propability of two artists occuring in the same users playlist"""
    dMat = 1/(normalizeMat(simMat)+eps)
    return np.log(dMat)

def normalizeMat(m):
    mc = np.array(m,dtype=np.float32)
    for i in range(mc.shape[0]):
        x = (scipy.sqrt(mc[i,i]))
        mc[i,:]/=x
        mc[:,i]/=x
    return mc


def findClosest(dm,AtI,itA,bands,n):
    """given a dissimilarity matrix between artists, a map from artist to index,
    an array from index to artist, a list of artists and a integer n < the total number of artists,
    it return n artists considered most similiar to the given list of artists
    """
    bandIndices = [AtI[key] for key in bands]
    keys = np.argsort(dm[bandIndices,:].sum(axis=0))
    pred = np.squeeze(np.array(itA[keys]))
    return pred[:n]

def init():
    """ loads the datastructures, if already existing. Parses the original dataset otherwise. """
    try:
        with open(dissimilarityMatrixFilename ,'rb') as f:
            if verbose: print('trying to read in artist information')
            (dM, itA, AtI) =pickle.load(f)
    except FileNotFoundError:
        if verbose: print('could not find %s\n falling back to generating it from the dataset' % dissimilarityMatrixFilename) 
        _, _, AtI, itA = buildDataStructures(fn)
        simMat = lmat.dot(lmat.transpose()).todense()
        dM = genDisMat(simMat)
        with open('dMat.p','wb') as f:
            pickle.dump((dM,itA,AtI),f)
    return dM, itA, AtI


def main():
    """takes a list of artists and recommends a number of simmiliar artists.
    then uses multidimensional scaling to embed the artists according to their simmilarity in a diagram.
    """
    numberRecommendations = 400
    #bands = ['50 cent','kollegah','k.i.z.','sido','fatboy slim', 'eminem', '2pac', 'kool savas']
    bands = ['metallica','rage against the machine','wizo', 'skindred','caliban','die Ärzte','muse', 'dropkick murphys', 'franz ferdinand',
            'die toten hosen','in flames', 'dark tranquillity', 'equilibrium','rammstein']

    dM, itA, AtI = init()
    bands = findClosest(dM, AtI, itA, bands, numberRecommendations)
    if verbose: print('recommended %d artists. generating plot.' % numberRecommendations)
    #bands =AtI.keys()
    bandIndices=[index for index in list(map(lambda x: AtI[x],bands))]
    dM = dM[np.ix_(bandIndices,bandIndices)] ##shrink to indices
    if dim3d:
        mds = sklearn.manifold.MDS(n_components=3, dissimilarity='precomputed')
    else:
        mds = sklearn.manifold.MDS(n_components=2, dissimilarity='precomputed')
    p = mds.fit_transform(dM)
    if dim3d:
        annotatedScatter3d(bands,p[:,0],p[:,1],p[:,2])
    else:
        fig, ax = plt.subplots(figsize=(40,40))
        ax.scatter(p[:,0],p[:,1])
        for i, txt in enumerate(bands):
            ax.annotate(txt, (p[i,0],p[i,1]+0.05), fontsize=12, ha="center")
    fig.savefig('plot.pdf',format='pdf')


    #plt.show()

def annotatedScatter3d(labels,xs,ys,zs):
    """3d scatter plot, sadly rotating breaks the association of dots and annotations"""
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    sc = ax.scatter(xs,ys,zs)
    for txt, x, y, z in zip(labels, xs, ys, zs):
        x2, y2, _ = proj3d.proj_transform(x,y,z, ax.get_proj())
        label = plt.annotate(txt, xy = (x2, y2))

if __name__== "__main__":
    main()
