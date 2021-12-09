
import copy
import math

import matplotlib
# matplotlib.use('Agg')

import matplotlib
from matplotlib import mlab
from matplotlib import cm
from matplotlib.colors import LinearSegmentedColormap
import numpy
import numpy as np

from rdkit import Chem
from rdkit import DataStructs
from rdkit.Chem import Draw
from rdkit.Chem import rdMolDescriptors as rdMD
from rdkit.six import iteritems

from rdkit.Chem.Draw import SimilarityMaps

from rdkit.Chem.MACCSkeys import maccsKeys,smartsPatts,_InitKeys






def _pyGenMACCSKeys(mol,atomId=-1,**kwargs):
  """ generates the MACCS fingerprint for a molecules
   **Arguments**
     - mol: the molecule to be fingerprinted
     - any extra keyword arguments are ignored

   **Returns**
      a _DataStructs.SparseBitVect_ containing the fingerprint.
  >>> m = Chem.MolFromSmiles('CNO')
  >>> bv = GenMACCSKeys(m)
  >>> tuple(bv.GetOnBits())
  (24, 68, 69, 71, 93, 94, 102, 124, 131, 139, 151, 158, 160, 161, 164)
  >>> bv = GenMACCSKeys(Chem.MolFromSmiles('CCC'))
  >>> tuple(bv.GetOnBits())
  (74, 114, 149, 155, 160)
  """
  global maccsKeys
  if maccsKeys is None:
    maccsKeys = [(None, 0)] * len(smartsPatts.keys())
    _InitKeys(maccsKeys, smartsPatts)
  ctor = kwargs.get('ctor', DataStructs.SparseBitVect)

  res = ctor(len(maccsKeys) + 1)
  for i, (patt, count) in enumerate(maccsKeys):
    if patt is not None:
      matches = mol.GetSubstructMatches(patt)
      try:
          matches = np.stack(matches)
      except:
          matches = matches

      if count == 0:
        if len(matches) > 0:
            if atomId in matches:
              res[i + 1] = 0
            else:
              res[i + 1] = 1
      else:
        if len(matches) > count:
          if atomId in matches:
              res[i + 1] = 0
          else:
              res[i + 1] = 1

    elif (i + 1) == 125:
      # special case: num aromatic rings > 1
      ri = mol.GetRingInfo()
      nArom = 0
      res[125] = 0
      for ring in ri.BondRings():
        isArom = True
        for bondIdx in ring:
          if not mol.GetBondWithIdx(bondIdx).GetIsAromatic():
            isArom = False
            break
        if isArom:
          nArom += 1
          if nArom > 1:
            res[125] = 1
            break
    elif (i + 1) == 166:
      res[166] = 0
      # special case: num frags > 1
      if len(Chem.GetMolFrags(mol)) > 1:
        res[166] = 1

  return res

def GetAtomicWeightsForFingerprint(refMol, probeMol, fpFunction, metric=DataStructs.DiceSimilarity):
  """
  Calculates the atomic weights for the probe molecule
  based on a fingerprint function and a metric.

  Parameters:
    refMol -- the reference molecule
    probeMol -- the probe molecule
    fpFunction -- the fingerprint function
    metric -- the similarity metric

  Note:
    If fpFunction needs additional parameters, use a lambda construct
  """
  if hasattr(probeMol, '_fpInfo'):
    delattr(probeMol, '_fpInfo')
  if hasattr(refMol, '_fpInfo'):
    delattr(refMol, '_fpInfo')
  refFP = fpFunction(refMol, -1)
  probeFP = fpFunction(probeMol, -1)
  baseSimilarity = metric(refFP, probeFP)
  # loop over atoms
  weights = []
  for atomId in range(probeMol.GetNumAtoms()):
    newFP = fpFunction(probeMol, atomId)
    newSimilarity = metric(refFP, newFP)
    weights.append(baseSimilarity - newSimilarity)
  if hasattr(probeMol, '_fpInfo'):
    delattr(probeMol, '_fpInfo')
  if hasattr(refMol, '_fpInfo'):
    delattr(refMol, '_fpInfo')
  return weights


def GetAtomicWeightsForModel(probeMol, fpFunction, predictionFunction):
  """
  Calculates the atomic weights for the probe molecule based on
  a fingerprint function and the prediction function of a ML model.

  Parameters:
    probeMol -- the probe molecule
    fpFunction -- the fingerprint function
    predictionFunction -- the prediction function of the ML model
  """
  if hasattr(probeMol, '_fpInfo'):
    delattr(probeMol, '_fpInfo')
  probeFP = fpFunction(probeMol, -1)
  baseProba = predictionFunction(probeFP)
  # loop over atoms
  weights = []
  for atomId in range(probeMol.GetNumAtoms()):
    newFP = fpFunction(probeMol, atomId)
    newProba = predictionFunction(newFP)
    weights.append(baseProba - newProba)
  if hasattr(probeMol, '_fpInfo'):
    delattr(probeMol, '_fpInfo')
  return weights



def calcAtomGaussians(mol,a=0.03,step=0.02,weights=None):

    x = np.arange(-1.5,2.5,step)
    y = np.arange(-1.5,2.5,step)

    X,Y = np.meshgrid(x,y)
    if weights is None:
        weights=[1.]*mol.GetNumAtoms()

    Z = np.zeros_like(X)
    for i in range(mol.GetNumAtoms()):
        Zp = matplotlib.mlab.bivariate_normal(X,Y,a,a,
                mol._atomPs[i][0], mol._atomPs[i][1])
        Zp = Zp/np.abs(Zp).max()
        Z += Zp*weights[i]

    try:
        lines,columns = np.argwhere(np.abs(Z)>0.005).T
        X = X[lines.min():lines.max(),columns.min():columns.max()]
        Y = Y[lines.min():lines.max(),columns.min():columns.max()]
        Z = Z[lines.min():lines.max(),columns.min():columns.max()]
    except:
        # None
        xmin,ymin = np.array(list(mol._atomPs.values())).min(0)
        xmax,ymax = np.array(list(mol._atomPs.values())).max(0)

        condx = (x>xmin)&(x<xmax)
        condy = (y>ymin)&(y<ymax)

        X = X[condx,:][:,condy]
        Y = Y[condx,:][:,condy]
        Z = Z[condx,:][:,condy]


    return X,Y,Z



def MolecularScalarMap(mol,weights,sigma=None,contour=False,coordScale=1.5,
            scale=None,nlevels=20,contourLines=10,colors='k',mplargs=dict(),
            **kwargs):

    if mol.GetNumAtoms() < 2:
        raise ValueError("too few atoms")

    fig = Draw.MolToMPL(mol, coordScale=coordScale, size=(250,250),**mplargs)
    ax = fig.axes[0]

    if sigma is None:
        if mol.GetNumBonds() > 0:
            bond = mol.GetBondWithIdx(0)
            idx1 = bond.GetBeginAtomIdx()
            idx2 = bond.GetEndAtomIdx()
            sigma = 0.3 * math.sqrt(
            sum([(mol._atomPs[idx1][i] - mol._atomPs[idx2][i])**2 for i in range(2)]))
        else:
            sigma = 0.3 * math.sqrt(sum([(mol._atomPs[0][i] - mol._atomPs[1][i])**2 for i in range(2)]))
    sigma = round(sigma, 2)

    x, y, z = calcAtomGaussians(mol, sigma, weights=weights)

    print([x.min(),x.max(),y.min(),y.max()])

    if scale==None:
        vmax = np.max(np.abs(z))
    else:
        vmax = scale

    levels = np.linspace(-vmax,vmax,nlevels)

    mappable=[]
    if np.any(z):
        cntf = ax.contourf(x,y,z,levels,vmin=-vmax,vmax=vmax,antialiased=True,**kwargs)
        mappable.append(cntf)
        for c in cntf.collections:
            c.set_edgecolor("face")
        if (contour)&(sum(np.array(weights)!=0)>0):
            cnt = ax.contour(x,y,z,contourLines,colors=colors,linestyles='solid')
            mappable.append(cnt)

    ax.axis('square')
    ax.set_axis_off()

    return fig,ax,mappable



def GetSimilarityMapFromWeights(mol, weights, colorMap=None, scale=-1, size=(250, 250),
                                sigma=None, coordScale=1.5, step=0.01, colors='k', contourLines=10,
                                alpha=0.5, **kwargs):
  """
  Generates the similarity map for a molecule given the atomic weights.
  Parameters:
    mol -- the molecule of interest
    colorMap -- the matplotlib color map scheme, default is custom PiWG color map
    scale -- the scaling: scale < 0 -> the absolute maximum weight is used as maximum scale
                          scale = double -> this is the maximum scale
    size -- the size of the figure
    sigma -- the sigma for the Gaussians
    coordScale -- scaling factor for the coordinates
    step -- the step for calcAtomGaussian
    colors -- color of the contour lines
    contourLines -- if integer number N: N contour lines are drawn
                    if list(numbers): contour lines at these numbers are drawn
    alpha -- the alpha blending value for the contour lines
    kwargs -- additional arguments for drawing
  """
  if mol.GetNumAtoms() < 2:
    raise ValueError("too few atoms")
  fig = Draw.MolToMPL(mol, coordScale=coordScale, size=size, **kwargs)
  if sigma is None:
    if mol.GetNumBonds() > 0:
      bond = mol.GetBondWithIdx(0)
      idx1 = bond.GetBeginAtomIdx()
      idx2 = bond.GetEndAtomIdx()
      sigma = 0.3 * math.sqrt(
        sum([(mol._atomPs[idx1][i] - mol._atomPs[idx2][i])**2 for i in range(2)]))
    else:
      sigma = 0.3 * math.sqrt(sum([(mol._atomPs[0][i] - mol._atomPs[1][i])**2 for i in range(2)]))
    sigma = round(sigma, 2)

  x, y, z = calcAtomGaussians(mol, sigma, weights=weights, step=step)

  # scaling
  if scale <= 0.0:
    maxScale = max(math.fabs(numpy.min(z)), math.fabs(numpy.max(z)))
    maxScale = maxScale
  else:
    maxScale = scale
  # coloring
  if colorMap is None:
    PiYG_cmap = cm.get_cmap('PiYG',2)
    colorMap = LinearSegmentedColormap.from_list('PiWG', [PiYG_cmap(0), (1.0, 1.0, 1.0), PiYG_cmap(1)], N=255)

  if np.any(z):
      scls = np.array([maxScale,-maxScale])
      cnt = fig.axes[0].contourf(x,y,z,np.linspace(scls.min(),scls.max(),50),cmap=colorMap, vmin=-maxScale, vmax=maxScale)
      fig.colorbar(cnt)

      ticks = cbar.get_ticks()
      ticklabels = ["{:.4f} ({:.2f})".format(micromol(pred+tick)*1e+3,pred+tick) for tick in ticks]
      ind = (len(ticklabels)-1)//2
      ticklabels[ind] = ticklabels[ind]+' Overall Contrib.'
      cbar.set_ticklabels(ticklabels)
      # This is the fix for the white lines between contour levels
      for c in cnt.collections:
        c.set_edgecolor("face")
      # contour lines
      # only draw them when at least one weight is not zero
      if len([w for w in weights if w != 0.0]):
        contourset = fig.axes[0].contour(x, y, z, contourLines, colors=colors, alpha=alpha, **kwargs)
        for j, c in enumerate(contourset.collections):
            if contourset.levels[j] == 0.0:
                c.set_linewidth(0.0)
            elif contourset.levels[j] < 0:
                c.set_dashes([(0, (3.0, 3.0))])
  fig.axes[0].set_axis_off()
  fig.axes[0].axis('equal')
  return fig,dict(X=x,Y=y,Z=z)


def GetSimilarityMapForFingerprint(refMol, probeMol, fpFunction, metric=DataStructs.DiceSimilarity,
                                   **kwargs):
  """
  Generates the similarity map for a given reference and probe molecule,
  fingerprint function and similarity metric.
  Parameters:
    refMol -- the reference molecule
    probeMol -- the probe molecule
    fpFunction -- the fingerprint function
    metric -- the similarity metric.
    kwargs -- additional arguments for drawing
  """
  weights = SimilarityMaps.GetAtomicWeightsForFingerprint(refMol, probeMol, fpFunction, metric)
  fig = GetSimilarityMapFromWeights(probeMol, weights, **kwargs)
  return fig,weights



def GetSimilarityMapForModel(probeMol, fpFunction, predictionFunction, **kwargs):
    """
    Generates the similarity map for a given ML model and probe molecule,
    and fingerprint function.
    Parameters:
        probeMol -- the probe molecule
        fpFunction -- the fingerprint function
        predictionFunction -- the prediction function of the ML model
        kwargs -- additional arguments for drawing
    """
    weights = SimilarityMaps.GetAtomicWeightsForModel(probeMol,
                                    fpFunction, predictionFunction)
    fig,grid = GetSimilarityMapFromWeights(probeMol,
                                    weights, **kwargs)
    return fig,weights,grid
