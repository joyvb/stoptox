#!/usr/bin/env python
# -*- coding: utf-8 -*-
__authors__ = ["Iury T. S. de Sousa"]
__email__ = ["iury@altox.com.br"]
__created_ = ["10-Out-2017"]
__modified__ = ["16-Nov-2017"]


from rdkit.Chem import rdDepictor,Draw,MolFromSmiles
#!/usr/bin/env python
# -*- coding: utf-8 -*-
__authors__ = ["Iury T. S. de Sousa","Rodolpho C. Braga"]
__email__ = ["iury@altox.com.br","rodolpho@altox.com.br"]
__created_ = ["08-Dez-2017"]
__modified__ = ["05-Mar-2018"]


import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

import os
import matplotlib
matplotlib.use('Agg')

from sklearn.preprocessing import LabelEncoder
from rdkit import Chem,DataStructs
from rdkit.Chem import MACCSkeys,AllChem
from rdkit.Chem.Draw import SimilarityMaps
import matplotlib.pyplot as plt


from io import StringIO
import pandas as pd
import numpy as np
from datetime import datetime,timedelta
from sklearn.externals import joblib

from rdkit_fix import _pyGenMACCSKeys
from rdkit_fix import GetSimilarityMapFromWeights
from rdkit_fix import GetSimilarityMapForModel
from rdkit_fix import GetSimilarityMapForFingerprint

from sklearn.metrics import pairwise_distances

pd.set_option('display.max_colwidth', -1)

import seaborn as sns

from sklearn.metrics import confusion_matrix



hex2rgb = lambda h: tuple(int(h[i:i+2], 16)/255 for i in (1, 3 ,5))


cdict_mutagen_r = {'red': ((0.0,  0.0, 0.0),
                          (0.25, 0.3, 0.3),
                          (0.45, 1.0, 1.0),
                          (0.55, 1.0, 1.0),
                          (0.75, 0.75, 0.75),
                          (1.0,  0.33, 0.33)),
                 'blue': ((0.0,  0.0, 0.0),
                           (0.25, 0.3, 0.3),
                           (0.45, 1.0, 1.0),
                           (0.55, 1.0, 1.0),
                           (0.75, 0.3, 0.3),
                           (1.0,  0.0, 0.0)),
                 'green': ((0.0,  0.33, 0.33),
                          (0.25, 0.75, 0.75),
                          (0.45, 1.0, 1.0),
                          (0.55, 1.0, 1.0),
                          (0.75, 0.3, 0.3),
                          (1.0,  0.0, 0.0))
}


cdict_mutagen = {'green': ((0.0,  0.0, 0.0),
                          (0.25, 0.3, 0.3),
                          (0.45, 1.0, 1.0),
                          (0.55, 1.0, 1.0),
                          (0.75, 0.75, 0.75),
                          (1.0,  0.33, 0.33)),
                 'blue': ((0.0,  0.0, 0.0),
                           (0.25, 0.3, 0.3),
                           (0.45, 1.0, 1.0),
                           (0.55, 1.0, 1.0),
                           (0.75, 0.3, 0.3),
                           (1.0,  0.0, 0.0)),
                 'red': ((0.0,  0.33, 0.33),
                          (0.25, 0.75, 0.75),
                          (0.45, 1.0, 1.0),
                          (0.55, 1.0, 1.0),
                          (0.75, 0.3, 0.3),
                          (1.0,  0.0, 0.0))
}


Y2B = matplotlib.colors.LinearSegmentedColormap.from_list("", list(map(hex2rgb,[
'#ffdb3f',
'#fff356',
'#fff356',
'#ffffff',
'#ffffff',
'#56cfff',
'#56cfff',
'#2faded'
])))

G2R = matplotlib.colors.LinearSegmentedColormap(
'G2R', cdict_mutagen, 100)

G2R_r = matplotlib.colors.LinearSegmentedColormap(
'G2R_r', cdict_mutagen_r, 100)




def simil_map(infoFP,molref,mol,size=1.5,**kwargs):

    fig,weights = GetSimilarityMapForFingerprint(molref,mol,
                    infoFP.funcFP,infoFP.funcSimil,**kwargs)

    output = StringIO()
    dSize = fig.get_size_inches()/fig.get_size_inches().max()
    fig.set_size_inches(tuple(dSize*size))

    fig.savefig(output, format='svg',bbox_inches='tight')
    plt.close('all')


    svg = output.getvalue()
    svg = svg.replace('svg:','')
    svg = svg.replace('#FFFFFF', 'none').replace('#ffffff', 'none')
    svg = svg.replace('\n','')

    return svg

def cbar_svg(cmap,title,limits,size=4.3):

    plt.ioff()
    # Make a figure and axes with dimensions as desired.
    fig = plt.figure(figsize=(5, 2))
    ax1 = fig.add_axes([0.05, 0.50, 0.9, 0.15])
    # Set the colormap and norm to correspond to the data for which
    # the colorbar will be used.

    norm = matplotlib.colors.Normalize(vmin=0, vmax=1)

    # ColorbarBase derives from ScalarMappable and puts a colorbar
    # in a specified axes, so it has everything needed for a
    # standalone colorbar.  There are many more kwargs, but the
    # following gives a basic continuous colorbar with ticks
    # and labels.
    cb1 = matplotlib.colorbar.ColorbarBase(ax1,cmap=cmap,
                                    norm=norm,
                                    orientation='horizontal')
    cb1.ax.set_title(title,fontweight='bold',fontsize=18)
    cb1.ax.tick_params(labelsize=15)
    cb1.set_ticks([0,1])
    cb1.set_ticklabels(limits)#,labelsize=20)

    output = StringIO()
    dSize = fig.get_size_inches()/fig.get_size_inches().max()
    fig.set_size_inches(tuple(dSize*size))

    fig.savefig(output, format='svg',bbox_inches='tight')
    plt.close()

    svg = output.getvalue()
    svg = svg.replace('svg:','')
    svg = svg.replace('#FFFFFF', 'none').replace('#ffffff', 'none')
    svg = svg.replace('\n','')

    return svg




def html_formatting(table,index=False):
    tableHTML = table.to_html(border=0,escape=False,index=index)

    tableHTML = tableHTML.replace('<table border="0" class="dataframe">',
                            '<table class="table table-striped table-hover text-center">')
    tableHTML = tableHTML.replace('<thead>',
                            '<thead class="table-inverse text-center">')
    tableHTML = tableHTML.replace('<th>',
                            '<th class="align-middle text-center" style="vertical-align: middle">')
    tableHTML = tableHTML.replace('<td>',
                            '<td class="align-middle" style="vertical-align: middle">')
    return tableHTML


def prediction_statistics_graph(modelsDF,n_samples,threshold=0.7,size=10):
    n = modelsDF.shape[0]

    norm = matplotlib.colors.Normalize(vmin=-n*2, vmax=n*2)
    cmap = matplotlib.cm.get_cmap('Blues')

    ind = np.linspace(0,n*0.7,3)
    width = n/15

    colors = cmap(norm(np.linspace(0,n*2,n)))

    yps = np.stack([np.linspace(i+width,i-width,n) for i in ind]).T

    haszero=False

    plt.ioff()
    sns.set(style='ticks',context='talk')
    fig = plt.figure(figsize=(17,8))
    ax = fig.add_axes([0.1, 0.2, 0.85, 0.7])
    for values,color,yp in zip(modelsDF.values,colors,yps):
        ax.barh(yp,values,
                width, color=color, label='M')
        for value,ypi in zip(values,yp):
            if value == 0:
                haszero=True
                ax.text(value,ypi,'*',
                    fontsize=20,va='center')
            else:
                ax.text(value,ypi,'{:.2f}'.format(value),
                    fontsize=13,va='center')

    ax.set_yticklabels([])
    ax.set(xticks=np.arange(0,1+0.1,0.1),
           yticks=ind,
           yticklabels=modelsDF.columns)
    ax.legend(modelsDF.index,
                bbox_to_anchor=(1.05, 1),
                loc=2,
                borderaxespad=0.)
    sns.despine()

    # fig.suptitle('Prediction '+\
    #              'Confidence '+\
    #              'Statistics \n',
    #              fontweight='bold',fontsize=15)

    ax.set_title('                    '+\
                '(n = {} compounds / Similarity >= {})\n\n\n\n'.format(n_samples,threshold),
                 fontweight='bold',fontsize=12)


    output = StringIO()
    dSize = fig.get_size_inches()/fig.get_size_inches().max()
    fig.set_size_inches(tuple(dSize*size))

    if haszero:
        ax.annotate(r'*  There is no experimental data for '+\
                    'the compounds with dice similarity '+\
                    'threshold >= {}'.format(threshold),
                    (0,-.1), (0, -20),
                    xycoords='axes fraction',
                    textcoords='offset points',
                    va='top')

    fig.savefig(output, format='svg',bbox_inches='tight')
    plt.close()

    sns.reset_orig()

    svg = output.getvalue()
    svg = svg.replace('svg:','')
    svg = svg.replace('#FFFFFF', 'none').replace('#ffffff', 'none')
    svg = svg.replace('\n','')

    return svg



def prediction_statistics(pred,simil,class_names,name='unnamed'):

    results = set(class_names)

    pos = class_names[-1]
    neg = list(results.difference({pred}))[0]

    Prediction = simil.Prediction[simil.Confiability>58]
    Outcome = simil.Outcome[simil.Confiability>58]

    Concordance = Prediction[Prediction==pred].count()/Prediction.count()


    TN, FP, FN, TP = confusion_matrix(Outcome,Prediction,labels=class_names).ravel()


    if TP==0:
        Sensitivity = 0
    else:
        Sensitivity = TP/(TP+FN)


    if TN==0:
        Specificity = 0
    else:
        Specificity = TN/(TN+FP)


    statistics = pd.Series([Specificity,Sensitivity,Concordance],
        index='Specificity Sensitivity Concordance'.split(),name=name)

    n = Prediction.shape[0]

    return statistics,n



def proba_map(model,molQuery,size=1.5,**kwargs):

    def _get_proba(fp, predictionFunction):
      return predictionFunction((fp,))[0][1]

    if hasattr(molQuery,'_fpinfo'): delattr(molQuery,'_fpinfo')
    fig,weights = GetSimilarityMapForModel(molQuery,
                    model.funcFP,
                    lambda x:  _get_proba(x, model.model.predict_proba),
                    **kwargs)

    output = StringIO()
    dSize = fig.get_size_inches()/fig.get_size_inches().max()
    fig.set_size_inches(tuple(dSize*size))

    fig.savefig(output, format='svg',bbox_inches='tight')
    plt.close()

    svg = output.getvalue()
    svg = svg.replace('svg:','')
    svg = svg.replace('#FFFFFF', 'none').replace('#ffffff', 'none')
    svg = svg.replace('\n','')

    return svg

def get_mol_svg(mol):
    if type(mol)==str:
        mol = MolFromSmiles(mol)
    rdDepictor.Compute2DCoords(mol)
    size = 400
    drawer = Draw.rdMolDraw2D.MolDraw2DSVG(size,size)
    drawer.DrawMolecule(mol)
    drawer.FinishDrawing()
    svg = drawer.GetDrawingText().replace('svg:','').replace('#FFFFFF', 'none')

    svg = svg.replace('\n','')
    return svg


def applicability_domain(info,molQuery):

    # modelname = self.name
    if hasattr(molQuery,'_fpinfo'): delattr(molQuery,'_fpinfo')
    fp = np.array(info.funcFP(molQuery)).reshape((1, -1))

    dist_da = info.applM
    dda = info.dda

    neighbors_k = pairwise_distances(np.vstack(info.fpDF),np.vstack([fp]*2),
                        metric='dice', n_jobs=-1).T[0]
    neighbors_k.sort(0)
    similarity = 1-neighbors_k
    dist_mq = similarity[info.k-1]

    sns.set(style='ticks',context='talk')

    if dist_mq<=dda:
        result = '(Outside the Domain)'
    else:
        result = '(Within the Domain)'

    fig,ax = plt.subplots(figsize=(3,3))
    # fig.suptitle(modelname,fontweight='bold')
    sns.kdeplot(dist_da,shade=True,ax=ax)
    x, y = ax.lines[0].get_xydata().T
    m = x<=dda
    ax.fill_between(x[m], 0, y[m], alpha=0.3, color="r")
    ax.set(title=result,yticklabels=[],xlim=(0,1),
                ylabel='Density')
    ax.set_xlabel('{}\nDice Similarity'.format(info.fpType),fontsize=12)

    ax.scatter(dist_mq,0,250,'k',marker='o',clip_on=False)
    sns.despine()

    sns.reset_orig()


    output = StringIO()
    # dSize = fig.get_size_inches()/fig.get_size_inches().max()
    # fig.set_size_inches(tuple(dSize*size))

    fig.savefig(output, format='svg',bbox_inches='tight')
    plt.close()

    svg = output.getvalue()
    svg = svg.replace('svg:','')
    svg = svg.replace('#FFFFFF', 'none').replace('#ffffff', 'none')
    svg = svg.replace('\n','')

    return svg
