#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
##########################
Alerts Notebook Tools
##########################

*Created on Wed Apr 22 16:37:35 2018 Rodolpho C. Braga*

A set of modelig tools to use in the IPython (JuPyTer) Notebook
"""


def standardMol(input_mol, output_rules_applied=None, verbose=False):
    '''
    Based on standardiser.standardise.run()
    '''
    # Get input molecule...

    if type(input_mol) == Chem.rdchem.Mol:
        mol = input_mol
        input_type = 'mol'
    else:
        mol = Chem.MolFromMolBlock(input_mol)
        if not mol:
            mol = Chem.MolFromSmiles(input_mol)
            if not mol:
                raise StandardiseException("not_built")
            else:
                input_type = 'smi'
        else:
            input_type = 'sdf'
    try:
        sanity_check(mol)
    except StandardiseException as err:
        print("Molecule failed sanity check")
        raise

    ######
    # Get disconnected fragments...
    non_salt_frags = []
    mol = break_bonds.run(mol)

    for n, frag in enumerate(Chem.GetMolFrags(mol, asMols=True), 1):
        if unsalt.is_nonorganic(frag): continue
        frag = neutralise.run(frag)
        frag = rules.run(frag, output_rules_applied=output_rules_applied, verbose=verbose)
        frag = neutralise.run(frag)
        if unsalt.is_salt(frag): continue
        non_salt_frags.append(frag)
    if len(non_salt_frags) == 0:
        raise StandardiseException("no_non_salt")
    if len(non_salt_frags) > 1:
        raise StandardiseException("multi_component")
    parent = non_salt_frags[0]

    ######
    # Return parent in same format as input...
    if input_type == 'mol':
        return parent
    elif input_type == 'sdf':
        return Chem.MolToMolBlock(parent)
    else:  # input_type == 'smi'
        return Chem.MolToSmiles(parent, isomericSmiles=True)


def getmol(smi):

    #PADRONIZANDO SMILES
    molSMILES = smi.split(" |")[0]
    molSMILES = molSMILES.replace("[C]",'C').replace("[c]",'c')

    try:
        molSMILES = standardMol(molSMILES)
    except:
        None

    molSMILES = molSMILES.replace("[C]",'C').replace("[c]",'c')

    return molSMILES


from bs4 import BeautifulSoup
import copy

def html_desc_formatting(page):

    soup = BeautifulSoup(page,'lxml')

    ps = soup.findAll('p')

    title = ps[0].getText().replace('\xa0','').strip()

    html = []
    html.append('<h4 class = "text-center">'+title+'</h4><br><br>')

    texts = []
    for p in ps[1:]:
        img = p.find('img')
        text = p.getText().replace('\xa0','').strip()
        if img!=None:
            texts.append(repr(img))
        else:
            texts.append(text)
    texts = ('&&'.join(texts)).replace('or&&','or ').split('&&')

    for text in texts:
        if text!='':
            if 'Cited' in text:
                html.append('<br><br><h5 class = "text-center">'+text+'</h5><br>')
            elif text.startswith('<img'):
                html.append('<p class = "text-center">'+text+'</p>')
            else:
                html.append('<p class = "text-justify">'+text+'</p>')

    # html = '<br>'.join(html)
    html = ''.join(html)

    return html


class iSens():

    def __init__(self):


        alerts_data = pd.read_excel('../Alerts/data/SkinSens_Alerts_DB.xlsx')
        html = pd.read_csv('../Alerts/data/SkinSens_Alerts_DB_HTML.csv','altox')
        alerts_data = alerts_data[~alerts_data['Curated SMARTS'].isna()]
        alerts_data['Description'] = html['Description'].apply(html_desc_formatting)
        alerts_data['Patterns'] = alerts_data['Curated SMARTS'].apply(Chem.MolFromSmarts)

        nots = alerts_data[alerts_data.columns[alerts_data.columns.to_series().str.contains('Not#')]].fillna('')
        nots = [N[[n!='' for n in N]].astype('str') for N in nots.values]

        alerts_data['Not'] = nots
        alerts_data['logP'] = alerts_data['physical-chemical values'].fillna('!=float')

        masks = alerts_data[alerts_data.columns[alerts_data.columns.to_series().str.contains('Mask')]].fillna('')
        masks = [list(map(Chem.MolFromSmarts,M[[m!='' for m in M]].astype('str'))) for M in masks.values]

        alerts_data['Masks'] = masks

        refs = alerts_data[alerts_data.columns[alerts_data.columns.to_series().str.contains('Reference')]].fillna('')
        refs = [R[[r!='' for r in R]] for R in refs.values]

        alerts_data['References'] = refs

        alerts_data = alerts_data[['Category','Alert','Patterns','Masks','Not','logP','References','Description']]

        self.alerts = alerts_data

        alert_d = alerts_data.Category.apply(lambda a: a.split('(')[0].strip().lower()).values+\
                    ' >> '+alerts_data.Alert.apply(lambda a: a.strip().lower()).values

        self.dummies = pd.Series(np.zeros(np.unique(alert_d).size),index=np.unique(alert_d))

    def alert_test(self,molQuery):

        alerts_data = copy.deepcopy(self.alerts)

        dummies = copy.deepcopy(self.dummies)

        mol = copy.deepcopy(molQuery)

        logp = MolLogP(mol)
        mols = [mol]

        molk = copy.deepcopy(mol)
        Chem.Kekulize(molk,clearAromaticFlags=True)

        mols = mols+[Chem.AddHs(mol),molk,Chem.AddHs(molk)]


        has_sma = np.array([any([m.HasSubstructMatch(sub) for m in mols]) for sub in alerts_data['Patterns']])
        has_mas = np.array([any([any([m.HasSubstructMatch(ma) for m in mols]) for ma in mas]) for mas in alerts_data['Masks']])
        has_log = np.array([all([eval('{}{}'.format(logp,r)) for r in phys.split('&')]) for phys in alerts_data['logP']])

        has_all = (has_sma)&(~has_mas)&(has_log)

        f_alerts = alerts_data.iloc[has_all,:]

        if f_alerts.size>0:
            contain = []
            for N in f_alerts.Not.values:
                contain.append(any([(f_alerts.Alert == n).any() for n in N]))

            f_alerts = f_alerts.iloc[~np.array(contain)]


    #         imgs = []
    #         for i,row in f_alerts.iterrows():
    #             matches = []
    #             for sub in f_alerts['Patterns'][f_alerts.Alert==row.Alert].values:
    #                 matches.append([np.array(m.GetSubstructMatches(sub)).ravel() for m in mols])
    #             matches = np.unique(np.hstack(matches[0]))
    #             matches = matches[matches<mol.GetNumAtoms()].astype('int').tolist()

    #             rdDepictor.Compute2DCoords(mol)

    #             drawer = Draw.rdMolDraw2D.MolDraw2DSVG(400,400)
    #             drawer.DrawMolecule(mol,highlightAtoms=matches)
    #             drawer.FinishDrawing()
    #             svg = drawer.GetDrawingText().replace('svg:','').replace('#FFFFFF', 'none')

    #             svg = svg.replace('\n','')

    #             # img = img2html(svg2png(svg,scale=4),'molImg')
    #             img = svg

    #             imgs.append(img)

    #         f_alerts['Image'] = imgs

            f_alerts = f_alerts.drop_duplicates('Alert',keep='first')

            f_alerts['References'] = f_alerts['References'].apply(lambda a: '<br>'.join(a))


            d1 = f_alerts.Category.apply(lambda a: a.split('(')[0].strip().lower()).values+\
                    ' >> '+f_alerts.Alert.apply(lambda a: a.strip().lower()).values

            dummies[d1] = 1


            A1 = f_alerts['Category'].str.contains('Category 1A').sum()
            B1 = f_alerts['Category'].str.contains('Category 1B').sum()
        else:
            A1,B1 = 0,0

        descriptions = f_alerts['Description'].values
        dummies = dummies.astype('bool')

        f_alerts = f_alerts[['Category','Alert','References']]

        return f_alerts,descriptions,(A1,B1),dummies.to_frame().T
