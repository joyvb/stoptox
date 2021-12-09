from __future__ import division
import os
from datetime import datetime
from flask import Flask ,  make_response, send_from_directory, send_file
from flask import render_template, url_for, request,  redirect, flash
from flask_bootstrap import Bootstrap
from flask_wtf import FlaskForm, RecaptchaField
from wtforms import StringField, TextField, StringField, TextAreaField, SubmitField, validators, form
from wtforms.validators import DataRequired
from sklearn.externals import joblib
from rdkit import Chem
import pandas as pd
from werkzeug.utils import secure_filename
from flask_wtf.file import FileField, FileAllowed, FileRequired
from wtforms.validators import InputRequired, Email
from flask_mail import Mail, Message
from flask import send_from_directory
#Binary Model
import  gzip, numpy, copy, math
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem, Draw, rdmolops
from rdkit.Chem.AtomPairs import Pairs
from rdkit.Chem import rdMolDescriptors
from matplotlib import cm
from sklearn.ensemble import RandomForestClassifier
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem, Draw, rdmolops
from rdkit.Chem.AtomPairs import Pairs
from rdkit.Chem import rdMolDescriptors
from matplotlib import cm
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import BernoulliNB
import numpy as np
# Libraries
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem, Draw, rdmolops
from rdkit.Chem.AtomPairs import Pairs
from rdkit.Chem import rdMolDescriptors
from matplotlib import cm
import numpy as np
from sklearn import metrics
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from array import array
import numpy as np
import matplotlib.pyplot as plt
import os
import argparse
import pprint
import numpy as np
from sklearn.utils import shuffle
from time import strftime
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, RandomForestClassifier, \
    GradientBoostingClassifier
from sklearn.externals import joblib
from sklearn.preprocessing import StandardScaler
from sklearn import svm
from sklearn.metrics import make_scorer
from multiprocessing import cpu_count
import sklearn.metrics.pairwise
import scipy.spatial.distance
from array import array
import pandas as pd
from io import StringIO
from sklearn.preprocessing import StandardScaler
import multiprocessing
from sklearn import decomposition, pipeline, metrics
from sklearn.svm import SVC
from rdkit.Chem import PandasTools
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem, Draw, rdmolops
from rdkit.Chem.AtomPairs import Pairs
from rdkit.Chem import rdMolDescriptors
from matplotlib import cm
from sklearn.ensemble import RandomForestClassifier
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem, Draw, rdmolops
from rdkit.Chem.AtomPairs import Pairs
from rdkit.Chem import rdMolDescriptors
from matplotlib import cm
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import BernoulliNB
from pandas import ExcelWriter
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem
from rdkit.Chem import Descriptors
from rdkit.Chem.Draw import SimilarityMaps
from sklearn.svm import SVC
from matplotlib import cm
import matplotlib
# Force matplotlib to not use any Xwindows backend.
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pickle
from sklearn.externals import joblib
import uuid
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, RandomForestClassifier
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.image import MIMEImage
from email.mime.base import MIMEBase
from email import encoders
import zipfile
from logging import FileHandler, WARNING


from rdkit.Chem.MACCSkeys import maccsKeys, smartsPatts,_InitKeys

from rdkit.Chem import MACCSkeys

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


def getNeighborsDitance(trainingSet, testInstance, k):
    neighbors_k=sklearn.metrics.pairwise.pairwise_distances(trainingSet, Y=testInstance, metric='dice', n_jobs=1)
    neighbors_k.sort(0)
    similarity= 1-neighbors_k
    return similarity[k-1,:]

import bz2
import pickle
import _pickle as cPickle

pd.set_option('display.max_rows', 1000)
pd.set_option('display.max_columns', 5)
pd.set_option('max_colwidth',250)
pd.set_option('precision', 2)


#Xls filename
current_time = str(datetime.utcnow())
current_time = "_".join(current_time.split()).replace(":","-")
current_time = current_time[:-7]
new_filename_xls = 'Pred-skin-Results_'+current_time+uuid.uuid1().hex+'.xlsx'

#Server side
new_file_path_xls = os.path.join('predskin_report', new_filename_xls)



## Acute Inhalation
model_binary_human = joblib.load('model/Modelo_acute_inhal_MACCS.pkl')
##  Eye irritation
model_binaryLLNA = joblib.load('model/Modelo_eye_irrit_MACCS.pkl')

## Acute dermal
model_DPRA = joblib.load('model/Modelo_acute_dermal_MACCS.pkl')

###Acute Oral
model_hCLAT = joblib.load('model/Modelo_acute_oral_MACCS.pkl')

###Skin Sensitization
model_SS_LLNA_new = joblib.load('model/Modelo_Skin_Sens.pkl')

###Skin Sensitization
model_SI = joblib.load('model/Modelo_skin_irrit_morgan.pkl')



# hard coded variables
bit_size = 2048
radius = 2

ncores=multiprocessing.cpu_count()
verbose=1
seed = 42

def mapperfunc( mol, model ):

    def getProba(fp, predictionFunction):
        return predictionFunction(np.array(fp).reshape(1, -1))[0][1]


    weights = SimilarityMaps.GetAtomicWeightsForModel(mol, _pyGenMACCSKeys, lambda x: getProba(x, model.predict_proba))
    SimilarityMaps.GetSimilarityMapFromWeights(mol, weights,colorMap=cm.PiYG)

    #Output model predictions
    fp = np.array(MACCSkeys.GenMACCSKeys(mol)).reshape(1, -1)
    cls_result= model.predict(fp)[0]

    #Output model probability
    proba = round(round(np.max(model.predict_proba(fp) ) ,2)*100, 1)
    #png base64
    from io import BytesIO
    figfile = BytesIO()
    plt.savefig(figfile, format='png', bbox_inches = "tight")
    figfile.seek(0)  # rewind to beginning of file
    import base64
    fname= base64.b64encode(figfile.getvalue())
    fname=fname.decode("utf-8")

    return cls_result, str(proba)+"%", fname


def mapperfuncMorgan( mol, model, nBits=2048, radius=2, useFeatures=False, class1="Active (+)", class2="Inactive (-)" ):

    def getProba( fp, probabilityfunc ):
        return probabilityfunc( fp )[0][1]

    def fpFunction(mol,atomId=-1):
        fp = SimilarityMaps.GetMorganFingerprint(mol,
                                                 atomId=atomId,
                                                 radius=radius,
                                                 nBits=nBits,
                                                 useFeatures=useFeatures
                                                )
        return fp

    fig, maxweight = SimilarityMaps.GetSimilarityMapForModel(mol,
                                                             fpFunction,
                                                             lambda x: getProba((x,),
                                                             model.predict_proba),
                                                             colorMap=cm.PiYG)

    fp = np.array( AllChem.GetMorganFingerprintAsBitVect( mol, radius, nBits,  useFeatures))


    # LabelEncoder can be used to normalize labels.
    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()
    le.fit([class1,  class2])

    #Output model predictions
    cls_result = le.inverse_transform( model.predict(fp.reshape(1, -1)).tolist() )

    #Output model probability
    proba = int((round(max(max(model.predict_proba(fp.reshape((1, -1)))) ) ,1) )*100)
    #png base64
    from io import BytesIO
    figfile = BytesIO()
    plt.savefig(figfile, format='png', bbox_inches = "tight")
    figfile.seek(0)  # rewind to beginning of file
    import base64
    fname= base64.b64encode(figfile.getvalue())
    fname=fname.decode("utf-8")

    return cls_result[0], str(proba)+"%", fname


def mapperfunc_sdf( mol):

    fp = np.array( AllChem.GetMorganFingerprintAsBitVect( mol, 2, 2048, useFeatures=False))

    # LabelEncoder can be used to normalize labels.
    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()
    le.fit(["Sensitizer", "Non-Sensitizer"])

    #Output model predictions
    cls_result = le.inverse_transform( model.predict(fp.reshape(1, -1)).tolist() )

    return cls_result[0]

def mapperfunc_sdf_Proba( mol):

    fp = np.array( AllChem.GetMorganFingerprintAsBitVect( mol, 2, 2048, useFeatures=False))

    # LabelEncoder can be used to normalize labels.
    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()
    le.fit(["Sensitizer", "Non-Sensitizer"])

    proba =  int((np.round(max(max(model.predict_proba(fp.reshape(1, -1))) ) ,1))*100)
    return str(proba)+"%"



#########################################################################################
################## physical chemical Predictions Function ###############################
#########################################################################################

def molprop_calc( mol ):
    mw = round( Descriptors.MolWt( mol ), 2 )
    mollogp = round( Descriptors.MolLogP( mol ), 2 )
    tpsa = round( Descriptors.TPSA( mol ), 2  )
    return [ mw, mollogp, tpsa ]

##email

def py_mail(SUBJECT, BODY, TO, FROM):
    """With this function we send out our html email"""

    # Create message container - the correct MIME type is multipart/alternative here!
    MESSAGE = MIMEMultipart('related')
    MESSAGE['subject'] = SUBJECT
    MESSAGE['To'] = TO
    MESSAGE['From'] = FROM

    msg = MIMEBase('application', 'zip')
    zf = open('Results.zip', 'rb')
    msg.set_payload(zf.read())
    encoders.encode_base64(msg)
    msg.add_header('Content-Disposition', 'attachment',
               filename='Results' + '.zip')
    #MESSAGE.attach(msg)

    MESSAGE.preamble = """
Your mail reader does not support the report format.
Please visit us <a href="http://www.mysite.com">online</a>!"""


    # Record the MIME type text/html.
    HTML_BODY = MIMEText(BODY, 'html')

    # Attach parts into message container.
    # According to RFC 2046, the last part of a multipart message, in this case
    # the HTML message, is best and preferred.
    MESSAGE.attach(HTML_BODY)

    # The actual sending of the e-mail
    server = smtplib.SMTP('smtp.gmail.com:587')

    # Print debugging output when testing
    #if __name__ == "__main__":
    #    server.set_debuglevel(0)

    # Credentials (if needed) for sending the mail
    password = "iizthltpchcpqwop"

    server.starttls()
    server.login(FROM,password)
    MESSAGE.attach(msg)
    server.sendmail(FROM, [TO], MESSAGE.as_string())

    server.quit()

email_content = """
<!DOCTYPE html PUBLIC "-//W3C//DTD XHTML 1.0 Transitional//EN" "http://www.w3.org/TR/xhtml1/DTD/xhtml1-transitional.dtd">
<html xmlns="http://www.w3.org/1999/xhtml">
<head>
<meta http-equiv="Content-Type" content="text/html; charset=UTF-8" />
<title>Untitled Document</title>
<style type="text/css">
.Title3 {
	color: #37761D;
	font-family: Arial, Helvetica, sans-serif;
	font-size: 16px;
	font-weight: bold;
}
.Text_body {
	font-family: Arial, Helvetica, sans-serif;
	font-size: 14px;
	font-style: normal;
	font-weight: normal;
	font-variant: normal;
}
</style>
</head>

<body>
<p align="center"><img src="http://www.labmol.com.br/wp-content/uploads/2018/02/logo-email-e1518390685621.png" /></p>
<h3 align="center" class="Title3">RESULTS</h3>
<p align="center" class="Text_body">There is an attached file named &quot;Results.zip&quot;</p>
<p align="center" class="Title3">Citation</p>
<p align="center" class="Text_body">If this server was useful to you, please cite our work and help us maintaining this service up to date. </p>
<p align="center" class="Text_body"><strong>1.</strong> Braga, R. C.; Alves, V. M.; Muratov, E. N.; Strickland, J.; Kleinstreuer, N.; Trospsha, A.; Andrade, C. H. Pred-Skin: A fast and reliable tool to assess chemically-induced skin sensitization. J. Chem. Inf. Model. <strong>2017</strong>, 57 (5), 1013-1017.</p>
<p align="center" class="Text_body"><strong>2.</strong> A Perspective and a New Integrated Computational Strategy for Skin Sensitization Assessment ACS Sustainable Chem. Eng., <strong>2018</strong>, Just Accepted </p>
<h3 align="center" class="Title3">TERMS OF CONFIDENTIALITY</h3>
<p class="Text_body">The information sent to PredSkin is confidential. The web server safeguard and keep the structures sent in the strictest confidence at all times. We do not disclose or divulge any of the confidential information to any third party. We take all reasonably available measures to preserve the confidentiality of, and not to disclose, any information sent by users. The server do not save any data regarding the structures or IP of users. All data is automatically erased from the server as soon as the job is done and the report sent.</p>
<p class="Title3">Thank you, </p>
<p class="Title3">LabMol Team</p>
</body>
</html>

"""
FROM ='labmol.group@gmail.com'

###


def create_app():
    application = Flask( __name__ )
    Bootstrap( application )
    return application


application = create_app()
#application.debug = True
#Log
#file_handler = FileHandler('errorlog.txt')
#file_handler.setLevel(WARNING)
#application.logger.addHandler(file_handler)



## Security
# This is the secret key that is used for session signing.
# You can generate a secure key with os.urandom(24)
application.secret_key = 'reallyhardtoguess'
SECRET_KEY = 'reallyhardtoguess'

# You can generate the WTF_CSRF_SECRET_KEY the same way as you have
# generated the SECRET_KEY. If no WTF_CSRF_SECRET_KEY is provided, it will
# use the SECRET_KEY.
WTF_CSRF_ENABLED = True
WTF_CSRF_SECRET_KEY = "reallyhardtoguess"

#Server side
UPLOAD_FOLDER = 'uploads'



#ALLOWED_EXTENSIONS = set(['txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif'])
application.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER




#application.config['RECAPTCHA_USE_SSL'] = False
#application.config['RECAPTCHA_PUBLIC_KEY'] = '6LeyZyQTAAAAADWeaporLt1wnTyXXWjyxpD6uyxm'
#application.config['RECAPTCHA_PRIVATE_KEY'] = '6LeyZyQTAAAAAF9tGkifbfXG2uBEOaIbHfJHDTar'
#application.config['RECAPTCHA_OPTIONS'] = {'theme': 'white'}


#mail

# add mail server config
application.config['MAIL_SERVER'] = 'smtp.gmail.com'
application.config['MAIL_PORT'] = 465
application.config['MAIL_USE_SSL'] = True
application.config['MAIL_USERNAME'] = 'labmol.group@gmail.com'
application.config['MAIL_PASSWORD'] = '9Hm-XzS-2Jz-Cs9'

def CheckNameLength(form, field):
  if len(field.data) < 4:
    raise ValidationError('Name must have more then 3 characters')

class ContactForm(FlaskForm):
    name = StringField('Your Name:', [validators.DataRequired(), CheckNameLength])
    email = StringField('Your e-mail address:', [validators.DataRequired(), validators.Email('your@email.com')])
    message = TextAreaField('Your message:', [validators.DataRequired()])
    file = FileField('image', validators=[
        FileRequired(),
        FileAllowed(['jpg', 'png'], 'Images only!')])
    #recaptcha = RecaptchaField()
    submit = SubmitField('Predict')



class UploadForm(FlaskForm):
    file = FileField('sdf', validators=[
        FileRequired(),
        FileAllowed(['sdf', 'SDF'], 'sdf only!')])
    name = TextField('Name:', validators=[validators.required()])
    email = TextField('Email:', validators=[validators.required(), validators.Length(min=6, max=35)])
    #recaptcha = RecaptchaField()







@application.route( "/top/" )
def top():
    return render_template( "top.html" )


@application.route( "/", methods=['GET', 'POST'])


def index():

    form = UploadForm()


    if form.validate_on_submit():
        filename = secure_filename(form.file.data.filename)
        file_path = os.path.join(application.config['UPLOAD_FOLDER'], filename)
        form.file.data.save(file_path)
        #rename file
        current_time = str(datetime.utcnow())
        current_time = "_".join(current_time.split()).replace(":","-")
        current_time = current_time[:-7]
        new_filename = 'sdf_'+current_time+'.sdf'
        new_file_path = os.path.join(application.config['UPLOAD_FOLDER'], new_filename)
        os.rename(file_path, new_file_path)

        #repor
        ncores=multiprocessing.cpu_count()
        verbose=1
        seed = 42
        file= new_file_path

        sdfInfo = dict(smilesName='SMILES',molColName='ROMol')
        moldf = PandasTools.LoadSDF(file,**sdfInfo)


        mols = [ m for m in Chem.ForwardSDMolSupplier((file)) if m != None]

        def calcfp(mol):
            arr = np.zeros((1,))
            fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, 2048,  useFeatures=False )
            DataStructs.ConvertToNumpyArray(fp, arr)
            return arr

        fp=moldf.ROMol.apply(calcfp)
        x = np.array(list(fp))

        # LabelEncoder can be used to normalize labels.
        from sklearn.preprocessing import LabelEncoder
        le = LabelEncoder()
        le.fit(["Sensitizer", "Non-Sensitizer"])
        le_mult = LabelEncoder()
        le_mult.fit(["Strong / Extreme Sensitizer", "Weak / Moderate Sensitizer", "Non-Sensitizer"])


        def report_data(x,model):
            ypred_binary = le.inverse_transform(model.predict(x))
            proba=np.round((model.predict_proba(x).max(axis=1)*100),0)
            yproba_binary= [str(suit)+"%" for suit in proba]
            report = pd.DataFrame({'1.SMILES':moldf.SMILES,
                                    '2.Prediction':ypred_binary,
                                    '3.Confidense':yproba_binary})
            return report

        def report_data_mult(x,model,modelmult):
            ypred_binary = le.inverse_transform(model.predict(x))
            proba=np.round((model.predict_proba(x).max(axis=1)*100),0)
            yproba_binary= [str(suit)+"%" for suit in proba]
            ypred_mult = le_mult.inverse_transform(modelmult.predict(x))
            proba_mult=np.round((modelmult.predict_proba(x).max(axis=1)*100),0)
            yproba_mult= [str(suit)+"%" for suit in proba_mult]
            report = pd.DataFrame({'1.SMILES':moldf.SMILES,
                                    '2.Binary Pred':ypred_binary,
                                    '3.Confidense':yproba_binary,
                                    '4.Multiclass Pred':ypred_mult,
                                    '5.Confidense':yproba_mult})
            return report
        #feed report
        report_human = report_data(x,model_binary_human)
        report_LLNA = report_data_mult(x,model_binaryLLNA,model_multiclassLLNA)
        report_DRPA = report_data(x,model_DPRA)
        report_DRPA = report_data(x,model_DPRA)
        report_hCLAT = report_data(x,model_hCLAT)

        report_Consensus = report_data(x,model_conensus)


        #Report to Excel
        number_rows = len(moldf.SMILES)
        # Create a Pandas Excel writer using XlsxWriter as the engine.
        writer = pd.ExcelWriter(new_file_path_xls, engine='xlsxwriter')

        #LOGO
        workbook = writer.book
        worksheet = workbook.add_worksheet('LabMol')
        worksheet.insert_image('A1', 'static/excel-logo.jpg',{'x_offset': 10, 'y_offset': 10})
        worksheet.set_zoom(65)


        report_human.to_excel(writer, index=False, sheet_name='Human Skin model')
        report_LLNA.to_excel(writer, index=False, sheet_name='LLNA model')
        report_DRPA.to_excel(writer, index=False, sheet_name='DRPA model')
        report_hCLAT.to_excel(writer, index=False, sheet_name='h-CLAT model')
        report_KeratinoSens.to_excel(writer, index=False, sheet_name='KeratoSens model')
        report_Consensus.to_excel(writer, index=False, sheet_name='FINAL RESULT (Consensus model)')


        ##Human
        # Get access to the workbook and sheet
        workbook = writer.book
        worksheet = writer.sheets['Human Skin model']
        worksheet.set_zoom(110)
        # Define our range for the color formatting
        color_range = "B2:B{}".format(number_rows+1)
        # Add a format. Light red fill with dark red text.
        format1 = workbook.add_format({'bg_color': '#FFC7CE',
                               'font_color': '#9C0006'})
        # Add a format. Green fill with dark green text.
        format2 = workbook.add_format({'bg_color': '#C6EFCE',
                               'font_color': '#006100'})

        # Highlight the bottom 5 values in Green
        worksheet.conditional_format(color_range, {'type': 'text','criteria': 'containing','value': 'Non-Sensitizer','format': format2})

        # Highlight the bottom 5 values in Red
        worksheet.conditional_format(color_range, {'type': 'text','criteria': 'containing', 'value': 'Sensitizer','format': format1})


        color_range_ad = "D2:D{}".format(number_rows+1)
        # Highlight the top 5 values in Green
        worksheet.conditional_format(color_range_ad, {'type': 'text',
                                           'criteria': 'containing',
                                           'value': 'Yes',
                                           'format': format2})
        # Highlight the bottom 5 values in Red
        worksheet.conditional_format(color_range_ad, {'type': 'text',
                                            'criteria': 'containing',
                                           'value': 'No',
                                           'format': format1})


        # APPLY CENTERING
        center = workbook.add_format({'align': 'center'})
        worksheet.set_column('B:L', None, center)
        # Set the columns widths.
        worksheet.set_column('A:A',40)
        worksheet.set_column('B:C', 12)
        worksheet.set_column('D:D', 6)


        #LLNA
        workbook = writer.book
        worksheet = writer.sheets['LLNA model']
        # Reduce the zoom a little
        worksheet.set_zoom(110)
        # Define our range for the color formatting
        color_range = "B2:B{}".format(number_rows+1)
        # Add a format. Light red fill with dark red text.
        format1 = workbook.add_format({'bg_color': '#FFC7CE',
                               'font_color': '#9C0006'})
        # Add a format. Green fill with dark green text.
        format2 = workbook.add_format({'bg_color': '#C6EFCE',
                               'font_color': '#006100'})

        # Highlight the bottom 5 values in Green
        worksheet.conditional_format(color_range, {'type': 'text','criteria': 'containing','value': 'Non-Sensitizer','format': format2})

        # Highlight the bottom 5 values in Red
        worksheet.conditional_format(color_range, {'type': 'text','criteria': 'containing', 'value': 'Sensitizer','format': format1})


        # Define our range for the color formatting Multi
        color_range = "D2:D{}".format(number_rows+1)
        # Add a format. Light red fill with dark red text.
        format1 = workbook.add_format({'bg_color': '#FFC7CE',
                               'font_color': '#9C0006'})
        # Add a format. Green fill with dark green text.
        format2 = workbook.add_format({'bg_color': '#C6EFCE',
                               'font_color': '#006100'})
        # Add a format. Yellow fill with dark green text.
        format3 = workbook.add_format({'bg_color': '#ffffb3',
                               'font_color': '#999900'})

        # Highlight the bottom 5 values in Red
        worksheet.conditional_format(color_range, {'type': 'text',
                                            'criteria': 'containing',
                                           'value': 'Strong / Extreme Sensitizer',
                                           'format': format1})

        # Highlight the top 5 values in Green
        worksheet.conditional_format(color_range, {'type': 'text',
                                           'criteria': 'containing',
                                           'value': 'Non-Sensitizer',
                                           'format': format2})

        # Highlight the top 5 values in Green
        worksheet.conditional_format(color_range, {'type': 'text',
                                           'criteria': 'containing',
                                           'value': 'Weak / Moderate Sensitizer',
                                           'format': format3})
        color_range_ad = "F2:F{}".format(number_rows+1)
        # Highlight the top 5 values in Green
        worksheet.conditional_format(color_range_ad, {'type': 'text',
                                           'criteria': 'containing',
                                           'value': 'Yes',
                                           'format': format2})
        # Highlight the bottom 5 values in Red
        worksheet.conditional_format(color_range_ad, {'type': 'text',
                                            'criteria': 'containing',
                                           'value': 'No',
                                           'format': format1})

        # APPLY CENTERING
        worksheet.set_column('B:L', None, center)
        # Set the columns widths.
        worksheet.set_column('A:A',40)
        worksheet.set_column('B:D', 22)
        worksheet.set_column('E:F', 12)


        ##DRPA
        # Get access to the workbook and sheet
        workbook = writer.book
        worksheet = writer.sheets['DRPA model']
        worksheet.set_zoom(110)
        # Define our range for the color formatting
        color_range = "B2:B{}".format(number_rows+1)
        # Add a format. Light red fill with dark red text.
        format1 = workbook.add_format({'bg_color': '#FFC7CE',
                               'font_color': '#9C0006'})
        # Add a format. Green fill with dark green text.
        format2 = workbook.add_format({'bg_color': '#C6EFCE',
                               'font_color': '#006100'})

        # Highlight the bottom 5 values in Green
        worksheet.conditional_format(color_range, {'type': 'text','criteria': 'containing','value': 'Non-Sensitizer','format': format2})

        # Highlight the bottom 5 values in Red
        worksheet.conditional_format(color_range, {'type': 'text','criteria': 'containing', 'value': 'Sensitizer','format': format1})


        color_range_ad = "D2:D{}".format(number_rows+1)
        # Highlight the top 5 values in Green
        worksheet.conditional_format(color_range_ad, {'type': 'text',
                                           'criteria': 'containing',
                                           'value': 'Yes',
                                           'format': format2})
        # Highlight the bottom 5 values in Red
        worksheet.conditional_format(color_range_ad, {'type': 'text',
                                            'criteria': 'containing',
                                           'value': 'No',
                                           'format': format1})


        # APPLY CENTERING
        center = workbook.add_format({'align': 'center'})
        worksheet.set_column('B:L', None, center)
        # Set the columns widths.
        worksheet.set_column('A:A',40)
        worksheet.set_column('B:C', 12)
        worksheet.set_column('D:D', 6)

        ##h-CLAT
        # Get access to the workbook and sheet
        workbook = writer.book
        worksheet = writer.sheets['h-CLAT model']
        worksheet.set_zoom(110)
        # Define our range for the color formatting
        color_range = "B2:B{}".format(number_rows+1)
        # Add a format. Light red fill with dark red text.
        format1 = workbook.add_format({'bg_color': '#FFC7CE',
                               'font_color': '#9C0006'})
        # Add a format. Green fill with dark green text.
        format2 = workbook.add_format({'bg_color': '#C6EFCE',
                               'font_color': '#006100'})

        # Highlight the bottom 5 values in Green
        worksheet.conditional_format(color_range, {'type': 'text','criteria': 'containing','value': 'Non-Sensitizer','format': format2})

        # Highlight the bottom 5 values in Red
        worksheet.conditional_format(color_range, {'type': 'text','criteria': 'containing', 'value': 'Sensitizer','format': format1})


        color_range_ad = "D2:D{}".format(number_rows+1)
        # Highlight the top 5 values in Green
        worksheet.conditional_format(color_range_ad, {'type': 'text',
                                           'criteria': 'containing',
                                           'value': 'Yes',
                                           'format': format2})
        # Highlight the bottom 5 values in Red
        worksheet.conditional_format(color_range_ad, {'type': 'text',
                                            'criteria': 'containing',
                                           'value': 'No',
                                           'format': format1})


        # APPLY CENTERING
        center = workbook.add_format({'align': 'center'})
        worksheet.set_column('B:L', None, center)
        # Set the columns widths.
        worksheet.set_column('A:A',40)
        worksheet.set_column('B:C', 12)
        worksheet.set_column('D:D', 6)

        ##KeratinoSens
        # Get access to the workbook and sheet
        workbook = writer.book
        worksheet = writer.sheets['KeratoSens model']
        worksheet.set_zoom(110)
        # Define our range for the color formatting
        color_range = "B2:B{}".format(number_rows+1)
        # Add a format. Light red fill with dark red text.
        format1 = workbook.add_format({'bg_color': '#FFC7CE',
                               'font_color': '#9C0006'})
        # Add a format. Green fill with dark green text.
        format2 = workbook.add_format({'bg_color': '#C6EFCE',
                               'font_color': '#006100'})

        # Highlight the bottom 5 values in Green
        worksheet.conditional_format(color_range, {'type': 'text','criteria': 'containing','value': 'Non-Sensitizer','format': format2})

        # Highlight the bottom 5 values in Red
        worksheet.conditional_format(color_range, {'type': 'text','criteria': 'containing', 'value': 'Sensitizer','format': format1})


        color_range_ad = "D2:D{}".format(number_rows+1)
        # Highlight the top 5 values in Green
        worksheet.conditional_format(color_range_ad, {'type': 'text',
                                           'criteria': 'containing',
                                           'value': 'Yes',
                                           'format': format2})
        # Highlight the bottom 5 values in Red
        worksheet.conditional_format(color_range_ad, {'type': 'text',
                                            'criteria': 'containing',
                                           'value': 'No',
                                           'format': format1})


        # APPLY CENTERING
        center = workbook.add_format({'align': 'center'})
        worksheet.set_column('B:L', None, center)
        # Set the columns widths.
        worksheet.set_column('A:A',40)
        worksheet.set_column('B:C', 12)
        worksheet.set_column('D:D', 6)

        ##Consensus
        # Get access to the workbook and sheet
        workbook = writer.book
        worksheet = writer.sheets['FINAL RESULT (Consensus model)']
        worksheet.set_zoom(110)
        # Define our range for the color formatting
        color_range = "B2:B{}".format(number_rows+1)
        # Add a format. Light red fill with dark red text.
        format1 = workbook.add_format({'bg_color': '#FFC7CE',
                               'font_color': '#9C0006'})
        # Add a format. Green fill with dark green text.
        format2 = workbook.add_format({'bg_color': '#C6EFCE',
                               'font_color': '#006100'})

        # Highlight the bottom 5 values in Green
        worksheet.conditional_format(color_range, {'type': 'text','criteria': 'containing','value': 'Non-Sensitizer','format': format2})

        # Highlight the bottom 5 values in Red
        worksheet.conditional_format(color_range, {'type': 'text','criteria': 'containing', 'value': 'Sensitizer','format': format1})


        color_range_ad = "D2:D{}".format(number_rows+1)
        # Highlight the top 5 values in Green
        worksheet.conditional_format(color_range_ad, {'type': 'text',
                                           'criteria': 'containing',
                                           'value': 'Yes',
                                           'format': format2})
        # Highlight the bottom 5 values in Red
        worksheet.conditional_format(color_range_ad, {'type': 'text',
                                            'criteria': 'containing',
                                           'value': 'No',
                                           'format': format1})

        # APPLY CENTERING
        center = workbook.add_format({'align': 'center'})
        worksheet.set_column('B:L', None, center)
        # Set the columns widths.
        worksheet.set_column('A:A',40)
        worksheet.set_column('B:C', 12)
        worksheet.set_column('D:D', 6)

        writer.save()

        #Send email
        # Create zip file
        f = zipfile.ZipFile('Results.zip', 'w')
        # add some files
        f.write('predskin_report/'+new_filename_xls)
        # flush and close
        f.close()

        TO = form.email.data
        py_mail("Pred-Skin Results", email_content, TO, FROM)


        return send_file(new_file_path_xls, mimetype='application/vnd.ms-excel')

    else:
        filename = None

    return render_template('home.html', form=form, filename=filename)



@application.route('/predict', methods=['POST'])

def predict():
    smiles =  request.form['smiles']



    if smiles:
        smi =  smiles
        try:
            mol = Chem.MolFromSmiles( smi )

        except:
            mol = Chem.MolFromSmiles( "c1ccccc1" )


        def parseFloat(str):
            try:
                return float(str)
            except:
                str = str.strip()
                if str.endswith("%"):
                    return float(str.strip("%").strip()) / 100
                raise Exception("Don't know how to parse %s" % str)


        #Human model
        res,  binaryproba, fname,  = mapperfunc( mol, model_binary_human )
        #LLNA model
        cls_resultLLNA, probaLLNA, fnameLLNA = mapperfunc( mol, model_binaryLLNA)

        #DRPA model
        cls_result_DPRA, proba_DPRA, fname_DPRA = mapperfunc( mol, model_DPRA )
        #h-CALT model
        cls_result_hCLAT, proba_hCLAT, fname_hCLAT = mapperfunc( mol, model_hCLAT )

        #SS Skin Sensitization LLNA model
        cls_result_SS_LLNA_mod, proba_SS_LLNA, fname_SS_LLNA = mapperfuncMorgan( mol, model_SS_LLNA_new, nBits=2048, radius=2, useFeatures=False, class1="Sensitizer (+)", class2="Non-Sensitizer (-)" )

        #SS Skin Irritation
        cls_result_SI, proba_SI, fname_SI = mapperfuncMorgan( mol, model_SI, nBits=1024, radius=2, useFeatures=False, class1="Postive (+)", class2="Negative (-)" )


        ###################### Acute_inhal ####################
        res_con = np.where(res >= 1, 1, -1)
        binaryproba_con = parseFloat(binaryproba)
        res = np.where(res>= 1, "Toxic (+)", "Non-Toxic (-)")
        fp = np.array(MACCSkeys.GenMACCSKeys(mol)).reshape(1, -1)
        with bz2.BZ2File('model/Dados_adicionais_acute_inhal_MACCS.pbz2', 'rb') as f:
            hclat=pickle.load(f)
        fps=np.vstack(hclat['fpDF']['FingerPrint'].values)
        hclat_AD_limit = (np.average(hclat['applM'])-(1*(np.std(hclat['applM']))))
        hclat_K= hclat["k"]
        limit_Acute_inhal = np.round(hclat_AD_limit ,2)
        hclat_AD = getNeighborsDitance(fps, (np.array(fp).reshape(1, -1)) , hclat_K)[0]
        AD_Acute_inhal= np.round(hclat_AD, 2)
        #Variable ACute
        AD_acute= np.where(hclat_AD >= hclat_AD_limit, "Yes", "No")
        AD_acute_num= np.where(hclat_AD >= hclat_AD_limit, 1, 0)
        acute_con= res_con * binaryproba_con
        acute_con_ad= res_con * binaryproba_con * AD_acute_num
        ###################### Acute_inhal ####################

        ################## #### Eye_irrit ####################
        cls_resultLLNA_con = np.where(cls_resultLLNA >= 1, 1, -1)
        probaLLNA_con = parseFloat(probaLLNA)
        cls_resultLLNA  = np.where(cls_resultLLNA  >= 1, "Toxic (+)", "Non-Toxic (-)")
        with bz2.BZ2File('model/Dados_adicionais_eye_irrit_MACCS.pbz2', 'rb') as f:
            hclat=pickle.load(f)

        fps=np.vstack(hclat['fpDF']['FingerPrint'].values)
        hclat_AD_limit = (np.average(hclat['applM'])-(1*(np.std(hclat['applM']))))
        hclat_K= hclat["k"]
        limit_eye_irrit = np.round(hclat_AD_limit ,2)
        hclat_AD = getNeighborsDitance(fps, (np.array(fp).reshape(1, -1)) , hclat_K)[0]
        AD_eye_irrit= np.round(hclat_AD, 2)
        #Variable ACute
        AD_eye = np.where(hclat_AD >= hclat_AD_limit, "Yes", "No")
        AD_eye_num= np.where(hclat_AD >= hclat_AD_limit, 1, 0)
        eye_con= cls_resultLLNA_con * probaLLNA_con
        eye_con_ad = cls_resultLLNA_con * probaLLNA_con * AD_eye_num
        ###################### END Eye_irrit ####################

        ###################### Acute dermal ####################
        cls_result_DPRA_con = np.where(cls_result_DPRA >= 1, 1, -1)
        proba_DPRA_con = parseFloat(proba_DPRA)
        cls_result_DPRA  = np.where(cls_result_DPRA  >= 1, "Toxic (+)", "Non-Toxic (-)")
        with bz2.BZ2File('model/Dados_adicionais_acute_dermal_MACCS.pbz2', 'rb') as f:
            hclat=pickle.load(f)

        fps=np.vstack(hclat['fpDF']['FingerPrint'].values)
        hclat_AD_limit = (np.average(hclat['applM'])-(1*(np.std(hclat['applM']))))
        hclat_K= hclat["k"]
        limit_Acute_dermal = np.round(hclat_AD_limit ,2)
        hclat_AD = getNeighborsDitance(fps, (np.array(fp).reshape(1, -1)) , hclat_K)[0]
        AD_Acute_dermal= np.round(hclat_AD, 2)
        #Variable ACute
        AD_dermal = np.where(hclat_AD >= hclat_AD_limit, "Yes", "No")
        AD_dermal_num= np.where(hclat_AD >= hclat_AD_limit, 1, 0)
        dermal_con= cls_result_DPRA_con * proba_DPRA_con
        dermal_con_ad = cls_result_DPRA_con * proba_DPRA_con * AD_dermal_num
        ###################### Acute dermal ####################

        ######################  Acute oral ####################
        cls_result_hCLAT_con = np.where(cls_result_hCLAT >= 1, 1, -1)
        proba_hCLAT_con = parseFloat(proba_hCLAT)
        cls_result_hCLAT  = np.where(cls_result_hCLAT  >= 1, "Toxic (+)", "Non-Toxic (-)")
        with bz2.BZ2File('model/Dados_adicionais_acute_oral_MACCS.pbz2', 'rb') as f:
            hclat=pickle.load(f)

        fps=np.vstack(hclat['fpDF']['FingerPrint'].values)
        hclat_AD_limit = (np.average(hclat['applM'])-(1*(np.std(hclat['applM']))))
        hclat_K= hclat["k"]
        limit_Acute_oral = np.round(hclat_AD_limit ,2)
        hclat_AD = getNeighborsDitance(fps, (np.array(fp).reshape(1, -1)) , hclat_K)[0]
        AD_Acute_oral= np.round(hclat_AD, 2)
        #Variable ACute
        AD_oral = np.where(hclat_AD >= hclat_AD_limit, "Yes", "No")
        AD_oral_num= np.where(hclat_AD >= hclat_AD_limit, 1, 0)
        oral_con= cls_result_hCLAT_con * proba_hCLAT_con
        oral_con_ad = cls_result_hCLAT_con * proba_hCLAT_con * AD_oral_num
        ######################  Acute oral ####################

        ######################  Skin Sensitization ####################
        nBits = 2048
        radius=2
        useFeatures= False
        from rdkit.Chem.AllChem import GetMorganFingerprintAsBitVect
        from rdkit.Chem import Descriptors


        fp = np.array(GetMorganFingerprintAsBitVect(mol,nBits=nBits,radius=radius, useFeatures=useFeatures)).reshape(1, -1)
        cls_result_SS_LLNA_con = np.where(cls_result_SS_LLNA_mod >= 'Sensitizer (+)', 1, -1)
        proba_SS_LLNA_con = parseFloat(proba_SS_LLNA)
        cls_result_SS_LLNA= cls_result_SS_LLNA_mod
        #cls_result_SS_LLNA  = np.where(cls_result_SS_LLNA_mod  >= 1, "Sensitizer (+)", "Non-Sensitizer (-)")
        with bz2.BZ2File('model/Dados_adicionais_Skin_Sens.pbz2', 'rb') as f:
            SS_LLNA=pickle.load(f)

        fps=np.vstack(SS_LLNA['fpDF']['FingerPrint'].values)
        SS_LLNA_AD_limit = (np.average(SS_LLNA['applM'])-(1*(np.std(SS_LLNA['applM']))))
        SS_LLNA_K= SS_LLNA["k"]
        limit_SS_LLNA = np.round(SS_LLNA_AD_limit ,2)
        SS_LLNA_AD = getNeighborsDitance(fps, (np.array(fp).reshape(1, -1)) , SS_LLNA_K)[0]
        AD_SS_LLNA_va= np.round(SS_LLNA_AD, 2)
        #Variable ACute
        AD_SS_LLNA = np.where(AD_SS_LLNA_va >= limit_SS_LLNA, "Yes", "No")
        AD_SS_LLNA_num= np.where(AD_SS_LLNA_va >= limit_SS_LLNA, 1, 0)
        SS_LLNA_con= cls_result_SS_LLNA_con * proba_SS_LLNA_con
        SS_LLNA_con_ad = cls_result_SS_LLNA_con * proba_SS_LLNA_con * AD_SS_LLNA_num
        ######################  Acute oral ####################

        ######################  Skin Irritation ####################
        nBits = 1024
        radius=2
        useFeatures= False
        from rdkit.Chem.AllChem import GetMorganFingerprintAsBitVect
        from rdkit.Chem import Descriptors

        fp = np.array(GetMorganFingerprintAsBitVect(mol,nBits=nBits,radius=radius, useFeatures=useFeatures)).reshape(1, -1)
        cls_result_SI_con = np.where(cls_result_SI>= 'Positive (+)', 1, -1)
        proba_SI_con = parseFloat(proba_SI)

        with bz2.BZ2File('model/Dados_adicionais_skin_irrit_morgan.pbz2', 'rb') as f:
            SI=pickle.load(f)

        fps=np.vstack(SI['fpDF']['FingerPrint'].values)
        SI_AD_limit = (np.average(SI['applM'])-(1*(np.std(SI['applM']))))
        SI_K= SI["k"]
        limit_SI = np.round(SI_AD_limit ,2)
        SI_AD = getNeighborsDitance(fps, (np.array(fp).reshape(1, -1)) , SI_K)[0]
        AD_SI_va= np.round(SI_AD, 2)
        #Variable ACute
        AD_SI = np.where(AD_SI_va >= limit_SI, "Yes", "No")
        AD_SI_num= np.where(AD_SI_va >= limit_SI, 1, 0)
        SI_con= cls_result_SI_con * proba_SI_con
        SI_con_ad = cls_result_SI_con * proba_SI_con * AD_SI_num
        ######################  Consensus ####################

        consensus =  acute_con+ eye_con+ dermal_con + oral_con+ SS_LLNA_con + SI_con 
        consensus = np.where(consensus  >= 0, "Toxic (+)", "Non-Toxic (-)")
        consensus_proba = np.round((abs(acute_con)+ abs(eye_con) + abs(dermal_con) + abs(oral_con) + abs(SS_LLNA_con + SI_con))/6*100,0)
        consensus_ad = acute_con_ad + eye_con_ad + dermal_con_ad + oral_con_ad+ SS_LLNA_con_ad + SI_con_ad
        consensus_ad = np.where( consensus_ad >= 0, "Toxic (+)", "Non-Toxic (-)")
        consensus_ad_proba = np.round(((abs(acute_con_ad) + abs(eye_con_ad) + abs(dermal_con_ad) + abs(oral_con_ad) + abs(SS_LLNA_con_ad) + abs(SI_con_ad))/6)*100,0)

        return render_template( "result.html", res = res, binaryproba= binaryproba, fname=fname, AD_acute= AD_acute, limit_Acute_inhal = limit_Acute_inhal, AD_Acute_inhal= AD_Acute_inhal,
                                    cls_eye = cls_resultLLNA, proba_eye= probaLLNA, fname_eye =fnameLLNA, AD_eye= AD_eye, limit_eye_irrit = limit_eye_irrit, AD_eye_irrit =AD_eye_irrit,
                                    cls_dermal = cls_result_DPRA, proba_dermal= proba_DPRA, fname_dermal=fname_DPRA, AD_dermal = AD_dermal, limit_Acute_dermal = limit_Acute_dermal, AD_Acute_dermal =AD_Acute_dermal,
                                    cls_SS_LLNA = cls_result_SS_LLNA, proba_SS_LLNA= proba_SS_LLNA, fname_SS_LNNA=fname_SS_LLNA, AD_SS_LLNA = AD_SS_LLNA, limit_SS_LLNA = limit_SS_LLNA, AD_SS_LLNA_va =AD_SS_LLNA_va,
                                    cls_result_SI = cls_result_SI, proba_SI= proba_SI, fname_SI=fname_SI, AD_SI = AD_SI, limit_SI = limit_SI, AD_SI_va = AD_SI_va,
                                    cls_result_oral = cls_result_hCLAT, proba_oral= proba_hCLAT, fname_oral=fname_hCLAT, AD_oral = AD_oral, limit_Acute_oral = limit_Acute_oral, AD_Acute_oral =AD_Acute_oral,
                                    consensus= consensus, consensus_proba =str(consensus_proba)+"%", consensus_ad =consensus_ad, consensus_ad_proba= str(consensus_ad_proba)+"%")
    else:
        return 'Please go back and enter your name...'




@application.route('/database')
def database():
    # generate some file name
    # save the file in the `database_reports` folder used below
    return render_template('database.html', filename= new_filename_xls)

@application.route('/database_download/<filename>')
def database_download(filename):
    return send_from_directory('predskin_report/', new_filename_xls)



  ################################################ mobile



@application.route( "/mobile", methods=['GET', 'POST'])
def mobile():
    return render_template( "home_mobile.html" , form=form)


@application.route( "/mobilehelp", methods=['GET', 'POST'])
def mobilehelp():
    return render_template( "help_mobile.html" )


@application.route('/predictmobile', methods=['POST'])

def predictmobile():
    smiles =  request.form['smiles']
    actcls = { 0: "Non-Sensitizer", 1: "Sensitizer" }
    actclsmult = { 0: "Non-Sensitizer", 1: "Weak/Moderate", 2: "strong/extreme" }

    if smiles:
        smi =  smiles
        try:
            mol = Chem.MolFromSmiles( smi )

        except:
            mol = Chem.MolFromSmiles( "c1ccccc1" )

        # get molwt, mollogp, tpsa
        #molprop = engine.molprop_calc( mol )
        # predict active / nonactive as integer and save image.
        res,  binaryproba, fname,  = mapperfunc( mol )
        res = int( res[0] )

        cls_resultLLNA, cls_result_multLNNA, probaLLNA, probamultLLNA, fnameLLNA = mapperfuncLLNNA( mol )
        cls_resultLLNA = int( cls_resultLLNA[0] )
        cls_result_multLNNA = int( cls_result_multLNNA[0] )
        cls_result_DPRA, proba_DPRA
        cls_result_DPRA = int( cls_result_DPRA[0] )
        #human Cell Line Activation Test (h-CLAT)
        cls_result_hCLAT, proba_DPRA
        cls_result_DPRA = int( cls_result_DPRA[0] )



        return render_template( "result_mobile.html", res = actcls[res], binaryproba= binaryproba, fname=fname, cls_resultLLNA = actcls[cls_resultLLNA], cls_result_multLNNA  = actclsmult[cls_result_multLNNA], probaLLNA= probaLLNA, probamultLLNA = probamultLLNA, fnameLLNA =fnameLLNA )
    else:
        return 'Please go back and enter your name...'





if __name__ == "__main__":
    application.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5003)), debug = True)
