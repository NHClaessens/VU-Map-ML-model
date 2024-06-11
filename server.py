from flask import Flask, request, jsonify
import pickle
import pandas as pd
import json
import random

app = Flask(__name__)

name = 'model-05-21--21-29-25'
model = pickle.load(open(f"sklearn_models/multilayer-distance/{name}.pkl", 'rb'))

AP_ORDER = ['NU-AP00001_distance', 'NU-AP00002_distance', 'NU-AP00003_distance', 'NU-AP00004_distance', 'NU-AP00005_distance', 'NU-AP00006_distance', 'NU-AP00007_distance', 'NU-AP00008_distance', 'NU-AP00009_distance', 'NU-AP01010_distance', 'NU-AP01011_distance', 'NU-AP01013_distance', 'NU-AP01014_distance', 'NU-AP01015_distance', 'NU-AP01016_distance', 'NU-AP01017_distance', 'NU-AP01018_distance', 'NU-AP01019_distance', 'NU-AP01020_distance', 'NU-AP01021_distance', 'NU-AP01022_distance', 'NU-AP01023_distance', 'NU-AP01024_distance', 'NU-AP01025_distance', 'NU-AP01026_distance', 'NU-AP01027_distance', 'NU-AP01028_distance', 'NU-AP01029_distance', 'NU-AP02030_distance', 'NU-AP02031_distance', 'NU-AP02032_distance', 'NU-AP02033_distance', 'NU-AP02034_distance', 'NU-AP02035_distance', 'NU-AP02036_distance', 'NU-AP02037_distance', 'NU-AP02038_distance', 'NU-AP02039_distance', 'NU-AP02040_distance', 'NU-AP02041_distance', 'NU-AP02042_distance', 'NU-AP02043_distance', 'NU-AP02044_distance', 'NU-AP02045_distance', 'NU-AP02046_distance', 'NU-AP02047_distance', 'NU-AP02048_distance', 'NU-AP02049_distance', 'NU-AP02050_distance', 'NU-AP02051_distance', 'NU-AP02052_distance', 'NU-AP02053_distance', 'NU-AP02054_distance', 'NU-AP02055_distance', 'NU-AP02056_distance', 'NU-AP02057_distance', 'NU-AP02058_distance', 'NU-AP02059_distance', 'NU-AP02060_distance', 'NU-AP02061_distance', 'NU-AP03062_distance', 'NU-AP03063_distance', 'NU-AP03064_distance', 'NU-AP03065_distance', 'NU-AP03066_distance', 'NU-AP03067_distance', 'NU-AP03068_distance', 'NU-AP03069_distance', 'NU-AP03070_distance', 'NU-AP03071_distance', 'NU-AP03072_distance', 'NU-AP03073_distance', 'NU-AP03074_distance', 'NU-AP03075_distance', 'NU-AP03076_distance', 'NU-AP03077_distance', 'NU-AP03078_distance', 'NU-AP03079_distance', 'NU-AP03080_distance', 'NU-AP03081_distance', 'NU-AP03082_distance', 'NU-AP03083_distance', 'NU-AP03084_distance', 'NU-AP03085_distance', 'NU-AP03086_distance', 'NU-AP03087_distance', 'NU-AP04088_distance', 'NU-AP04089_distance', 'NU-AP04090_distance', 'NU-AP04091_distance', 'NU-AP04092_distance', 'NU-AP04093_distance', 'NU-AP04094_distance', 'NU-AP04095_distance', 'NU-AP04096_distance', 'NU-AP04097_distance', 'NU-AP04098_distance', 'NU-AP04099_distance', 'NU-AP04100_distance', 'NU-AP04101_distance', 'NU-AP04102_distance', 'NU-AP04103_distance', 'NU-AP04104_distance', 'NU-AP04105_distance', 'NU-AP04106_distance', 'NU-AP04107_distance', 'NU-AP04108_distance', 'NU-AP04109_distance', 'NU-AP04110_distance', 'NU-AP04111_distance', 'NU-AP04112_distance', 'NU-AP04113_distance', 'NU-AP04114_distance', 'NU-AP04333_distance', 'NU-AP05115_distance', 'NU-AP05116_distance', 'NU-AP05117_distance', 'NU-AP05118_distance', 'NU-AP05119_distance', 'NU-AP05120_distance', 'NU-AP05121_distance', 'NU-AP05122_distance', 'NU-AP05123_distance', 'NU-AP05124_distance', 'NU-AP05125_distance', 'NU-AP05126_distance', 'NU-AP05127_distance', 'NU-AP05128_distance', 'NU-AP05129_distance', 'NU-AP05130_distance', 'NU-AP05131_distance', 'NU-AP05132_distance', 'NU-AP05133_distance', 'NU-AP05134_distance', 'NU-AP05135_distance', 'NU-AP05136_distance', 'NU-AP05137_distance', 'NU-AP05138_distance', 'NU-AP05139_distance', 'NU-AP05140_distance', 'NU-AP05326_distance', 'NU-AP05327_distance', 'NU-AP05328_distance', 'NU-AP06141_distance', 'NU-AP06142_distance', 'NU-AP06143_distance', 'NU-AP06144_distance', 'NU-AP06145_distance', 'NU-AP06146_distance', 'NU-AP06147_distance', 'NU-AP06148_distance', 'NU-AP06149_distance', 'NU-AP06150_distance', 'NU-AP06151_distance', 'NU-AP06152_distance', 'NU-AP06153_distance', 'NU-AP06154_distance', 'NU-AP06155_distance', 'NU-AP06156_distance', 'NU-AP06157_distance', 'NU-AP06158_distance', 'NU-AP06159_distance', 'NU-AP06160_distance', 'NU-AP06161_distance', 'NU-AP06162_distance', 'NU-AP06163_distance', 'NU-AP06164_distance', 'NU-AP06165_distance', 'NU-AP06166_distance', 'NU-AP06167_distance', 'NU-AP06168_distance', 'NU-AP06169_distance', 'NU-AP06170_distance', 'NU-AP06329_distance', 'NU-AP06330_distance', 'NU-AP06331_distance', 'NU-AP06332_distance', 'NU-AP07171_distance', 'NU-AP07172_distance', 'NU-AP07173_distance', 'NU-AP07174_distance', 'NU-AP07175_distance', 'NU-AP07176_distance', 'NU-AP07177_distance', 'NU-AP07178_distance', 'NU-AP07179_distance', 'NU-AP07180_distance', 'NU-AP07181_distance', 'NU-AP07182_distance', 'NU-AP07183_distance', 'NU-AP07184_distance', 'NU-AP07185_distance', 'NU-AP07186_distance', 'NU-AP07187_distance', 'NU-AP07188_distance', 'NU-AP07189_distance', 'NU-AP07190_distance', 'NU-AP07191_distance', 'NU-AP07192_distance', 'NU-AP07193_distance', 'NU-AP07194_distance', 'NU-AP07292_distance', 'NU-AP07293_distance', 'NU-AP07294_distance', 'NU-AP07295_distance', 'NU-AP07296_distance', 'NU-AP07297_distance', 'NU-AP07298_distance', 'NU-AP07299_distance', 'NU-AP07300_distance', 'NU-AP07301_distance', 'NU-AP08195_distance', 'NU-AP08196_distance', 'NU-AP08197_distance', 'NU-AP08198_distance', 'NU-AP08199_distance', 'NU-AP08200_distance', 'NU-AP08201_distance', 'NU-AP08202_distance', 'NU-AP08203_distance', 'NU-AP08204_distance', 'NU-AP08205_distance', 'NU-AP08206_distance', 'NU-AP08207_distance', 'NU-AP08208_distance', 'NU-AP08209_distance', 'NU-AP08210_distance', 'NU-AP08211_distance', 'NU-AP08212_distance', 'NU-AP08213_distance', 'NU-AP08302_distance', 'NU-AP08304_distance', 'NU-AP08305_distance', 'NU-AP08306_distance', 'NU-AP08307_distance', 'NU-AP08308_distance', 'NU-AP09214_distance', 'NU-AP09215_distance', 'NU-AP09216_distance', 'NU-AP09217_distance', 'NU-AP09218_distance', 'NU-AP09219_distance', 'NU-AP09220_distance', 'NU-AP09221_distance', 'NU-AP09222_distance', 'NU-AP09223_distance', 'NU-AP09224_distance', 'NU-AP09225_distance', 'NU-AP09226_distance', 'NU-AP09227_distance', 'NU-AP09228_distance', 'NU-AP09229_distance', 'NU-AP09230_distance', 'NU-AP09231_distance', 'NU-AP09232_distance', 'NU-AP09233_distance', 'NU-AP09234_distance', 'NU-AP09309_distance', 'NU-AP09310_distance', 'NU-AP09311_distance', 'NU-AP09312_distance', 'NU-AP09313_distance', 'NU-AP10235_distance', 'NU-AP10236_distance', 'NU-AP10237_distance', 'NU-AP10238_distance', 'NU-AP10239_distance', 'NU-AP10240_distance', 'NU-AP10241_distance', 'NU-AP10242_distance', 'NU-AP10243_distance', 'NU-AP10244_distance', 'NU-AP10245_distance', 'NU-AP10246_distance', 'NU-AP10247_distance', 'NU-AP10248_distance', 'NU-AP10249_distance', 'NU-AP10250_distance', 'NU-AP10251_distance', 'NU-AP10252_distance', 'NU-AP10314_distance', 'NU-AP10315_distance', 'NU-AP10316_distance', 'NU-AP10317_distance', 'NU-AP10318_distance', 'NU-AP11253_distance', 'NU-AP11254_distance', 'NU-AP11255_distance', 'NU-AP11256_distance', 'NU-AP11257_distance', 'NU-AP11258_distance', 'NU-AP11259_distance', 'NU-AP11260_distance', 'NU-AP11261_distance', 'NU-AP11262_distance', 'NU-AP11263_distance', 'NU-AP11264_distance', 'NU-AP11265_distance', 'NU-AP11266_distance', 'NU-AP11267_distance', 'NU-AP11268_distance', 'NU-AP11269_distance', 'NU-AP11270_distance', 'NU-AP11271_distance', 'NU-AP11319_distance', 'NU-AP11320_distance', 'NU-AP11321_distance', 'NU-AP11322_distance', 'NU-AP12272_distance', 'NU-AP12273_distance', 'NU-AP12274_distance', 'NU-AP12275_distance', 'NU-AP12276_distance', 'NU-AP12278_distance', 'NU-AP12279_distance', 'NU-AP12280_distance', 'NU-AP12281_distance', 'NU-AP12282_distance', 'NU-AP12283_distance', 'NU-AP12284_distance', 'NU-AP12285_distance', 'NU-AP12323_distance', 'NU-AP12324_distance', 'NU-AP12325_distance']

@app.route('/predict/default', methods=['POST'])
def predict():
    data = request.json

    print("received", data)
    
    return jsonify({'x': 10, 'y': 20, 'z': 30})

@app.route('/predict/options', methods=['GET'])
def get_options():
    return jsonify([
        {
            'title': 'default',
            'endpoint': 'predict/default'
        },
        {
            'title': 'random',
            'endpoint': 'predict/random'
        },
        {
            'title': 'test3',
            'endpoint': 'predict/test'
        },
        {
            'title': 'test4',
            'endpoint': 'predict/test'
        },
    ])

@app.route('/predict/random', methods=['POST'])
def test():
    return jsonify({
        'x': random.uniform(2, 68), 
        'y': random.uniform(0, 24.6), 
        'z': random.uniform(2, 68)
    })



if __name__ == '__main__':
    app.run(debug=True, host="0.0.0.0")