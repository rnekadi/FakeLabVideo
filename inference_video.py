import numpy as np
import torch
from model import create_model
from utils import get_face_crop, parse_vid, predict_with_model, get_front_face_detector
import shutil
import os
import uuid
import datetime
from mysqlhelper import *
from PredictionInfo import *
import cv2


def pred_test(v_path, start_frame=0, end_frame=1500, cuda=False, n_frames=5, user='101'):
    unlabeled_dir = "dataset/unlabeled"
    inf_metadata = dict()

    shutil.copy(v_path, unlabeled_dir)

    v_filenames = os.listdir(unlabeled_dir)
    for v_filename in v_filenames:
        vd_path = str(unlabeled_dir + '/' + v_filename)


    """
    Predict and give result as numpy array
    """
    pred_frames = [int(round(x)) for x in np.linspace(start_frame, end_frame, n_frames)]
    predictions = []
    outputs = []
    inf_metadata = dict()

    imgs, num_frames, fps, width, height = parse_vid(vd_path)

    print(num_frames)

    front_face_detector = get_front_face_detector()

    # Load model
    # Define Hyperparams
    params = {
        'dropout': float(np.random.choice([0.8, 0.9, 0.75])),
        'use_hidden_layer': int(np.random.choice([0])),

    }

    model = create_model(bool(params['use_hidden_layer']), params['dropout'])

    model_path = 'fitted_object/best_model.pth'

    state_dict = torch.load(model_path, map_location=torch.device('cpu'))

    model.load_state_dict(state_dict, strict=False)

    model.eval()

    inf_metadata['pred_id'] = str(uuid.uuid1())

    for fid, img in enumerate(imgs):
        if fid in pred_frames:

            cropped_face = get_face_crop(front_face_detector, img)

            # Actual prediction using our model
            prediction, output = predict_with_model(cropped_face, model,
                                                        cuda=cuda)

            imname = str(inf_metadata['pred_id'] + '_' + str(fid))

            cv2.imwrite('images/' + imname + ".jpg", img)

            # ------------------------------------------------------------------

            # 'predicted_proba real, fake | prediction | label (0: real 1: fake)'

            predictions.append(prediction)
            outputs.append(output)

    plen = len(predictions)
    count_zero = 0

    for x in predictions:
        if x == 0:
            count_zero += 1

    prob = count_zero / plen

    inf_metadata['user_id'] = str(user)
    inf_metadata['time_stamp'] = str(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    inf_metadata['prob'] = str(prob)
    inf_metadata['stype'] = str('V')
    inf_metadata['fps'] = str(fps)
    inf_metadata['nframes'] = str(num_frames)

    """Write to Local MySql Database"""

    insert_sql_value = PredictionInfo(inf_metadata['pred_id'], inf_metadata['user_id'], inf_metadata['time_stamp'],
                                      inf_metadata['prob'],inf_metadata['stype'],inf_metadata['fps'],
                                      inf_metadata['nframes']);

    # create a database connection
    conn = create_connection()
    # create a new prediction
    create_prediction(conn, insert_sql_value)

    #return predictions, outputs
    print('Pred List of Frame:', predictions)
    print('Probabailities of Frame', outputs)


pred_test(v_path='dataset/test/test12.mp4', start_frame=0, end_frame=500,cuda=False, n_frames=5)


