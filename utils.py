import cv2
import torch
import numpy as np
from pathlib import Path
import dlib
import matplotlib.pyplot as plt
import os
from torchvision import transforms
import torch.nn as nn
from PIL import Image as pil_image


default_data_transforms = {
    'train': transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3)
    ]),
    'val': transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.5] * 3, [0.5] * 3)
    ]),
    'test': transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.5] * 3, [0.5] * 3)
    ]),
}


def get_front_face_detector():
    return dlib.get_frontal_face_detector()


def get_landmarks_predictor(path):
    if os.path.exists(path) and path.endswith('.dat'):
        return dlib.shape_predictor(path)
    else:
        raise ValueError('{} is not valid...'.format(path))


def preprocess_image(image, cuda=True):
    """
    Preprocesses the image such that it can be fed into our network.
    During this process we envoke PIL to cast it into a PIL image.
    :param image: numpy image in opencv form (i.e., BGR and of shape
    :return: pytorch tensor of shape [1, 3, image_size, image_size], not
    necessarily casted to cuda
    """
    # Revert from BGR
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # Preprocess using the preprocessing function used during training and
    # casting it to PIL image
    preprocess = default_data_transforms['test']
    preprocessed_image = preprocess(pil_image.fromarray(image))
    # Add first dimension as the network expects a batch
    preprocessed_image = preprocessed_image.unsqueeze(0)
    if cuda:
        preprocessed_image = preprocessed_image.cuda()
    return preprocessed_image


def predict_with_model(image, model, post_function=nn.Softmax(dim=1),
                       cuda=True):
    """
    Predicts the label of an input image. Preprocesses the input image and
    casts it to cuda if required
    :param image: numpy image
    :param model: torch model with linear layer at the end
    :param post_function: e.g., softmax
    :param cuda: enables cuda, must be the same parameter as the model
    :return: prediction (1 = fake, 0 = real)
    """
    # Preprocess
    preprocessed_image = preprocess_image(image, cuda)

    # Model prediction
    output = model(preprocessed_image)
    output = post_function(output)

    # Cast to desired
    max_output, prediction = torch.max(output, 1)  # argmax
    prediction = float(prediction.cpu().numpy())

    return int(prediction), max_output


def parse_vid(video_path):
    vidcap = cv2.VideoCapture(video_path,0)
    frame_num = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = vidcap.get(cv2.CAP_PROP_FPS) # Frames per seconds
    width = np.int32(vidcap.get(cv2.CAP_PROP_FRAME_WIDTH)) # float
    height = np.int32(vidcap.get(cv2.CAP_PROP_FRAME_HEIGHT))  # float
    imgs = []
    while True:
        hasFrames, image = vidcap.read()
        imname ='image'
        if hasFrames:
            #image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            imgs.append(image)
            print(image.shape)
            cv2.imwrite('images/' + imname + ".jpg", image)
        else:
            break
    vidcap.release()
    print(len(imgs),frame_num)
    if len(imgs) != frame_num:
        frame_num = len(imgs)
    return imgs, frame_num, fps, width, height


def get_boundingbox(face, width, height, scale=1.3, minsize=None):
    x1 = face.left()
    y1 = face.top()
    x2 = face.right()
    y2 = face.bottom()
    size_bb = int(max(x2 - x1, y2 - y1) * scale)
    if minsize:
        if size_bb < minsize:
            size_bb = minsize
    center_x, center_y = (x1 + x2) // 2, (y1 + y2) // 2

    # Check for out of bounds, x-y top left corner
    x1 = max(int(center_x - size_bb // 2), 0)
    y1 = max(int(center_y - size_bb // 2), 0)
    # Check for too big bb size for given x, y
    size_bb = min(width - x1, size_bb)
    size_bb = min(height - y1, size_bb)

    return x1, y1, size_bb


def load_and_preprocess_image(image_filename, output_image_size, face_detector):
    image = cv2.imread(image_filename)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    cropped_image = get_face_crop(face_detector, image)
    if cropped_image is None:
        return None
    
    resized_image = cv2.resize(cropped_image, (output_image_size, output_image_size))
    return resized_image


def get_face_crop(face_detector, image):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    faces = face_detector(gray, 1)
    
    height, width = image.shape[:2]
    
    if len(faces) == 0:
        return None
    else:
        face = faces[0]
        x, y, size = get_boundingbox(face, width, height)
        cropped_face = image[y:y + size, x:x + size]
        return cropped_face
    

def visualize_metrics(records, extra_metric, name):
    fig, axes = plt.subplots(nrows=1, ncols=4, figsize=(15, 6))
    axes[0].plot(list(range(len(records.train_losses))), records.train_losses, label='train')
    axes[0].plot(list(range(len(records.train_losses_wo_dropout))), records.train_losses_wo_dropout, label='train w/o dropout')
    axes[0].plot(list(range(len(records.base_val_losses))), records.base_val_losses, label='base_val')
    axes[0].plot(list(range(len(records.augment_val_losses))), records.augment_val_losses, label='augment_val')
    axes[0].set_title('loss')
    axes[0].legend()
    
    axes[1].plot(list(range(len(records.train_accs))), records.train_accs, label='train')
    axes[1].plot(list(range(len(records.train_accs_wo_dropout))), records.train_accs_wo_dropout, label='train w/o dropout')
    axes[1].plot(list(range(len(records.base_val_accs))), records.base_val_accs, label='base_val')
    axes[1].plot(list(range(len(records.augment_val_accs))), records.augment_val_accs, label='augment_val')
    axes[1].axhline(y=0.5, color='g', ls='--')
    axes[1].axhline(y=0.667, color='r', ls='--')
    axes[1].set_title('acc')
    axes[1].legend()
    
    axes[2].plot(list(range(len(records.train_custom_metrics))), records.train_custom_metrics, label='train')
    axes[2].plot(list(range(len(records.train_custom_metrics_wo_dropout))), records.train_custom_metrics_wo_dropout, label='train w/o dropout')
    axes[2].plot(list(range(len(records.base_val_custom_metrics))), records.base_val_custom_metrics, label='base_val')
    axes[2].plot(list(range(len(records.augment_val_custom_metrics))), records.augment_val_custom_metrics, label='augment_val')
    axes[2].axhline(y=0.5, color='g', ls='--')
    axes[2].axhline(y=0.5, color='r', ls='--')
    axes[2].set_title(f'{extra_metric.__name__}')
    axes[2].legend()
    
    axes[3].plot(list(range(len(records.lrs))), records.lrs)
    _ = axes[3].set_title('lr')
    plt.tight_layout()
    plt.savefig(name, format='png')
    
    
def display_predictions_on_image(model, precomputed_cached_path, val_iter, name):
    # val
    model.eval()
    
    data = next(val_iter)
    
    inputs = data['image']
    labels = data['label'].view(-1)
    filenames = data['filename']
    
    inputs = inputs.cuda(device=0)
    labels = labels.cuda(device=0)
    
    with torch.no_grad():
        outputs = model(inputs)
        outputs_predicbilty = torch.nn.functional.softmax(outputs, dim=1)
        assert len(outputs_predicbilty) == len(outputs), f'proba shape: {len(outputs_predicbilty)}'
        
        _, predicted = torch.max(outputs.data, 1)
    
    nrows = int(len(inputs) ** 0.5)
    ncols = int(np.ceil(len(inputs) / nrows))
    
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(30, 40))
    step = 0
    for i in range(nrows):
        for j in range(ncols):
            image_id = Path(filenames[step]).stem
            face_crop = precomputed_cached_path / f'processed_{image_id}.npy'
            face_crop = np.load(face_crop)
            axes[i, j].set_title(f'{outputs_predicbilty[step][0]:.2f},{outputs_predicbilty[step][1]:.2f}|{predicted[step]}|{labels[step]}')
            axes[i, j].imshow(face_crop)
            step += 1
            if step == len(inputs):
                break
    plt.title('predicted_proba real, fake | prediction | label (0: real 1: fake)')
    plt.tight_layout()
    plt.savefig(name, format='png')


def parse_and_override_params(params):
    data_dict = {'base': 0, 'augment': 1, 'both': 2}
    
    parsed_params = params.copy()
    parsed_params['train_data'] = data_dict[params['train_data']]
    foundations.log_params(parsed_params)
    return data_dict
