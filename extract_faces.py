import os
import numpy as np
import glob
import torch
from tqdm import tqdm
from facenet_pytorch import MTCNN
from dataprepare import FaceExtractor, get_metadata
import argparse
import json
import re


def main(args):

    ROOT_DIR = '/home/songzhu/PycharmProjects/DeepFake/www.kaggle.com/c/16880/'
    TRAIN_DIR = '/home/songzhu/PycharmProjects/DeepFake/www.kaggle.com/c/16880/datadownload/part_00/dfdc_train_part_0'
    TMP_DIR = '/home/songzhu/PycharmProjects/DeepFake/www.kaggle.com/c/16880/datadownload/part_00/dfdc_train_part_00/faces'
    ZIP_NAME = 'dfdc_train_part_00.zip'
    METADATA_PATH = TRAIN_DIR + 'metadata.json'

    TRAIN_DIR = np.array(os.listdir('/data/songzhu/deepfake/'))
    TRAIN_DIR = TRAIN_DIR[np.where([len(x) <= 2 for x in TRAIN_DIR])[0].astype(np.int32)]

    train_dir = ['/data/songzhu/deepfake/00/dfdc_train_part_0']
    face_dir = ['/data/songzhu/deepfake/00/faces']
    for foldername in TRAIN_DIR:
        if foldername == '00':
            continue
        temp = "/data/songzhu/deepfake/" + foldername + '/dfdc_train_part_' + foldername.lstrip('0')
        train_dir.append(temp)
        temp = "/data/songzhu/deepfake/" + foldername + "/faces/"
        face_dir.append(temp)

    SCALE = 0.8
    N_FRAMES = 50

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print('Running on device: {}, using {} GPUs'.format(args.gpu, args.n_gpus))

    # Load face detector
    face_detector = MTCNN(margin=80, keep_all=True, factor=0.8, device=device).eval()
    # Define face extractor
    face_extractor = FaceExtractor(detector=face_detector, n_frames=N_FRAMES, resize=SCALE)
    # Get the paths of all train videos
    count = 0
    # To Do:
    # Sample differently to get rid off the imbalance problem

    if args.batch == 1:
        train_dir = train_dir[0:12]
        face_dir = face_dir[0:12]
    elif args.batch == 2:
        train_dir = train_dir[12:24]
        face_dir = face_dir[12:24]
    elif args.batch == 3:
        train_dir = train_dir[24:36]
        face_dir = face_dir[24:36]
    else:
        train_dir = train_dir[36:]
        face_dir = face_dir[36:]

    # train_dir = [train_dir[1]]
    # count = 1

    for trainfiles in train_dir:

        meta_file = glob.glob(os.path.join(trainfiles, '*.json'))[0]
        with open(meta_file, "r") as f:
            metadata = json.load(f)
        all_train_videos = glob.glob(os.path.join(trainfiles, '*.mp4'))

        print(trainfiles)

        with torch.no_grad():
            for path in tqdm(all_train_videos, ascii=True, ncols=50):

                mp4file = re.findall("\w+.mp4", path)
                label = metadata[mp4file[0]]['label']

                if label == 'FAKE':
                    N_FRAMES = 0
                else:
                    N_FRAMES = 50

                file_name = path.split('/')[-1]

                face_dir = trainfiles[0:25] + "/faces"
                if not os.path.isdir(face_dir):
                    os.mkdir(face_dir)

                save_dir = os.path.join(face_dir, file_name.split(".")[0])
                if not os.path.exists(save_dir):
                    os.makedirs(save_dir)
                # Detect all faces appear in the video and save t   hem.
                face_extractor.n_frames = N_FRAMES
                face_extractor(path, save_dir)

                # import matplotlib.pyplot as plt
                # plt.imshow(plt.imshow(np.transpose(face_extractor(path, save_dir)[0].numpy().squeeze(), [1, 2, 0])))
        count += 1

    return 0


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', default=6, help='delimited list input of GPUs', type=int)
    parser.add_argument('--n_gpus', default=1, help="num of GPUS to use", type=int)
    parser.add_argument('--batch', default=1, help="data batch to process", type=int)
    args = parser.parse_args()

    opt_gpus = [i for i in range(args.gpu, (args.gpu + args.n_gpus))]
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(str(x) for x in opt_gpus)

    main(args)

