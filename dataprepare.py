import os
import json
import cv2
from PIL import Image
import numpy as np
import glob
from tqdm import tqdm

import torch
from torch.utils.data import Dataset

from BlazeFace.blazeface import BlazeFace

import pickle as pkl


class FaceExtractor:
    def __init__(self, detector, n_frames=None, resize=None):
        """
        Parameters:
            n_frames {int} -- Total number of frames to load. These will be evenly spaced
                throughout the video. If not specified (i.e., None), all frames will be loaded.
                (default: {None})
            resize {float} -- Fraction by which to resize frames from original prior to face
                detection. A value less than 1 results in downsampling and a value greater than
                1 result in upsampling. (default: {None})
        """

        self.detector = detector
        self.n_frames = n_frames
        self.resize = resize

    def __call__(self, filename, save_dir):
        """Load frames from an MP4 video, detect faces and save the results.

        Parameters:
            filename {str} -- Path to video.
            save_dir {str} -- The directory where results are saved.
        """

        # Create video reader and find length
        v_cap = cv2.VideoCapture(filename)
        v_cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1080)
        v_cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
        v_len = int(v_cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # Pick 'n_frames' evenly spaced frames to sample
        if self.n_frames is None:
            sample = np.arange(0, v_len)
        else:
            sample = np.linspace(0, v_len - 1, self.n_frames).astype(int)

        # Loop through frames
        for j in range(v_len):
            success = v_cap.grab()
            if j in sample:
                # Load frame
                success, frame = v_cap.retrieve()
                if not success:
                    continue
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = Image.fromarray(frame)

                # Resize frame to desired size
                if self.resize is not None:
                    frame = frame.resize([int(d * self.resize) for d in frame.size])

                save_path = os.path.join(save_dir, f'{j}.png')

                self.detector([frame], save_path=save_path)
                #
                # import matplotlib.pyplot as plt
                # plt.imshow(np.transpose(self.detector([frame], save_path=save_path)[0].numpy().squeeze(), [1, 2, 0]))

        v_cap.release()


class FaceExtractor2:
    def __init__(self):
        self.temp = None


def get_metadata(root_path):

    TRAIN_DIR = np.array(os.listdir(root_path))
    TRAIN_DIR = TRAIN_DIR[np.where([len(x) <= 2 for x in TRAIN_DIR])[0].astype(np.int32)]

    TRAIN_DIR = ["21", "08", "31", "36", "39", "06", "16",
                 "40", "41", "42", "46", "47", "48", "45"]
    train_dir = []
    face_dir = []
    face_list = []
    label_list = []
    origin_list = []

    for foldername in TRAIN_DIR:

        trainfiles = root_path + "/" + foldername
        temp = trainfiles + '/dfdc_train_part_' + foldername.lstrip('0')
        if foldername == "00":
            train_dir.append('/data/songzhu/deepfake/00/dfdc_train_part_0')
        else:
            train_dir.append(temp)

        temp = trainfiles + "/faces/"
        if foldername == "00":
            face_dir.append('/data/songzhu/deepfake/00/faces')
        else:
            face_dir.append(temp)

        all_train_videos = os.listdir(trainfiles + '/faces')
        if not (foldername == '00'):
            foldername = foldername.lstrip('0')
        meta_file = glob.glob(os.path.join(trainfiles + '/dfdc_train_part_' + foldername, '*.json'))[0]
        # video_list = glob.glob(os.path.join(trainfiles + '/dfdc_train_part_' + foldername, '*.mp4'))

        with open(meta_file, 'r') as f:
            metadata = json.load(f)
        for video in all_train_videos:

            video_file = video + '.mp4'

            if not(video_file in metadata.keys()):
                label_list.append("")
                origin_list.append("")
                face_list.append("")
                continue

            face_list.append(trainfiles + '/faces' + '/'+video)
            label_list.append(metadata[video_file]['label'])
            if 'original' in metadata[video_file].keys():
                origin_list.append(metadata[video_file]['original'])
            else:
                origin_list.append("")

            # train_df = pd.DataFrame(
            #     [
            #         (video_file, metadata[video_file]['label'], metadata[video_file]['split'],
            #          metadata[video_file]['original'] if 'original' in metadata[video_file].keys() else '')
            #         for video_file in metadata.keys()
            #     ],
            #     columns=['filename', 'label', 'split', 'original']
            # )
    return np.array(face_list), np.array(label_list), np.array(origin_list)
#
# get_metadata('/data/songzhu/deepfake')


class DeepFakeFrame(Dataset):

    def __init__(self, root_path, split="Training", train_val_ratio=0.9, transform=None):

        self.labels = []
        self.origin = []
        self.split = split
        self.transform = transform
        self.root_path = root_path
        image_list = []

        video_list, label_list, origin_list = get_metadata(root_path)
        video_list = video_list[np.where(np.invert(video_list == ''))]
        label_list = label_list[np.where(np.invert(video_list == ''))]
        origin_list = origin_list[np.where(np.invert(video_list == ''))]

        train_idx = np.random.choice(range(len(label_list)), np.floor(len(label_list)*train_val_ratio).astype(np.int32),
                                     replace=False).astype(np.int32)
        val_idx = np.array(list(set(list(range(len(label_list)))) - set(list(train_idx))))

        if split == "training":
            video_list = np.array(video_list)[train_idx]
            label_list = np.array(label_list)[train_idx]
            origin_list = np.array(origin_list)[train_idx]
            print("Building Data Set")
            for i in tqdm(range(len(video_list)), ncols=100):
            # for i in tqdm(range(256*2), ncols=100):
                for frame in os.listdir(video_list[i]):
                    if label_list[i] == 'FAKE' and np.random.binomial(1, 0.75, 1) :
                        continue
                    image_list.append(video_list[i]+'/'+frame)
                    # images = cv2.cvtColor(cv2.imread(video_list[i]+'/'+frame), cv2.COLOR_BGR2RGB)
                    # images = cv2.resize(images, (150, 150))
                    # self.X.append(images)
                    self.labels.append(int(label_list[i] == 'REAL'))
                    self.origin.append(origin_list[i])
        if split == "validating":
            video_list = video_list[val_idx]
            label_list = label_list[val_idx]
            origin_list = origin_list[val_idx]
            for i in tqdm(range(len(video_list)), ncols=100):
            # for i in tqdm(range(256*50), ncols=100):
                for frame in os.listdir(video_list[i]):
                    image_list.append(video_list[i]+'/'+frame)
                    # images = cv2.cvtColor(cv2.imread(video_list[i]+'/'+frame), cv2.COLOR_BGR2RGB)
                    # images = cv2.resize(images, (150, 150))
                    # self.X.append(images)
                    self.labels.append(int(label_list[i] == 'REAL'))
                    self.origin.append(origin_list[i])
        if split == "testing":
            for i in range(len(video_list)):
                for frame in os.listdir(video_list[i]):
                    image_list.append(video_list[i]+'/'+frame)
                    # images = cv2.cvtColor(cv2.imread(video_list[i] + '/' + frame), cv2.COLOR_BGR2RGB)
                    # images = cv2.resize(images, (150, 150))
                    # self.X.append(images)
                    self.labels.append(int(label_list[i] == 'REAL'))
                    self.origin.append(origin_list[i])

        # self.X = np.array(self.X)
        # self.X = np.transpose(self.X, [0, 3, 1, 2])
        # self.X = np.vstack(self.X.reshape([-1, 150, 150, 3]))
        self.image_list = image_list
        self.labels = np.array(self.labels)
        self.origin = np.array(self.origin)
        self.X = [1000 for x in range(len(image_list))]

    def __len__(self):
        # return len(self.X)
        return len(self.image_list)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        if self.X[idx] == 1000:
            image_id = self.image_list[idx]
            img = cv2.cvtColor(cv2.imread(image_id), cv2.COLOR_BGR2RGB)
            # img = cv2.cvtColor(cv2.imread(image_id), cv2.CV_32F)
            img = cv2.resize(img, (150, 150))
            self.X[idx] = img
        else:
            img = self.X[idx]

        if self.transform is not None:
            res = self.transform(image=img)
            img = res['image']

        labels = self.labels[idx].astype(np.int32)
        origin = self.origin[idx]

        return img, labels, origin

    def __repr__(self):
        fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
        fmt_str += '    Number of datapoints: {}\n'.format(self.__len__())
        tmp = self.split
        fmt_str += '    Split: {}\n'.format(tmp)
        fmt_str += '    Root Location: {}\n'.format(self.root_path)
        tmp = '    Transforms (if any): '
        fmt_str += '{0}{1}\n'.format(tmp, self.transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        return fmt_str

    # def mean(self):
    #     return self.X.mean(axis=(0, 1), dtype=np.float64)
    #
    # def std(self):
    #     return self.X.std(axis=(0, 1), dtype=np.float64)

    def save(self, file_path):
        with open(file_path, "w") as f:
            pkl.dump(f, self)
        f.close()

# temp = DeepFakeFrame("/data/songzhu/deepfake", split = 'training', train_val_ratio=0.9, transform = None)
# print(temp)