from SCNN_Lane_Detection.model import SCNN
from SCNN_Lane_Detection.utils.transforms import *
import urllib.request
import tempfile
import os.path

# Global

weightPath = ''
p_threshold = 0.5
# Change this if you are not using pre-trained model
# --------------------------------------------------------
mean = (0.3598, 0.3653, 0.3662)  # CULane mean, std
std = (0.2573, 0.2663, 0.2756)

trained_model_width = 800
trained_model_height = 288
# --------------------------------------------------------
# End Global

net = SCNN(input_size=(trained_model_width, trained_model_height), pretrained=False)
transform_img = Resize((trained_model_width, trained_model_height))
transform_to_net = Compose(ToTensor(), Normalize(mean=mean, std=std))


def init(val):
    global weightPath
    weightPath = val


def predictThreshold(val):
    global p_threshold
    p_threshold = val


def predictCpu(img):
    if isWeightExists():
        loadModel()
        img, x = transformImage(img)
        seg_pred, exist_pred = net(x)[:2]
        img, lane_img = drawLane(img, seg_pred, exist_pred)

        return img, lane_img

    raise ValueError('The weight does not exists, please provide weight path in init()')


def predictGpu(img):
    if isWeightExists():

        if torch.cuda.is_available():
            net.cuda()

            loadModel()
            img, x = transformImage(img)
            seg_pred, exist_pred = net(x.cuda())[:2]
            img, lane_img = drawLane(img, seg_pred, exist_pred)

            return img, lane_img

        raise ValueError('The Library unable to detect CUDA '
                         'on this device, please use CPU instead.')

    raise ValueError('The weight does not exists, please provide weight path in init()')


def demo(url, gpu=False):
    file = tempfile.gettempdir() + '/lane_demo.jpg'
    downloadDemoFile(file, url)
    img = cv2.imread(file)
    if gpu:
        return predictGpu(img)
    else:
        return predictCpu(img)


def downloadDemoFile(file, url):
    urllib.request.urlretrieve(url, file)


def getAddWeight(img, lane_img):
    return cv2.addWeighted(src1=lane_img, alpha=0.8, src2=img, beta=1., gamma=0.)


def loadModel():
    save_dict = torch.load(weightPath)
    net.load_state_dict(save_dict['net'])
    net.eval()


def transformImage(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = transform_img({'img': img})['img']
    x = transform_to_net({'img': img})['img']
    x.unsqueeze_(0)
    return img, x


def drawLane(img, seg_pred, exist_pred):
    seg_pred = seg_pred.detach().cpu().numpy()
    exist_pred = exist_pred.detach().cpu().numpy()
    seg_pred = seg_pred[0]

    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    lane_img = np.zeros_like(img)
    color = np.array([[255, 125, 0], [0, 255, 0], [0, 0, 255], [0, 255, 255]], dtype='uint8')
    coord_mask = np.argmax(seg_pred, axis=0)
    for i in range(0, 4):
        if exist_pred[0, i] > p_threshold:
            lane_img[coord_mask == (i + 1)] = color[i]
    return img, lane_img


def isWeightExists():
    return os.path.isfile(weightPath)
