'''
One Pixel Attack Methos
Reference Paper: https://arxiv.org/abs/1710.08864

    pixels - number of pixels to change (L0 norm)
    iters - number of iterations
    popsize - population size

'''

import torch
import cv2
import numpy as np
from scipy.optimize import differential_evolution
import torch.nn as nn
from torch.autograd import Variable
from model import simple_CNN

def preprocess(img):
    img = img.astype(np.float32)
    img /= 255.0
    img = img.transpose(2, 0, 1)
    return img

def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()

def perturb(x):
    adv_img = img.copy()

    # calculate pixel values
    pixs = np.array(np.split(x, len(x)/5)).astype(int)
    loc = (pixs[:, 0], pixs[:,1])
    val = pixs[:, 2:]
    adv_img[loc] = val

    return adv_img

def optimize(x):
    adv_img = perturb(x)

    inp = Variable(torch.from_numpy(preprocess(adv_img)).float().unsqueeze(0))
    out = model(inp)
    prob = softmax(out.data.numpy()[0])

    return prob[pred_orig]


def callback(x, convergence):
    global pred_adv, prob_adv
    adv_img = perturb(x)

    inp = Variable(torch.from_numpy(preprocess(adv_img)).float().unsqueeze(0))
    out = model(inp)
    prob = softmax(out.data.numpy()[0])

    pred_adv = np.argmax(prob)
    prob_adv = prob[pred_adv]
    if pred_adv != pred_orig and prob_adv >= 0.9:
        print('Attack successful..')
        print('Prob [%s]: %f' %(cifar10_class_names[pred_adv], prob_adv))
        print()
        return True
    else:
        print('Prob [%s]: %f' %(cifar10_class_names[pred_orig], prob[pred_orig]))


def scale(x, scale=5):
    return cv2.resize(x, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)

cifar10_class_names = {0: 'airplane', 1: 'automobile', 2: 'bird', 3: 'cat', 4: 'deer', 5: 'dog', 6: 'frog', 7: 'horse', 8: 'ship', 9: 'truck'}

image_path = "car.jpeg"
# Increase the pixel disturbance
pixels = 10
iters = 500
popsize = 10
model_path = "cifar10_basiccnn.pth.tar"

orig = cv2.imread(image_path)[..., ::-1]
orig = cv2.resize(orig, (32, 32))
img = orig.copy()
shape = orig.shape

model = simple_CNN()
saved = torch.load(model_path, map_location='cpu')
model.load_state_dict(saved['state_dict'])
model.eval()

inp = Variable(torch.from_numpy(preprocess(img)).float().unsqueeze(0))
prob_orig = softmax(model(inp).data.numpy()[0])
pred_orig = np.argmax(prob_orig)
print('Prediction before adversarial attack: %s' %(cifar10_class_names[pred_orig]))
print('Probability: %f' %(prob_orig[pred_orig]))

pred_adv = 0
prob_adv = 0

while True:
    bounds = [(0, shape[0]-1), (0, shape[1]), (0, 255), (0, 255), (0, 255)] * pixels
    result = differential_evolution(optimize, bounds, maxiter=iters, popsize=popsize, tol=1e-5, callback=callback)

    adv_img = perturb(result.x)
    inp = Variable(torch.from_numpy(preprocess(adv_img)).float().unsqueeze(0))
    out = model(inp)
    prob = softmax(out.data.numpy()[0])
    print('Prob [%s]: %f --> Prob[%s]: %f' %(cifar10_class_names[pred_orig], prob_orig[pred_orig], cifar10_class_names[pred_adv], prob_adv))

    cv2.imshow('adversarial image', scale(adv_img[..., ::-1]))

    key = 0
    while True:
        print("Press 'esc' to exit, 'space' to re-run..", end="\r")
        key = cv2.waitKey(100) & 0xFF
        if key == 27:
            breakfgsm
        elif key == ord('s'):
            cv2.imwrite('adv_img.png', scale(adv_img[..., ::-1]))
        elif key == 32:
            break
    if key == 27:
        break
cv2.destroyAllWindows()