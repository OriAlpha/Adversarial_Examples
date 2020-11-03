'''
Fast Gradient Sign Method
[Paper](https://arxiv.org/abs/1412.6572)

cv2 window interaction
ese - to exit and apply noise to images
'''

import torch
from torch.autograd import Variable
import torch.nn as nn
import numpy as np
import cv2
from model import simple_CNN


def nothing(x):
    pass

image_path = 'images/img_7.jpg'

print('Fast Gradient Sign Method')

adversarial_window = 'adversarial image'
cv2.namedWindow(adversarial_window)
cv2.createTrackbar('con_perturb', adversarial_window, 1, 255, nothing)

orig_img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
img = orig_img.copy().astype(np.float32)
perturbation = np.empty_like(orig_img)

mean = [0.5]
std = [0.5]
img /= 255.0
img = (img - mean)/std


# load model
model = simple_CNN(1, 10)
saved = torch.load('model.pth.tar', map_location='cpu')
model.load_state_dict(saved['state_dict'])
model.eval()
criterion = nn.CrossEntropyLoss()

# predictiction before attack
input = Variable(torch.from_numpy(img).float().unsqueeze(0).unsqueeze(0), requires_grad=True)

out = model(input)
predict = np.argmax(out.data.cpu().numpy())
print('predictiction before adversarialersial attack: %s' %(predict))

while True:
    # get trackbar position
    # Controls the strength of the perturbation
    con_perturb = cv2.getTrackbarPos('con_perturb', adversarial_window)

    input = Variable(torch.from_numpy(img).float().unsqueeze(0).unsqueeze(0), requires_grad=True)

    out = model(input)
    loss = criterion(out, Variable(torch.Tensor([float(predict)]).long()))

    # gradients in network
    loss.backward()

    # Measures the distance between legitimate and perturbed examples.
    input.data = input.data + ((con_perturb/255.0) * torch.sign(input.grad.data))
    input.data = input.data.clamp(min=-1, max=1)
    # Every time in back propogated, gradients will accumulate instead of replaced so reset to zeros.
    input.grad.data.zero_()

    # predictict on the adversarial image
    adversarial_predict = np.argmax(model(input).data.cpu().numpy())
    print(" "*60, end='\r')
    print("After attack: con_perturb [%f] \t%s"
            %(con_perturb, adversarial_predict), end="\r")#, end='\r')#'con_perturb:', con_perturb, end='\r')


    # deprocess image
    adversarial = input.data.cpu().numpy()[0][0]
    perturbation = adversarial-img
    adversarial = (adversarial * std) + mean
    adversarial = adversarial * 255.0
    adversarial = np.clip(adversarial, 0, 255).astype(np.uint8)
    perturbation = perturbation*255
    perturbation = np.clip(perturbation, 0, 255).astype(np.uint8)


    # display images
    cv2.imshow(adversarial_window, perturbation)
    cv2.imshow('perturbation', adversarial)
    key = cv2.waitKey(500) & 0xFF
    if key == 27:
        break
    elif key == ord('s'):
        cv2.imwrite('image_adversarial.png', adversarial)
        cv2.imwrite('perturbation.png', perturbation)
print()
cv2.destroyAllWindows()