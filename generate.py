from keras.models import load_model
import cv2
from tqdm import tqdm
import numpy as np
from glob import glob
import time


gen = load_model('vroum\\vroumgen.h5')
dis = load_model('vroum\\vroumdis.h5')


while(1):

    rand_noise = np.random.normal(0, 1, (1, 100))
    pred = gen.predict(rand_noise)
    confidence = dis.predict(pred)

    gen_img = (0.5 * pred[0] + 0.5)*255

    cv2.imwrite('gen\\'+str(time.time())+'_'+str(confidence[0][0])+'.png', gen_img)
