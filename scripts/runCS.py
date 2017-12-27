# -*- coding: utf-8 -*-
"""
Created on Tue Dec 26 15:50:16 2017

@author: avarfolomeev
"""

image_shape = (160, 512)
image = scipy.misc.imresize(img, image_shape)
im_softmax = sess.run([tf.nn.softmax(logits)], {keep_prob: 1.0, image_in: [image]})
out_shape = (image_shape[0], image_shape[1], 35)
res = im_softmax[0].reshape(out_shape)
mx=np.argmax(res,2)

#%%
alfa = (127,)
col = np.zeros((160,512,4),dtype=np.uint8)

for y in range(160):
    for x in range(512):
        col[y,x,:] = labels[mx[y,x]].color + alfa

#%%
col0 = scipy.misc.imresize(col,shape0)
col_img = scipy.misc.toimage(col0, mode="RGBA")

street_im = scipy.misc.toimage(img)
street_im.paste(col_img,box=None,mask=col_img)

scipy.misc.imsave("b.png",street_im)
