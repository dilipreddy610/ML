import tensorflow as tf
import numpy as np

from scipy import misc
import glob
'''
for image_path in glob.glob('..//resources//Numerals//0//*.png'):
    image = misc.imread(image_path)
    print(image.shape)
    print(image.dtype)
    print(np.asarray(image))
'''
filenames = ['..//resources//Numerals//0//0001a.png', '..//resources//Numerals//0//0001b.png']

# step 2
filename_queue = tf.train.string_input_producer(filenames)

# step 3: read, decode and resize images
reader = tf.WholeFileReader()
filename, content = reader.read(filename_queue)
image = tf.image.decode_png(content)
#image = tf.cast(image, tf.float32)
#resized_image = tf.image.resize_images(image, [224, 224])
xyz = tf.Variable(tf.zeros([10]))
       
# launch the model in an InteractiveSession
sess = tf.InteractiveSession()
print('sess', sess)
# creating an operation to initialize the variables we created
tf.global_variables_initializer().run()
       
                  
    #image = tf.image.decode_png(image_file)
    #print(tf.Print(image, [image]))
    #print(sess.run(image))
#img = image.eval()
#print(np.asarray(img))


'''
# Make a queue of file names including all the JPEG images files in the relative
# image directory.
filename_queue = tf.train.string_input_producer(
    tf.train.match_filenames_once("C:/dump/my-space/Mlprojects/proj3/mnist/resources/Numerals/0/*.jpg"))



filename_queue = tf.train.string_input_producer(
    tf.train.match_filenames_once("C:/Users/rathj/OneDrive/Pictures/Saved Pictures/"))

# Read an entire image file which is required since they're JPEGs, if the images
# are too large they could be split in advance to smaller files or use the Fixed
# reader to split up the file.
image_reader = tf.WholeFileReader()

# Read a whole file from the queue, the first returned value in the tuple is the
# filename which we are ignoring.
_, image_file = image_reader.read(filename_queue)

# Decode the image as a JPEG file, this will turn it into a Tensor which we can
# then use in training.
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    #image = tf.image.decode_png(image_file)
    image = tf.image.decode_jpeg(image_file)
    #print(tf.Print(image, [image]))
    print(sess.run(image))
'''
    