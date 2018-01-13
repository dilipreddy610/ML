# Typical setup to include TensorFlow.
import tensorflow as tf
from PIL.Image import Image

# Make a queue of file names including all the JPEG images files in the relative
# image directory.
filename_queue = tf.train.string_input_producer(
    tf.train.match_filenames_once('C:/Users/rathj/Downloads/img_align_celeba/000001.jpg'))

#filename_queue = tf.train.string_input_producer(
#    tf.train.match_filenames_once('../resources/celeba/*.jpg'))

#filename_queue = tf.train.string_input_producer(
#    tf.train.match_filenames_once('C:/Users/rathj/OneDrive/Pictures/Saved Pictures/*.jpg'))
#filename_queue = tf.train.string_input_producer(
#    tf.train.match_filenames_once("C:/Users/rathj/OneDrive/Pictures/Saved Pictures/*calicut.jpg"))

# Read an entire image file which is required since they're JPEGs, if the images
# are too large they could be split in advance to smaller files or use the Fixed
# reader to split up the file.
image_reader = tf.WholeFileReader()

# Read a whole file from the queue, the first returned value in the tuple is the
# filename which we are ignoring.
_, image_file = image_reader.read(filename_queue)


image_orig = tf.image.decode_jpeg(image_file, channels=3)
image = tf.image.resize_images(image_orig, [28, 28])
image.set_shape((28, 28, 3))
batch_size = 2000
num_preprocess_threads = 1
min_queue_examples = 256

images = tf.train.shuffle_batch(
[image],
batch_size=batch_size,
num_threads=num_preprocess_threads,
capacity=min_queue_examples + 3 * batch_size,
min_after_dequeue=min_queue_examples)


# Decode the image as a JPEG file, this will turn it into a Tensor which we can
# then use in training.
#image = tf.image.decode_jpeg(image_file)

# Start a new session to show example output.
with tf.Session() as sess:
    # launch the model in an InteractiveSession
    # creating an operation to initialize the variables we created
    tf.local_variables_initializer().run()

    # Coordinate the loading of image files.
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)

    '''
    for i in range(1): #length of your filename list
        mimg = image.eval() #here is your image Tensor :) 
        print(mimg.shape)

    '''
    
    # Get an image tensor and print its value.
    image_tensor = sess.run(images)
    print(image_tensor)
    print(tf.shape(image_tensor))
    
    # Finish off the filename queue coordinator.
    coord.request_stop()
    coord.join(threads)
    
    
    
    # Display the training images in the visualizer.
    tf.summary.image('images', images)




    
    