
"""""""""""""""""""""""""""""""""""""""""""""IMPORTING THE REQUIREMENTS"""""""""""""""""""""""""""""""""""""""""""""
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
tf.compat.v1.disable_resource_variables()
import numpy as np
from seaborn import color_palette
import cv2
import time
from collecting_data import capturing_training_images
from collecting_data import capturing_testing_images
from face_identification_model import training
from face_identification import detection

""" VERSIONS USED """

print("Tensorflow : {}".format(tf.__version__))
print("Numpy : {}".format(np.__version__))


""" HYPERPARAMETERS """

_BATCH_NORM_DECAY = 0.9
_BATCH_NORM_EPSILON = 1e-05
_LEAKY_RELU = 0.1
_ANCHORS = [(10, 13), (16, 30), (33, 23),
            (30, 61), (62, 45), (59, 119),
            (116, 90), (156, 198), (373, 326)]
_MODEL_SIZE = (416, 416)
_MAX_OUTPUT_SIZE = 20
IOU_THRESHOLD = 0.5
CONFIDENCE_THRESHOLD = 0.5



""""""""""""""""""""""""""""""""""BATCH NORMALIZATION AND FIXED PADDING FOR CONV LAYERS"""""""""""""""""""""""""""""

def batch_norm(inputs, training, data_format):

  return tf.layers.batch_normalization(
      inputs=inputs, axis=1 if data_format == 'channels_first' else 3,
      momentum=_BATCH_NORM_DECAY, epsilon=_BATCH_NORM_EPSILON,
      scale=True, training=training)


def fixed_padding(inputs, kernel_size, data_format):

  pad_total = kernel_size - 1
  pad_beg = pad_total // 2
  pad_end = pad_total - pad_beg

  if data_format == 'channels_first':
      padded_inputs = tf.pad(inputs, [[0, 0], [0, 0],
                                      [pad_beg, pad_end],
                                      [pad_beg, pad_end]])
  else:
      padded_inputs = tf.pad(inputs, [[0, 0], [pad_beg, pad_end],
                                      [pad_beg, pad_end], [0, 0]])
  return padded_inputs


def conv2D_fixed_padding(inputs, filters, kernel_size, data_format, strides=1):

  if strides > 1:
      inputs = fixed_padding(inputs, kernel_size, data_format)

  return tf.layers.conv2d(
      inputs=inputs, filters=filters, kernel_size=kernel_size,
      strides=strides, padding=('SAME' if strides == 1 else 'VALID'),
      use_bias=False, data_format=data_format)


""""""""""""""""""""""""""""""""""""""""" DARKNET-53 RESIDUAL BLOCK """""""""""""""""""""""""""""""""""""""

def darknet_residual_block(inputs, filters, training, data_format, strides=1):
    shortcut = inputs

    inputs = conv2D_fixed_padding(inputs, filters=filters, kernel_size=1, strides=strides, data_format=data_format)
    inputs = batch_norm(inputs, training=training, data_format=data_format)
    inputs = tf.nn.leaky_relu(inputs, alpha=_LEAKY_RELU)

    inputs = conv2D_fixed_padding(inputs, filters=2 * filters, kernel_size=3, strides=strides, data_format=data_format)
    inputs = batch_norm(inputs, training=training, data_format=data_format)
    inputs = tf.nn.leaky_relu(inputs, alpha=_LEAKY_RELU)

    inputs += shortcut

    return inputs


""""""""""""""""""""""""""""""""""""""""""" DARKNET-53 FULL LAYER """""""""""""""""""""""""""""""""""""""""""

def darknet(inputs, training, data_format):
    inputs = conv2D_fixed_padding(inputs, filters=32, kernel_size=3,
                                  data_format=data_format)
    inputs = batch_norm(inputs, training=training, data_format=data_format)
    inputs = tf.nn.leaky_relu(inputs, alpha=_LEAKY_RELU)
    inputs = conv2D_fixed_padding(inputs, filters=64, kernel_size=3,
                                  strides=2, data_format=data_format)
    inputs = batch_norm(inputs, training=training, data_format=data_format)
    inputs = tf.nn.leaky_relu(inputs, alpha=_LEAKY_RELU)

    inputs = darknet_residual_block(inputs, filters=32, training=training,
                                    data_format=data_format)

    inputs = conv2D_fixed_padding(inputs, filters=128, kernel_size=3,
                                  strides=2, data_format=data_format)
    inputs = batch_norm(inputs, training=training, data_format=data_format)
    inputs = tf.nn.leaky_relu(inputs, alpha=_LEAKY_RELU)

    for _ in range(2):
        inputs = darknet_residual_block(inputs, filters=64,
                                        training=training,
                                        data_format=data_format)

    inputs = conv2D_fixed_padding(inputs, filters=256, kernel_size=3,
                                  strides=2, data_format=data_format)
    inputs = batch_norm(inputs, training=training, data_format=data_format)
    inputs = tf.nn.leaky_relu(inputs, alpha=_LEAKY_RELU)

    for _ in range(8):
        inputs = darknet_residual_block(inputs, filters=128,
                                        training=training,
                                        data_format=data_format)  # FEATURE VECTOR_1 FOR DETECTION 52 X 52

    route1 = inputs

    inputs = conv2D_fixed_padding(inputs, filters=512, kernel_size=3,
                                  strides=2, data_format=data_format)
    inputs = batch_norm(inputs, training=training, data_format=data_format)
    inputs = tf.nn.leaky_relu(inputs, alpha=_LEAKY_RELU)

    for _ in range(8):
        inputs = darknet_residual_block(inputs, filters=256,
                                        training=training,
                                        data_format=data_format)  # FEATURE VECTOR_2 FOR DETECTION 26 X 26

    route2 = inputs

    inputs = conv2D_fixed_padding(inputs, filters=1024, kernel_size=3,
                                  strides=2, data_format=data_format)
    inputs = batch_norm(inputs, training=training, data_format=data_format)
    inputs = tf.nn.leaky_relu(inputs, alpha=_LEAKY_RELU)

    for _ in range(4):
        inputs = darknet_residual_block(inputs, filters=512,
                                        training=training,
                                        data_format=data_format)  # FEATURE VECTOR_3 FOR DETECTION 16 X 16

    return route1, route2, inputs


""""""""""""""""""""""""""""""""""""""""""" YOLO-V3 CONVOLUTIONAL LAYERS """""""""""""""""""""""""""""""""""""""""""

def yolo_convolution_block(inputs, filters, training, data_format):

  inputs = conv2D_fixed_padding(inputs, filters=filters, kernel_size=1, data_format=data_format)
  inputs = batch_norm(inputs, training=training, data_format=data_format)
  inputs = tf.nn.leaky_relu(inputs, alpha=_LEAKY_RELU)

  inputs = conv2D_fixed_padding(inputs, filters=2 * filters, kernel_size=3, data_format=data_format)
  inputs = batch_norm(inputs, training=training, data_format=data_format)
  inputs = tf.nn.leaky_relu(inputs, alpha=_LEAKY_RELU)

  inputs = conv2D_fixed_padding(inputs, filters=filters, kernel_size=1, data_format=data_format)
  inputs = batch_norm(inputs, training=training, data_format=data_format)
  inputs = tf.nn.leaky_relu(inputs, alpha=_LEAKY_RELU)

  inputs = conv2D_fixed_padding(inputs, filters=2 * filters, kernel_size=3, data_format=data_format)
  inputs = batch_norm(inputs, training=training, data_format=data_format)
  inputs = tf.nn.leaky_relu(inputs, alpha=_LEAKY_RELU)

  inputs = conv2D_fixed_padding(inputs, filters=filters, kernel_size=1, data_format=data_format)
  inputs = batch_norm(inputs, training=training, data_format=data_format)
  inputs = tf.nn.leaky_relu(inputs, alpha=_LEAKY_RELU)

  route = inputs # CONCATINATE WITH THE NEXT FEATURE VECTOR

  inputs = conv2D_fixed_padding(inputs, filters=2 * filters, kernel_size=3, data_format=data_format)
  inputs = batch_norm(inputs, training=training, data_format=data_format)
  inputs = tf.nn.leaky_relu(inputs, alpha=_LEAKY_RELU)

  return route, inputs


""""""""""""""""""""""""""""""""""""""""""" YOLO-V3 DETECTION LAYER """""""""""""""""""""""""""""""""""""""""""


def yolo(inputs, n_classes, anchors, img_size, data_format):

    n_anchors = len(anchors)

    inputs = tf.layers.conv2d(inputs, filters=n_anchors * (5 + n_classes),
                              kernel_size=1, strides=1, use_bias=True,
                              data_format=data_format)

    shape = inputs.get_shape().as_list()
    grid_shape = shape[2:4] if data_format == 'channels_first' else shape[1:3]
    if data_format == 'channels_first':
        inputs = tf.transpose(inputs, [0, 2, 3, 1])
    inputs = tf.reshape(inputs, [-1, n_anchors * grid_shape[0] * grid_shape[1],
                                 5 + n_classes])

    strides = (img_size[0] // grid_shape[0], img_size[1] // grid_shape[1])

    print("Detection_layer : tf.split : {}".format(inputs))

    box_centers, box_shapes, confidence, classes = \
        tf.split(inputs, [2, 2, 1, n_classes], axis=-1)

    x = tf.range(grid_shape[0], dtype=tf.float32)
    y = tf.range(grid_shape[1], dtype=tf.float32)
    x_offset, y_offset = tf.meshgrid(x, y)
    x_offset = tf.reshape(x_offset, (-1, 1))
    y_offset = tf.reshape(y_offset, (-1, 1))
    x_y_offset = tf.concat([x_offset, y_offset], axis=-1)
    x_y_offset = tf.tile(x_y_offset, [1, n_anchors])
    x_y_offset = tf.reshape(x_y_offset, [1, -1, 2])
    box_centers = tf.nn.sigmoid(box_centers)
    box_centers = (box_centers + x_y_offset) * strides

    anchors = tf.tile(anchors, [grid_shape[0] * grid_shape[1], 1])
    box_shapes = tf.exp(box_shapes) * tf.to_float(anchors)

    confidence = tf.nn.sigmoid(confidence)

    classes = tf.nn.sigmoid(classes)

    inputs = tf.concat([box_centers, box_shapes,
                        confidence, classes], axis=-1)

    return inputs


"""""""""" UNSAMPLING THE DATA FROM YOLO-V3 CONVOLUTIONAL LAYER BEFORE CONCATINATING WITH THE FEATURE VECTOR """""""""""

def upsample(inputs, out_shape, data_format):

    if data_format == 'channels_first':
        inputs = tf.transpose(inputs, [0, 2, 3, 1])
        new_height = out_shape[3]
        new_width = out_shape[2]
    else:
        new_height = out_shape[2]
        new_width = out_shape[1]

    inputs = tf.image.resize_nearest_neighbor(inputs, (new_height, new_width))

    if data_format == 'channels_first':
        inputs = tf.transpose(inputs, [0, 3, 1, 2])

    return inputs


""""""""""""""""""""""""""""""""""""""" BUILDING BOXES AROUND THE DETECTED RESULT """""""""""""""""""""""""""""""""

def build_boxes(inputs):
    print("Build_boxes : inputs : {}".format(tf.split(inputs, [1, 1, 1, 1, 1, -1], axis=-1)))

    center_x, center_y, width, height, confidence, classes = \
        tf.split(inputs, [1, 1, 1, 1, 1, -1], axis=-1)

    print(center_x, center_y, width, height, confidence, classes)

    top_left_x = center_x - width / 2
    top_left_y = center_y - height / 2
    bottom_right_x = center_x + width / 2
    bottom_right_y = center_y + height / 2

    print(top_left_x)

    boxes = tf.concat([top_left_x, top_left_y,
                       bottom_right_x, bottom_right_y,
                       confidence, classes], axis=-1)

    return boxes


""""""""""""""""""""""""""""""""""" NON-MAX SUPRESSION FOR FINDING THE BEST DETECTION """""""""""""""""""""""""""""""""

def non_max_suppression(inputs, n_classes, max_output_size, iou_threshold,
                        confidence_threshold):
    batch = tf.unstack(inputs)
    boxes_dicts = []
    for boxes in batch:
        boxes = tf.boolean_mask(boxes, boxes[:, 4] > confidence_threshold)
        classes = tf.argmax(boxes[:, 5:], axis=-1)
        classes = tf.expand_dims(tf.to_float(classes), axis=-1)
        boxes = tf.concat([boxes[:, :5], classes], axis=-1)

        boxes_dict = dict()
        for cls in range(n_classes):
            mask = tf.equal(boxes[:, 5], cls)
            mask_shape = mask.get_shape()
            if mask_shape.ndims != 0:
                class_boxes = tf.boolean_mask(boxes, mask)
                boxes_coords, boxes_conf_scores, _ = tf.split(class_boxes,
                                                              [4, 1, -1],
                                                              axis=-1)
                boxes_conf_scores = tf.reshape(boxes_conf_scores, [-1])
                indices = tf.image.non_max_suppression(boxes_coords,
                                                       boxes_conf_scores,
                                                       max_output_size,
                                                       iou_threshold)
                class_boxes = tf.gather(class_boxes, indices)
                boxes_dict[cls] = class_boxes[:, :5]

        boxes_dicts.append(boxes_dict)

    return boxes_dicts


""""""""""""""""""""""""""""""""""""""""""""""""""""""""""" YOLO-V3 """""""""""""""""""""""""""""""""""""""""""""""""""

class Yolo_v3:

    def __init__(self, n_classes, model_size, max_output_size, iou_threshold,
                 confidence_threshold, data_format=None):

        if not data_format:
            if tf.test.is_built_with_cuda():
                data_format = 'channels_first'
            else:
                data_format = 'channels_last'

        self.n_classes = n_classes
        self.model_size = model_size
        self.max_output_size = max_output_size
        self.iou_threshold = iou_threshold
        self.confidence_threshold = confidence_threshold
        self.data_format = data_format

    def __call__(self, inputs, training):

        with tf.variable_scope('yolo_v3_model'):
            if self.data_format == 'channels_first':
                inputs = tf.transpose(inputs, [0, 3, 1, 2])

            inputs = inputs / 255

            route1, route2, inputs = darknet(inputs, training=training,
                                             data_format=self.data_format)

            route, inputs = yolo_convolution_block(
                inputs, filters=512, training=training,
                data_format=self.data_format)
            detect1 = yolo(inputs, n_classes=self.n_classes,
                           anchors=_ANCHORS[6:9],
                           img_size=self.model_size,
                           data_format=self.data_format)

            inputs = conv2D_fixed_padding(route, filters=256, kernel_size=1,
                                          data_format=self.data_format)
            inputs = batch_norm(inputs, training=training,
                                data_format=self.data_format)
            inputs = tf.nn.leaky_relu(inputs, alpha=_LEAKY_RELU)
            upsample_size = route2.get_shape().as_list()
            inputs = upsample(inputs, out_shape=upsample_size,
                              data_format=self.data_format)
            axis = 1 if self.data_format == 'channels_first' else 3
            inputs = tf.concat([inputs, route2], axis=axis)
            route, inputs = yolo_convolution_block(
                inputs, filters=256, training=training,
                data_format=self.data_format)
            detect2 = yolo(inputs, n_classes=self.n_classes,
                           anchors=_ANCHORS[3:6],
                           img_size=self.model_size,
                           data_format=self.data_format)

            inputs = conv2D_fixed_padding(route, filters=128, kernel_size=1,
                                          data_format=self.data_format)
            inputs = batch_norm(inputs, training=training,
                                data_format=self.data_format)
            inputs = tf.nn.leaky_relu(inputs, alpha=_LEAKY_RELU)
            upsample_size = route1.get_shape().as_list()
            inputs = upsample(inputs, out_shape=upsample_size,
                              data_format=self.data_format)
            inputs = tf.concat([inputs, route1], axis=axis)
            route, inputs = yolo_convolution_block(
                inputs, filters=128, training=training,
                data_format=self.data_format)
            detect3 = yolo(inputs, n_classes=self.n_classes,
                           anchors=_ANCHORS[0:3],
                           img_size=self.model_size,
                           data_format=self.data_format)

            inputs = tf.concat([detect1, detect2, detect3], axis=1)

            inputs = build_boxes(inputs)

            boxes_dicts = non_max_suppression(
                inputs, n_classes=self.n_classes,
                max_output_size=self.max_output_size,
                iou_threshold=self.iou_threshold,
                confidence_threshold=self.confidence_threshold)

            return boxes_dicts


""""""""""""""""""""""""""""""""""""""""""""""""" LOADING CLASS NAMES """""""""""""""""""""""""""""""""""""""""""""""""

def load_class_names(file_name):

    with open(file_name, 'r') as f:
        class_names = f.read().splitlines()
    return class_names


""""""""""""""""""""""""""""""""""""" DRAWING FRAMES AROUND DETECTED RESULTS """""""""""""""""""""""""""""""""""""""""""

def draw_frame(frame, frame_size, boxes_dicts, class_names, model_size):

    boxes_dict = boxes_dicts[0]
    resize_factor = (frame_size[0] / model_size[1], frame_size[1] / model_size[0])
    colors = ((np.array(color_palette("hls", 80)) * 255)).astype(np.uint8)
    for cls in range(len(class_names)):
        boxes = boxes_dict[cls]
        color = colors[cls]
        color = tuple([int(x) for x in color])
        if np.size(boxes) != 0:
            for box in boxes:
                xy = box[:4]
                xy = [int(xy[i] * resize_factor[i % 2]) for i in range(4)]
                cv2.rectangle(frame, (xy[0], xy[1]), (xy[2], xy[3]), color[::-1], 2)
                (test_width, text_height), baseline = cv2.getTextSize(class_names[cls],
                                                                      cv2.FONT_HERSHEY_SIMPLEX,
                                                                      0.75, 1)
                cv2.rectangle(frame, (xy[0], xy[1]),
                              (xy[0] + test_width, xy[1] - text_height - baseline),
                              color[::-1], 1)
                cv2.putText(frame, class_names[cls], (xy[0], xy[1] - baseline),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.75, color[::-1], 2)


""""""""""""""""""""""""" LOADING WEIGHTS FOR YOLO-V3 FULL NETWORK INCLUDING DARKNET """""""""""""""""""""""""

def load_weights(variables, file_name):

    print("Variables {}".format(len(variables)))

    with open(file_name, "rb") as f:

        np.fromfile(f, dtype=np.int32, count=5)
        weights = np.fromfile(f, dtype=np.float32)

        assign_ops = []
        ptr = 0

        for i in range(52):
            conv_var = variables[5 * i]
            gamma, beta, mean, variance = variables[5 * i + 1:5 * i + 5]
            batch_norm_vars = [beta, gamma, mean, variance]

            for var in batch_norm_vars:
                shape = var.shape.as_list()
                num_params = np.prod(shape)
                var_weights = weights[ptr:ptr + num_params].reshape(shape)
                ptr += num_params
                assign_ops.append(tf.assign(var, var_weights))

            shape = conv_var.shape.as_list()
            num_params = np.prod(shape)
            var_weights = weights[ptr:ptr + num_params].reshape(
                (shape[3], shape[2], shape[0], shape[1]))
            var_weights = np.transpose(var_weights, (2, 3, 1, 0))
            ptr += num_params
            assign_ops.append(tf.assign(conv_var, var_weights))


        ranges = [range(0, 6), range(6, 13), range(13, 20)]
        unnormalized = [6, 13, 20]
        for j in range(3):
            for i in ranges[j]:
                current = 52 * 5 + 5 * i + j * 2
                conv_var = variables[current]
                gamma, beta, mean, variance =  \
                    variables[current + 1:current + 5]
                batch_norm_vars = [beta, gamma, mean, variance]

                for var in batch_norm_vars:
                    shape = var.shape.as_list()
                    num_params = np.prod(shape)
                    var_weights = weights[ptr:ptr + num_params].reshape(shape)
                    ptr += num_params
                    assign_ops.append(tf.assign(var, var_weights))

                shape = conv_var.shape.as_list()
                num_params = np.prod(shape)
                var_weights = weights[ptr:ptr + num_params].reshape(
                    (shape[3], shape[2], shape[0], shape[1]))
                var_weights = np.transpose(var_weights, (2, 3, 1, 0))
                ptr += num_params
                assign_ops.append(tf.assign(conv_var, var_weights))

            bias = variables[52 * 5 + unnormalized[j] * 5 + j * 2 + 1]
            shape = bias.shape.as_list()
            num_params = np.prod(shape)
            var_weights = weights[ptr:ptr + num_params].reshape(shape)
            ptr += num_params
            assign_ops.append(tf.assign(bias, var_weights))

            conv_var = variables[52 * 5 + unnormalized[j] * 5 + j * 2]
            shape = conv_var.shape.as_list()
            num_params = np.prod(shape)
            var_weights = weights[ptr:ptr + num_params].reshape(
                (shape[3], shape[2], shape[0], shape[1]))
            var_weights = np.transpose(var_weights, (2, 3, 1, 0))
            ptr += num_params
            assign_ops.append(tf.assign(conv_var, var_weights))

    return assign_ops


""""""""""""""""""""""""""""""""""""""""""""""""""""" DETECTION """""""""""""""""""""""""""""""""""""""""""""""

def main(type, IOU_THRESHOLD, CONFIDENCE_THRESHOLD):
    class_names = load_class_names('D:\Object_detection\coco.names')
    n_classes = len(class_names)

    model = Yolo_v3(n_classes=n_classes, model_size=_MODEL_SIZE,
                    max_output_size=_MAX_OUTPUT_SIZE,
                    iou_threshold=IOU_THRESHOLD,
                    confidence_threshold=CONFIDENCE_THRESHOLD)

    if type == 'video':
        inputs = tf.placeholder(tf.float32, [1, *_MODEL_SIZE, 3])
        detections = model(inputs, training=False)
        model_vars = tf.global_variables(scope='yolo_v3_model')
        assign_ops = load_weights(model_vars, '')  # DOWNLOAD YOLO_V3 WEIGHTS AND PASTE IT IN HERE !!!!

        with tf.Session() as sess:
            sess.run(assign_ops)
            capture_duration = 40
            cap = cv2.VideoCapture(0)
            frame_size = (cap.get(cv2.CAP_PROP_FRAME_WIDTH),
                          cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fourcc = cv2.VideoWriter_fourcc(*'X264')
            fps = cap.get(cv2.CAP_PROP_FPS)
            out = cv2.VideoWriter('D:\Object_detection\detections.mp4', fourcc, fps,
                                  (int(frame_size[0]), int(frame_size[1])))
            print("DETECTING OBJECTS.")
            try:
                start = time.time()
                while (int(time.time() - start) < capture_duration):
                    ret, frame = cap.read()

                    if not ret:
                        break
                    resized_frame = cv2.resize(frame, dsize=_MODEL_SIZE[::-1],
                                               interpolation=cv2.INTER_NEAREST)
                    detection_result = sess.run(detections,
                                                feed_dict={inputs: [resized_frame]})

                    draw_frame(frame, frame_size, detection_result,
                               class_names, _MODEL_SIZE)

                    key = cv2.waitKey(1) & 0xFF

                    if key == ord('q'):
                        break

                    out.write(frame)

            finally:
                cv2.destroyAllWindows()
                cap.release()
                print('PREPARING FOR FACE DETECTION.')


    else:
        raise ValueError("INAPPROPRIATE DATA.")


""""""""""""""""""""""""""""""""""" CALLING CAPTURING DATA FUNCTION  FOR FACE DETCTION """""""""""""""""""""""""""""""""

names = []
def capturing_datasets():

    name = input("\nENTER CLASS NAME : ")
    names.append(name)
    print("\nCAPTURING TRAINING IMAGES\n")
    capturing_training_images(name)
    print("\nCAPTURING TESTING IMAGES\n")
    capturing_testing_images(name)


""""""""""""""""""""""""""""""""""""""""""""""" MAIN FUNCTION OF THE PROJECT """""""""""""""""""""""""""""""""""""""""

if __name__ == '__main__':

    print("\nPROCEED TO TAKE IMAGES ?  ('yes'/'no')")
    ans = str(input())
    ans = ans.lower()
    if ans == 'yes':
        n_classes = int(input("\nNUMBER OF CLASSES : "))
        for i in range(n_classes):
            capturing_datasets()
        print("\n DATA COLLECTED \n")

    main("video", IOU_THRESHOLD, CONFIDENCE_THRESHOLD)
    print("\nTRAINING MODEL WITH NEW DATAS\n")
    classes = training()
    print('\nFACE DETECTION AND IDENTIFICATION.\n')
    detection(classes)









