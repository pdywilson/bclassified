#!/usr/bin/env python
"""
 Copyright (c) 2018 Intel Corporation

 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at

      http://www.apache.org/licenses/LICENSE-2.0

 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
"""

from __future__ import print_function
import sys
import os
from argparse import ArgumentParser
import cv2
import time
import logging as log
from  openvino.inference_engine import IENetwork, IEPlugin
import numpy as np

def build_argparser():
    parser = ArgumentParser()
    parser.add_argument("-m", "--model", help="Path to an .xml file with a trained model.", required=True, type=str)
    parser.add_argument("-i", "--input",
                        help="Path to video file or image. 'cam' for capturing video stream from camera", required=True,
                        type=str)
    parser.add_argument("-l", "--cpu_extension",
                        help="MKLDNN (CPU)-targeted custom layers.Absolute path to a shared library with the kernels "
                             "impl.", type=str, default=None)
    parser.add_argument("-pp", "--plugin_dir", help="Path to a plugin folder", type=str, default=None)
    parser.add_argument("-d", "--device",
                        help="Specify the target device to infer on; CPU, GPU, FPGA or MYRIAD is acceptable. Demo "
                             "will look for a suitable plugin for device specified (CPU by default)", default="CPU",
                        type=str)
    parser.add_argument("--labels", help="Labels mapping file", default=None, type=str)
    parser.add_argument("-pt", "--prob_threshold", help="Probability threshold for detections filtering",
                        default=0.5, type=float)

    return parser


# output: sklearn.cluster.MeanShift() clustering object
def bclassify():
    import numpy as np
    from sklearn.decomposition import PCA
    import sklearn.cluster
    import pandas as pd
    import os
    import imageio
    import matplotlib.pyplot as plt
    from sklearn.cluster import MeanShift
	
    l = []
    for fname in os.listdir("imagesselected2/"):
        l.append(np.array(imageio.imread("imagesselected2/"+fname)).flatten())
    data = np.array(l)
    data = data/255
	
    def perform_pca_k(data, k):
        pca = PCA(n_components=k)
        principalComponents = pca.fit_transform(data)
        pca_df = pd.DataFrame(data = principalComponents, 
		                           columns = ["pc"+str(i) for i in range(1,k+1)])
        return pca, pca_df
    #pca, pca_df=perform_pca_k(data, 4)
    #print("explained_variance: ", pca.explained_variance_ratio_)
    #print("resulting df: ")
    #print(pca_df)
	
    def perform_pca_ratio(data, ratio):
        pca = PCA(ratio)
        principalComponents = pca.fit_transform(data)
        pca_df = pd.DataFrame(data = principalComponents, 
		                           columns = ["pc"+str(i) for i in range(1,pca.n_components_+1)])
        return pca, pca_df
    pca, pca_df = perform_pca_ratio(data, 0.9)
    #print("explained_variance: ", pca.explained_variance_ratio_)
    #print("resulting df: ")
    #print(pca_df)
	
    clustering = MeanShift(bandwidth=23,cluster_all=False).fit(data)
    print(clustering.labels_)
    return clustering, pca


def main():
	#get freshly trained Clustering and PCA for inference
    clustering, pca = bclassify()
    #line for log configuration
    log.basicConfig(format="[ %(levelname)s ] %(message)s", level=log.INFO, stream=sys.stdout)
	#parser for the arguments
    args = build_argparser().parse_args()
	#get xml model argument
    model_xml = args.model
	#get weight model argument
    model_bin = os.path.splitext(model_xml)[0] + ".bin"
    #Hardware plugin initialization for specified device and 
    log.info("Initializing plugin for {} device...".format(args.device))
    plugin = IEPlugin(device=args.device, plugin_dirs=args.plugin_dir)
	#load extensions library if specified
    if args.cpu_extension and 'CPU' in args.device:
        plugin.add_cpu_extension(args.cpu_extension)
    #Read intermediate representation of the model
    log.info("Reading IR...")
    net = IENetwork.from_ir(model=model_xml, weights=model_bin)
	# check if the model is supported
    if plugin.device == "CPU":
        supported_layers = plugin.get_supported_layers(net)
        not_supported_layers = [l for l in net.layers.keys() if l not in supported_layers]
        if len(not_supported_layers) != 0:
            log.error("Following layers are not supported by the plugin for specified device {}:\n {}".
                      format(plugin.device, ', '.join(not_supported_layers)))
            log.error("Please try to specify cpu extensions library path in demo's command line parameters using -l "
                      "or --cpu_extension command line argument")
            sys.exit(1)
	# check if the input and output of model is the right format, here we expect just one input (one image) and one output type (bounding boxes)
    assert len(net.inputs.keys()) == 1, "Demo supports only single input topologies"
    assert len(net.outputs) == 1, "Demo supports only single output topologies"
	# start the iterator on the input nodes
    input_blob = next(iter(net.inputs))
    print(input_blob)
	# start the iterator on the output
    out_blob = next(iter(net.outputs))
    log.info("Loading IR to the plugin...")
	# load the network
    exec_net = plugin.load(network=net)
    # Read and pre-process input image
    n, c, h, w = net.inputs[input_blob].shape
    del net
	#take care of the input data (camera or video file)
    if args.input == 'cam':
        input_stream = 0
    else:
        input_stream = args.input
        assert os.path.isfile(args.input), "Specified input file doesn't exist"
	#take care of the labels
    if args.labels:
        with open(args.labels, 'r') as f:
            labels_map = [x.strip() for x in f]
    else:
        labels_map = None

	#opencv function to take care of the video reading/capture
    cap = cv2.VideoCapture(input_stream)

    log.info("Starting inference ")
    log.info("To stop the demo execution press Esc button")

    render_time = 0
	#open the camera
    ret, frame = cap.read()
	#if open, we loop over the incoming frames
    while cap.isOpened():
        #we get the frame 
        ret, frame = cap.read()
        if not ret:
            break
		#get frame size
        initial_w = cap.get(3)
        initial_h = cap.get(4)
		#start the time counter
        inf_start = time.time()
        #reshape the frame size and channels order to fit the model input
        in_frame = cv2.resize(frame, (w, h))
        in_frame = in_frame.transpose((2, 0, 1))  # Change data layout from HWC to CHW
        in_frame = in_frame.reshape((n, c, h, w))
		
		#start the inference 
        exec_net.infer(inputs={input_blob: in_frame})

        #stop the clock
        inf_end = time.time()
        det_time = inf_end - inf_start
           
        # Parse detection results
        res = exec_net.requests[0].outputs[out_blob]
        
		#iterate over the results 
        for obj in res[0][0]:
            # check the object probability , if higher than threshold it will create the bounding box
            if obj[2] > args.prob_threshold:
			    # class ID
                class_id = int(obj[1])
			    #define top left corner column value
                xmin = int(obj[3] * initial_w)
				#define top left corner row value
                ymin = int(obj[4] * initial_h)
				#define bottom right  corner column value
                xmax = int(obj[5] * initial_w)
				#define bottom right corner row value
                ymax = int(obj[6] * initial_h)
                

                deltax = int((xmax - xmin)/2)
                deltay = int((ymax - ymin)/2)
                xmin = xmin - deltax
                xmax = xmax + deltax
                ymin = ymin + deltay
                ymax = ymax + deltay
                
                
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                gray = gray[ymin:ymax,xmin:xmax]/255
                #print(gray)
                if gray.size > 0:
                	gray = cv2.resize(gray,(128,100))
                	class_id = int(clustering.predict(gray.reshape(1,-1))[0])
                	#print(class_id)
                	#cv2.imshow("littleframe",gray)
                	color = (min(class_id * 12.5, 255), min(class_id * 7, 255), min(class_id * 5, 255))
                	cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), color, 2)
                	det_label = labels_map[class_id] if labels_map else str(class_id)
                	cv2.putText(frame, det_label + ' ' + str(round(obj[2] * 100, 1)) + ' %', (xmin, ymin - 7),
		                            cv2.FONT_HERSHEY_COMPLEX, 0.6, color, 1)
                    
					
            # Draw performance stats
            inf_time_message = "Inference time: {:.3f} ms".format(det_time * 1000)
            

            cv2.putText(frame, inf_time_message, (15, 15), cv2.FONT_HERSHEY_COMPLEX, 0.5, (200, 10, 10), 1)
            
       
		# show the result image
        cv2.imshow("Detection Results", frame)
       
        key = cv2.waitKey(1)
        if key == 27:
            break

    cv2.destroyAllWindows()
    del exec_net
    del plugin


if __name__ == '__main__':
    sys.exit(main() or 0)
