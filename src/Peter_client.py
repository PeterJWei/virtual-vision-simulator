#!/usr/bin/env python
#
# Copyright (c) 2011-2012 Wiktor Starzyk, Faisal Z. Qureshi
#
# This file is part of the Virtual Vision Simulator.
#
# The Virtual Vision Simulator is free software: you can 
# redistribute it and/or modify it under the terms 
# of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, 
# or (at your option) any later version.
#
# The Virtual Vision Simulator is distributed in the hope 
# that it will be useful, but WITHOUT ANY WARRANTY; 
# without even the implied warranty of MERCHANTABILITY or 
# FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with the Virtual Vision Simulator.  
# If not, see <http://www.gnu.org/licenses/>.
#


import wx
import sys
import socket
import tensorflow as tf
import cv2
import thread
import argparse
import logging
import numpy as np
import time
import math
import csv
from random import randint

from pandac.PandaModules import NetDatagram
from simulator.panda3d.server.socket_packet import SocketPacket

#Message Types
VV_ACK_OK = 0
VV_CAM_LIST = 1
VV_IMG = 2
VV_REQ_SIGNATURE = 3
VV_TRACK_SIGNATURE = 4
VV_REQ_VIDEO_ANALYSIS = 5
VV_SYNC_ACK = 6
VV_READY = 7
VV_CAM_MESSAGE = 8
VV_VP_ACK_OK = 9
VV_VP_ACK_FAILED = 10

VP_SESSION = 100
VP_REQ_IMG = 101
VP_REQ_CAM_LIST = 102
VP_TRACKING_DATA = 103
VP_SIGNATURE = 104
VP_CAM_PAN = 105
VP_CAM_TILT = 106
VP_CAM_ZOOM = 107
VP_CAM_DEFAULT = 108
VP_CAM_IMAGE = 109
VP_CAM_RESOLUTION = 110

SYNC_SESSION = 200
SYNC_REQ_CAM_LIST = 201
SYNC_REMOTE_CAM_LIST = 202
SYNC_STEP = 203
SYNC_CAM_MESSAGE = 204

STATIC_PIPELINE = 0
ACTIVE_PIPELINE = 1

SESSION_TYPE = 1

class Client:
    def __init__(self, parent, id, title, ip_address, port, pipeline, init_pan, init_tilt, init_x, init_y,
                 save_images=False):
        try:
            logging.debug("Connecting to server with IP: %s and PORT: %s" 
                          %(ip_address, port))
            self.client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.client_socket.connect((ip_address, port))
            #self.client_socket.settimeout(3)
            self.client_socket.setblocking(0)
            logging.debug("Connected!")

        except Exception as e:
            print "Error connecting to server!"
            sys.exit(0)
        self.init_x = init_x
        self.init_y = init_y
        self.init_z = 90.0
        self.init_pan = init_pan
        self.init_tilt = init_tilt
        self.onoff = True
        self.background = None
        self.read_buffer = ''
        self.write_buffer = ''
        self.read_state = 0
        self.packet = SocketPacket()
        self.connected = False
        self.x1 = 100
        self.x2 = 140
        self.p = 0
        self.t = 0
        self.z = 100
        self.count = 0
        self.pipeline = pipeline
        self.save_images = save_images
        self.image_num = 0

    def step(self):
        messages, imageData = self.reader_polling()
        logging.debug("Reader Polling Complete!")
        self.write_polling()
        logging.debug("Write Polling Complete!")
        return messages, imageData


    def write_packet(self, packet):
        self.write_buffer = self.write_buffer + packet.get_header() + packet.get_body()


    def write_polling(self):
        if self.write_buffer != '':
            self.client_socket.send(self.write_buffer)
        self.write_buffer = ''
        return

    def reader_polling(self):
        data = ""
        messages = []
        imageData = []
        try:
            data = self.client_socket.recv(1024)
            logging.debug("No Socket Error")
        except socket.error, ex:
            logging.debug("Socket Error, " + str(ex))
            pass

        if data != '':
            self.read_buffer = self.read_buffer + data

        while (True):
            if self.read_state == 0:
                logging.debug("Packet length: " + str(len(self.read_buffer)))
                logging.debug("Packet header length: " + str(self.packet.header_length))
                if len(self.read_buffer) >= self.packet.header_length:
                    bytes_consumed = self.packet.header_length
                    self.packet.header = self.read_buffer[:bytes_consumed]
                    self.read_body_length = self.packet.decode_header()  # consumes packet.data
                    self.read_buffer = self.read_buffer[bytes_consumed:]
                    self.read_state = 1

                else:
                    break
            
            if self.read_state == 1:
                if len(self.read_buffer) >= self.read_body_length:
                    bytes_consumed = self.read_body_length
                    self.packet.data = self.read_buffer[:bytes_consumed]
                    self.packet.offset = 0
                    self.read_body_length = 0
                    self.read_buffer = self.read_buffer[bytes_consumed:]
                    self.read_state = 0
                    message_type, d = self.new_data_callback(self.packet)
                    messages.append(message_type)
                    imageData.append(d)
                else:
                    break
        return messages, imageData

    def setResolution(self, event):
        try:
            width = int(self.width.GetValue())
            height = int(self.height.GetValue())
            self.width.SetValue("")
            self.height.SetValue("")
        except Exception as e:
            logging.error("Resolution is not valid!")
        else:
            w = SocketPacket()
            w.add_int(VP_CAM_RESOLUTION)
            w.add_int(self.client_id)
            w.add_int(self.cam_id)
            w.add_int(width)
            w.add_int(height)
            w.encode_header()
            self.write_packet(w)

    def pan(self, pos=True):
        if pos:
            direction = 1
        else:
            direction = -1
        angle = direction * 10
        if (self.p + angle > 30) or (self.p + angle < -30):
            print("Panning Constrained")
        else:
            self.p += angle
        w = SocketPacket()
        w.add_int(VP_CAM_PAN)
        w.add_int(self.client_id)
        w.add_int(self.cam_id)
        w.add_float(angle)
        w.encode_header()
        self.write_packet(w)


    def tilt(self, pos=True):
        if pos:
            direction = 1
        else:
            direction = -1
        angle = direction * 10
        if self.t + angle > 30 or self.t + angle < -30:
            print("Tilt Constrained")
        else:
            self.t += angle
        w = SocketPacket()
        w.add_int(VP_CAM_TILT)
        w.add_int(self.client_id)
        w.add_int(self.cam_id)
        w.add_float(angle)
        w.encode_header()
        self.write_packet(w)


    def zoom(self, pos=True):
        if pos:
            direction = 1
        else:
            direction = -1
        angle = direction * 10
        if self.z + angle > 100 or self.z + angle < 30:
            print("Zoom Constrained")
        else:
            self.z += angle
        w = SocketPacket()
        w.add_int(VP_CAM_ZOOM)
        w.add_int(self.client_id)
        w.add_int(self.cam_id)
        w.add_float(angle)
        w.encode_header()
        self.write_packet(w)


    def default(self, event):
        w = SocketPacket()
        w.add_int(VP_CAM_DEFAULT)
        w.add_int(self.client_id)
        w.add_int(self.cam_id)
        w.encode_header()
        self.write_packet(w)


    def getImage(self):
        w = SocketPacket()
        w.add_int(VP_CAM_IMAGE)
        w.add_int(self.client_id)
        w.add_int(self.cam_id)
        w.encode_header()
        self.write_packet(w)
        #print("Sending packet to get image")


    def new_data_callback(self, packet):
        message_type = packet.get_int()
        data = None
        #print("Got packet, " + str(message_type))
        if message_type == VV_ACK_OK:
            logging.debug("VV_ACK_OK")
            server_ip = packet.get_string()
            server_port = packet.get_int()
            self.client_id = packet.get_int()
            logging.info("Connection Established.")

            w = SocketPacket()
            w.add_int(VP_SESSION)
            w.add_int(self.client_id)
            w.add_char(SESSION_TYPE)
            w.add_char(self.pipeline)
            w.encode_header()
            self.write_packet(w)


        elif message_type == VV_VP_ACK_OK:
            logging.debug("VV_VP_ACK_OK")
            packet.get_string()
            packet.get_int()
            self.cam_id = packet.get_int()
            self.connected = True


        elif message_type == VV_REQ_VIDEO_ANALYSIS:
            logging.debug("VV_REQ_VIDEO_ANALYSIS")
            ip = packet.get_string()
            port = packet.get_int()
            self.camera_id = packet.get_int()
            #self.SetTitle("Camera %s Controller" % self.camera_id)


        elif message_type == VV_IMG:
            logging.debug("NEW IMAGE")
            server_ip = packet.get_string()
            server_port = packet.get_int()
            cam_id = packet.get_int()
            width = packet.get_int()
            height = packet.get_int()
            depth = packet.get_int()
            color_code = packet.get_char()
            jpeg =  packet.get_bool()
            time = packet.get_double()
            image = packet.get_string()

            cv_im = self.createImage(image, width, height, depth, color_code, jpeg)
            data = cv_im

            #cv_im = cv2.flip(cv_im, 0)
            self.camera_id = cam_id
            #cv2.imshow("Image", cv_im)

            # if self.save_images:
            #     self.image_num += 1
            #     cv2.imwrite("camera" + str(self.camera_id) + "_" + str(self.image_num) + ".png", cv_im)
                

                #cv.SaveImage("cam%s_%s.jpg" % (cam_id, self.count), cv_im)
                #self.count+=1
            #cv2.waitKey(0)
        return message_type, data

    def createImage(self, image_data, width, height, depth, color_code, jpeg=False):
        #print(image_data)
        if jpeg:
            length = len(image_data)
            nparr = np.fromstring(image_data, np.uint8)
            #nparr = nparr.reshape((768, 1024, 3))
            #image = cv.CreateMatHeader(1, length, cv.CV_8UC1)
            #cv.SetData(image, image_data, length)
            return cv2.imdecode(nparr,  cv2.IMREAD_COLOR)#cv2.IMREAD_COLOR)#nparr.reshape((1024, 768, 3))#cv.DecodeImage(image)
        else:
            image = cv.CreateImageHeader((width, height), depth, 4)
            cv.SetData(image, image_data)

    def initBackground(self):
        self.background = None
        self.getImage()
        messages = []
        d = []
        while 2 not in messages:
            messages, d = self.step()
            time.sleep(0.01)
        for i in range(len(messages)):
            if messages[i] == 2:
                image = d[i]
                self.background = image

    def MainLoop(self):
        #initialization
        messages = []
        while 0 not in messages or 9 not in messages or 5 not in messages:
            m, d = self.step()
            messages.extend(m)
            #print(messages)
            time.sleep(0.5)
        #self.initBackground()
        #print("initialized background")

    def classify(self, image):
        sensitivity = 0.2
        boxes, scores, classes, num = RD.getClassification(image)
        limit = 0
        for i in range(scores[0].shape[0]):
            limit = i
            if scores[0][i] < sensitivity:
                break
        nBoxes = boxes[0][0:limit]
        nScores = scores[0][0:limit]
        nClasses = classes[0][0:limit]
        for i, box1 in enumerate(nBoxes):
            x1 = min(1023,int(round(box1[1]*1023)))
            y1 = min(767,int(round(box1[0]*767)))
            x2 = min(1023,int(round(box1[3]*1023)))
            y2 = min(767,int(round(box1[2]*767)))
            print((x1, x2, y1, y2, i))
            #currentBoxes.append((x1, x2, y1, y2))
            image = self.drawBox(image, x1, x2, y1, y2, [0,255,0])
        return image

    def callImage(self):
        self.getImage()

    def receiveImage(self):
        messages = []
        d = []
        while 2 not in messages:
            messages, d = self.step()
            time.sleep(0.001)
        for i in range(len(messages)):
            if messages[i] == 2:
                #continue
                self.background = d[i]

    def controlInput(self, Input=0):
        if Input == 0:
            positionDictionary = {}
            if self.onoff:
                # self.getImage()
                # messages = []
                # d = []
                # while 2 not in messages:
                #     messages, d = self.step()
                #     time.sleep(0.01)
                # for i in range(len(messages)):
                #     if messages[i] == 2:
                #         #continue
                #         self.background = d[i]
                # self.getImage()
                messages = []
                d = []
                colorDict = {1:"Red",2:"Green",3:"Blue",4:"Yellow",5:"White",6:"Purple"}
                while 2 not in messages:
                    messages, d = self.step()
                    time.sleep(0.001)
                for i in range(len(messages)):
                    if messages[i] == 2:
                        boundaries = [
                            ([0,0,170],[80,80,255]),#red
                            ([190,190,0],[255,255,80]),#blue
                            ([0,170,0],[82,255,80]),#green
                            ([0,151,168],[108,255,255]),#yellow
                            ([190,0,0],[255,80,80]),#white
                            ([210,0,210],[255,80,255]),#brown
                        ]

                        currentBoxes = []
                        image = d[i]

                        newImage = np.copy(image)

                        cv2.absdiff(image, self.background, newImage)
                        newImage2 = np.copy(newImage)
                        newImage[newImage > 1] = 255
                        #stack = np.hstack((image, self.background, newImage2, newImage))
                        
                        newImage = cv2.cvtColor(newImage, cv2.COLOR_BGR2GRAY)
                        #print((image.shape, newImage.shape))
                        image = cv2.bitwise_and(image, image, mask=newImage.astype(np.uint8))
                        outputs = []
                        params = cv2.SimpleBlobDetector_Params()
                        params.minThreshold = 1
                        params.maxThreshold = 255
                        params.filterByArea = True
                        params.minArea = 400.0
                        params.filterByColor = False
                        params.filterByCircularity = False
                        params.filterByConvexity = False
                        params.filterByInertia = False
                        #params.filterByColor = True
                        detector = cv2.SimpleBlobDetector_create(params)
                        D = {0:"red",
                            1:"blue",
                            2:"green",
                            3:"yellow",
                            4:"white",
                            5:"purple"}
                        print("Camera " + str(self.camera_id))
                        found = []
                        for i, (lower, upper) in enumerate(boundaries):
                            lower = np.array(lower, dtype="uint8")
                            upper = np.array(upper, dtype="uint8")
                            mask = cv2.inRange(image, lower, upper)
                            output = cv2.bitwise_and(image, image, mask=mask)
                            output = cv2.cvtColor(output,cv2.COLOR_BGR2GRAY)
                            outputs.append(output)
                            keypoints = detector.detect(output)
                            if len(keypoints) > 0:
                                found.append(D[i])
                                
                                x,y = lowestPointFlood(output, keypoints[0].pt[0], keypoints[0].pt[1])
                                #print("Lowest Point: " + str(x) + "," + str(y))
                                half_fov = self.z/2
                                pan_angle = self.init_pan - self.p - (float(x)-512.0)/512.0*(half_fov)
                                tilt_angle = self.init_tilt - self.t + (float(y)-384.0)/384.0*(half_fov*768.0/1024.0)
                                r = self.init_z*1.0/math.tan(math.radians(tilt_angle))
                                #print("Angles: (" + str(pan_angle) + "," + str(tilt_angle) + "); r = " + str(r))
                                new_y = self.init_y - math.sin(math.radians(pan_angle))*r
                                new_x = self.init_x + math.cos(math.radians(pan_angle))*r
                                #print("Ralph " + D[i] + " at position " + str(new_x) + "," + str(new_y))
                                positionDictionary[i] = (new_x,new_y)
                        
                        print("Ralphs found: " + ",".join(found))
                        stack = np.vstack((np.hstack(tuple(outputs[0:3])),np.hstack(tuple(outputs[3:6]))))
                        #cv2.imshow("ImageMask" + str(self.camera_id), stack)
                        #cv2.waitKey(0)
                        #image = self.classify(image)
                        if self.save_images:
                            #cv2.imshow("Image"+str(self.camera_id), image)
                            #cv2.waitKey(0)
                            self.image_num += 1
                            #cv2.imwrite("camera" + str(self.camera_id) + "_2"+ str(self.image_num) + ".png", stack)
                        break
                #print("wrote image")
            return positionDictionary
        elif Input == 1:
            self.pan(True)
        elif Input == 2:
            self.pan(False)
        elif Input == 3:
            self.tilt(True)
        elif Input == 4:
            self.tilt(False)
        elif Input == 5:
            self.zoom(True)
        elif Input == 6:
            self.zoom(False)
        elif Input == 7:
            self.onoff = not self.onoff
        elif Input == 8:
            return None
        else:
            self.onoff = not self.onoff
        return None

    def drawBox(self, img, x1, x2, y1, y2, colors):
        R = colors[0]
        G = colors[1]
        B = colors[2]
        img[y1:y2, x1, 0] = R
        img[y1:y2, x1, 1] = G
        img[y1:y2, x1, 2] = B
        img[y1:y2, x2, 0] = R
        img[y1:y2, x2, 1] = G
        img[y1:y2, x2, 2] = B
        img[y1, x1:x2, 0] = R
        img[y1, x1:x2, 1] = G
        img[y1, x1:x2, 2] = B
        img[y2, x1:x2, 0] = R
        img[y2, x1:x2, 1] = G
        img[y2, x1:x2, 2] = B
        return img

def lowestPointFlood(image, x, y):
    lowest_x = x
    lowest_y = y
    rows = len(image)
    cols = len(image[0])
    stack = [(x,y)]
    visited = {}
    while stack:
        point = stack.pop()
        if point in visited:
            continue
        else:
            visited[point] = True
        if point[1] > lowest_y:
            lowest_y = point[1]
            lowest_x = point[0]
        if checkBounds(rows, cols, x+1, y):
            stack.append((x+1, y))
        if checkBounds(rows, cols, x, y+1):
            stack.append((x, y+1))
        if checkBounds(rows, cols, x-1, y):
            stack.append((x-1, y))
        if checkBounds(rows, cols, x, y-1):
            stack.append((x, y-1))
    return (lowest_x, lowest_y)

def checkBounds(rows, cols, x, y):
    return (x >= 0 and y >= 0 and y < rows and x < cols)


class RalphDetector:
    def __init__(self, graphFile):
        PATH_TO_MODEL = graphFile
        self.detection_graph = tf.Graph()
        with self.detection_graph.as_default():
            od_graph_def = tf.GraphDef()
            with tf.gfile.GFile(PATH_TO_MODEL, 'rb') as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name='')
            self.image_tensor = self.detection_graph.get_tensor_by_name('image_tensor:0')
            self.d_boxes = self.detection_graph.get_tensor_by_name('detection_boxes:0')
            self.d_scores = self.detection_graph.get_tensor_by_name('detection_scores:0')
            self.d_classes = self.detection_graph.get_tensor_by_name('detection_classes:0')
            self.num_d = self.detection_graph.get_tensor_by_name('num_detections:0')
        self.sess = tf.Session(graph=self.detection_graph)

    def getClassification(self, img):
    # Bounding Box Detection.
        with self.detection_graph.as_default():
            # Expand dimension since the model expects image to have shape [1, None, None, 3].
            img_expanded = np.expand_dims(img, axis=0)  
            (boxes, scores, classes, num) = self.sess.run(
                [self.d_boxes, self.d_scores, self.d_classes, self.num_d],
                feed_dict={self.image_tensor: img_expanded})
        return boxes, scores, classes, num


def getState(clients, positions):
    state = []
    #camera positions
    for client in clients:
        on = 1 if client.onoff else 0
        state.extend([client.p*1.0/30, client.t*1.0/30, client.z*1.0/100, on])
    #camera
    # for position in positions:
    #     state.extend([position[0], position[1]])
    #print(state)
    return state

def controller(clients):
    #RD = RalphDetector("frozen_inference_graph.pb")
    #print("Loaded Detector")
    for client in clients:
        client.MainLoop()
    positions = [(-1000,-1000)]*6
    state = getState(clients, positions)
    nextState = state
    rewards = []
    rewardsSeen = []
    while True:
        state = nextState
        #get state

        
        # choose actions

        randomAction = False
        if randomAction:
            clientNum = randint(0,9)
            client = clients[clientNum]
            action = randint(1,16)
            if action > 8:
                action = 7
            client.controlInput(action)
            actionNum = clientNum*8+(action-1)
            assert(0 <= actionNum < 80)
            print("Camera " + str(clientNum+1) + ", Action " + str(action))
        else:
            sess1 = tf.Session()
            saver = tf.train.import_meta_graph('./model_a_0_b_1/model_a_0_b_1.meta', clear_devices=True)
            saver.restore(sess1, tf.train.latest_checkpoint('./model_a_0_b_1'))

            graph = tf.get_default_graph()
            x1 = graph.get_tensor_by_name('s:0')
            y1 = graph.get_tensor_by_name('eval_net/l3/output:0')

            npState = np.array([state])

            with tf.Session() as sess:
                sess.run(tf.global_variables_initializer())
                y_out = sess.run(y1, feed_dict = {x1:npState})
                print("y_out length: " + str(y_out.shape))
                #print(y_out)
                #print(str(y_out.shape))

            clientActions = y_out[0]
            for i in range(10):
                print(clientActions[i*8:i*8+8])
            # actionNum = np.argmax(clientActions)
            # client = clients[actionNum/8]
            # action = actionNum%8
            # client.controlInput(action)
            # print("Camera " + str(actionNum/8+1) + ", Action " + str(actionNum%8))
            for i in range(len(clients)):
              clientActions = y_out[0][i*8:i*8+8]
              client = clients[i]
              action = np.argmax(clientActions)+1
              client.controlInput(action)
              actionNum = i*8+action


        #wait

        #note reward

        #get state

        print("Getting images...")

        positions = [(-1000,-1000)]*6 #no positions
        for client in clients:
            if client.onoff:
                client.callImage()

        for client in clients:
            if client.onoff:
                client.receiveImage()

        for client in clients:
            if client.onoff:
                client.callImage()

        camerasOn = 0
        for client in clients:
            if client.onoff:
                camerasOn += 1
                positionDictionary = client.controlInput(0)
                for color in positionDictionary: #color is a number, 1 for red, etc
                    positions[color] = positionDictionary[color]
                #boxes, scores, classes, num = RD.getClassification(img2)


        seen = False
        peopleSeen = 0
        for position in positions:
            if position[0] != -1000 or position[1] != -1000:
                seen = True
                peopleSeen += 1


        if not seen:
            with open('r.csv', 'a') as csvfile:
                statewriter = csv.writer(csvfile, delimiter=',')
                print("Writing rewards file...")
                rewards = [sum(rewards)*1.0/len(rewards)] + rewards
                statewriter.writerow(rewards)
                rewardsSeen = [sum(rewardsSeen)*1.0/len(rewardsSeen)] + rewardsSeen
                statewriter.writerow(rewardsSeen)
            break
        rewards.append(camerasOn)
        rewardsSeen.append(peopleSeen)
        print("\n\n##################################")
        print("Cameras on: " + str(camerasOn))
        print("People tracked: " + str(peopleSeen))
        print("##################################\n\n")
        nextState = getState(clients, positions)
        #save state, next state
        with open('x.csv', 'a') as csvfile:
            statewriter = csv.writer(csvfile, delimiter=',')
            #print("Writing state file...")
            #print(state)
            statewriter.writerow(state)

        with open('a.csv', 'a') as csvfile:
            statewriter = csv.writer(csvfile, delimiter=',')
            #print("Writing action file...")
            #print(actionNum)
            statewriter.writerow([actionNum])

        with open('z.csv', 'a') as csvfile:
            statewriter = csv.writer(csvfile, delimiter=',')
            #print("Writing next state file...")
            #print(nextState)
            statewriter.writerow(nextState)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-a', '--address', dest='address', default='localhost',
        help='set the IP address of the virtual world')
    parser.add_argument('-p', '--port', dest='port', type=int, default=9099,
        help='the port of the virtual world')
    parser.add_argument('-s', '--save-images', dest='save_images',
        action='store_true', default=False,
        help='save the images received from the server')
    parser.add_argument('--static', dest='pipeline', action='store_const',
        const=STATIC_PIPELINE, default=ACTIVE_PIPELINE)
    parser.add_argument('--debug', dest='debug', action='store_const',
        const=logging.DEBUG, default=logging.INFO, help='show debug messages')
    args = parser.parse_args()

    logging.basicConfig(format='%(levelname)s: %(message)s', level=args.debug)

    #app = wx.App()
    clients = []
    D = {0:(-90.0, 26.565, -575, -320),
        1:(90, 26.565, -575, 320),
        2:(90, 26.565, 575, 300),
        3:(-90, 26.565, 575, -320),
        4:(90, 26.565, -800, 120),
        5:(45, 26.565, -350, 100),
        6:(-45, 26.565, -350, -100),
        7:(-45, 26.565, 50, -100),
        8:(-26.565, 26.565, -525, -320),
        9:(26.565, 26.565, -525, 320)}
    for i in range(10):
        init_angle = D[i]
        client = Client(None, -1, 'Sample Client', args.address, args.port,
        args.pipeline, init_angle[0], init_angle[1], init_angle[2], init_angle[3], save_images=args.save_images)
        clients.append(client)
    controller(clients)


if __name__ == "__main__":
    main()
