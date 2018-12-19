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


import sys, os
sys.path.append(os.path.join(os.getcwd(), '..' ))
import cv2

from socket_packet import SocketPacket
from message_handler import *



class SyncSession():
    def __init__(self, conn_id, connection, server, type):
        self.conn_id  = conn_id
        self.type = type
        self.client_address = connection.getAddress()
        self.conn = connection
        self.server = server
        self.message_queue = []


    def getMessages(self, cam_id=None):
        messages = []
        while self.message_queue:
            messages.append(self.message_queue.pop(0))
        return messages


    def sendMessage(self, message):
        self.server.sendMessage(message, self.conn)


    def newMessage(self, type, message):
        self.message_queue.append((type, message))
        
    def getSessionType(self):
        return 2
