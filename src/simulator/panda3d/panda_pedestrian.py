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


from math import sqrt, degrees, atan2

import numpy as np

from pandac.PandaModules import Vec3, VBase3
from direct.interval.ActorInterval import ActorInterval
from direct.interval.IntervalGlobal import Func, Parallel, Sequence, Wait

from panda3d_helper import *
from panda3d.core import Point3

HEADING_X = 225
HEADING_X_FEMALE = 0

def Length(a, b):
    ax, ay, az = a
    bx, by, bz = b
    length = (ax - bx) ** 2 + (ay - by) ** 2
    return length


class Action:


  def __init__(self, name, rotation, diff):
    self.name = name
    self.rotation = rotation
    self.diff = diff


class PandaPedestrian:


    def __init__(self, actor, texture, taskMgr, joint):
        self.taskMgr = taskMgr
        self.actor = actor
        self.texture = texture
        self.joint = self.actor.exposeJoint(None, 'modelRoot', joint)
        
        self.animations = {}
        self.actions = {}
        self.command_queue = []
        self.cur_action = 'stand'
        
        self.type = "male"
        self.id = -1
        self.pos = np.array([0,0,0])
        self.loop = True
        self.sequence = Parallel()
        
        self.live_controller = False
        self.print_action = False
        self.start_time = 0.0

    
    def getActor(self):
        return self.actor
        
      
    def getTexture(self):
        return self.texture
        
        
    def setColor(self, color):
        pass
        
        
    def setId(self, id):
        self.id = id
        
        
    def getId(self):
        return self.id
    
    
    def setType(self, type):
        self.type = type
    
    
    def getType(self):
        return self.type


    def getBounds(self):
        return self.actor.getTightBounds()


    def addAnimation(self, name, path):
        self.animations[name] = path
        self.actor.loadAnims({name:path})


    def addAction(self, name, angle, diff):
        self.actions[name] = Action(name, angle, VBase3(*diff))


    def addCommand(self, command):
        if command in self.actions:
            self.command_queue.append(command)
        else:
            print "Error: invalid command"


    def addCommands(self, commands):
        for command in commands:
            self.addCommand(command)


    def setStartTime(self, time):
        self.start_time = time


    def update(self, time):
        if self.cur_action:
            name = self.cur_action
            anim_controller = self.actor.getAnimControl(name)
            next_frame = anim_controller.getNextFrame()
            cur_frame = anim_controller.getFullFframe()
            num_frames = anim_controller.getNumFrames()

            if not self.sequence.isPlaying():
#            if not anim_controller.isPlaying():
                if self.command_queue:
                    next_action = self.command_queue.pop(0)
                    self.startAction(next_action)
                else:
                    if self.live_controller:
                        if self.cur_action == 'run':
                            next_action = 'run'
                        elif self.cur_action == 'stand':
                            next_action = 'stand'
                        else:
                            next_action = 'walk'
                    
                        self.startAction(next_action)
                    else:
                        next_action = 'walk'
                        self.startAction(next_action)

    
    def isActive(self, time):
        #print(self.live_controller)
        if time >= self.start_time and self.command_queue or \
           time >= self.start_time and self.live_controller:
            if self.actor.isHidden():
                self.actor.show()
            return True
        else:
            return False
            # if not self.actor.isHidden():
            #     self.actor.hide()
            # return False


    def reposition(self, x, y, z, rotation):
        #x,y,z = diff
        #x1, y1, z1 = pos
        #print((x, y, z, rotation))
        #print((interval, x1, y1, z1, x,y,z))
        #return self.actor.posInterval(Point3(0, 0, 0), duration=3.0)
        #return self.actor.posInterval(interval, Point3(x+x1, y+y1, z+z1))
        self.actor.setX(x)
        self.actor.setY(y)
        self.actor.setH(self.actor.getH() + rotation)
#        name = self.cur_action
#        anim_controller = self.actor.getAnimControl(name)
#        cur_frame = anim_controller.getFullFframe()
#        print cur_frame


    def liveController(self):
            self.print_action = True
            self.live_controller = True


    def startAction(self, action):
        cur_action = self.cur_action
        rotation = self.actions[cur_action].rotation
        self.cur_action = action
        paction = self.actions[action]
        
        if cur_action != "stand":
            diff = self.joint.getPos(render) - paction.diff#self.pos = diff
        else:
            diff = self.actor.getPos()
            
        if self.print_action:
            print paction.name,
        x, y, z = diff
        interval = self.actor.actorInterval(paction.name)
        func = Func(self.reposition, x, y, z, rotation)
        func1 = self.actor.posInterval(interval.duration, Point3(x, y, z))
        func2 = self.actor.hprInterval(interval.duration, Vec3(self.actor.getH() + rotation, 0,0))
        #self.sequence=Parallel(Sequence(func1), interval)
        self.sequence = Parallel(func2, func1, interval)
        self.sequence.start()

#        self.actor.play(paction.name)
#        self.taskMgr.doMethodLater(0.0, self.reposition, "test", [diff, rotation])

