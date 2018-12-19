import xml.etree.cElementTree as ET
from random import randint
import random
import sys
print(sys.argv[1])
def generateRandomPath(init_x, init_y, heading):
	actions = 40
	path = ""
	x = init_x-50
	y = init_y
	turnDict = {
		0: ('l90', -90),
		1: ('l45', -45),
		2: ('r45', 45),
		3: ('r90', 90)
	}
	walkDict = {
		0: ("walkleft", 0, -50),
		45: ("walkleftforward", 30, -30),
		90: ("walk", 50, 0),
		135: ("walkrightforward", 30, 30),
		180: ("walkright", 0, 50),
		225: ("walkrightback", -30, 30),
		270: ("walkback", -50, 0),
		315: ("walkleftback", -30, -30)
	}
	while actions > 0:
		turn = randint(0,9)
		if turn == 0:
			turnint = randint(0,3)
			action, deltaHeading = turnDict[turnint]

			path += action + " "
			actions -= 1
			heading += deltaHeading
			heading = heading % 360
		else:
			walkAction, delta_x, delta_y = walkDict[heading]
			if validMove(x+delta_x, y+delta_y):
				path += walkAction + " "
				x += delta_x
				y += delta_y
				actions -= 1
	return path[:-1]


def validMove(new_x, new_y):
	if new_x > 649 or new_x < -800 or new_y > 325 or new_y < -325:
		return False
	if (new_y >= -270 and new_y <= -100 and new_x >= -425 and new_x <= 425): #south wall
		return False
	if (new_y <= 270 and new_y >= 100 and new_x >= -425 and new_x <= 425): #north wall
		return False
	#side walls
	if ((new_y >= -100 and new_y <= -50) or (new_y >= 50 and new_y <= 100)):
		if ((new_x >= -425 and new_x <= -375) or (new_x >= 375 and new_x <= 425)):
			return False
	#mid wall
	if (new_y >= -100 and new_y <= 100 and new_x <= 25 and new_x >= -25):
		return False
	return True


randomLocations = [(-800, 0), (-575, 0), (-600, -280), (-750, 200), (0, 290), (0, -290), (575, 0), (575, 60), (200, 50), (200, -50), (-200, 50), (-200, -50)]
locationList = random.sample(range(len(randomLocations)), 6)

ped = ET.Element("pedestrians")
greenPos = str(randomLocations[locationList[0]][0]) + " 0 " + str(randomLocations[locationList[0]][1])
bluePos = str(randomLocations[locationList[1]][0]) + " 0 " + str(randomLocations[locationList[1]][1])
yellowPos = str(randomLocations[locationList[2]][0]) + " 0 " + str(randomLocations[locationList[2]][1])
brownPos = str(randomLocations[locationList[3]][0]) + " 0 " + str(randomLocations[locationList[3]][1])
whitePos = str(randomLocations[locationList[4]][0]) + " 0 " + str(randomLocations[locationList[4]][1])
redPos = str(randomLocations[locationList[5]][0]) + " 0 " + str(randomLocations[locationList[5]][1])

green = ET.SubElement(ped, "pedestrian", character="ralphGreen", texture="ralphGreen.png",
						pos=greenPos, scale="10", hpr="90 0 0")
path = generateRandomPath(randomLocations[locationList[0]][0], randomLocations[locationList[0]][1], 90)
commands = ET.SubElement(green, "commands", start_time="0").text=path

blue = ET.SubElement(ped, "pedestrian", character="ralphBlue", texture="ralphBlue.png",
						pos=bluePos, scale="10", hpr="90 0 0")
path = generateRandomPath(randomLocations[locationList[1]][0], randomLocations[locationList[1]][1], 90)
commands = ET.SubElement(blue, "commands", start_time="0").text=path

yellow = ET.SubElement(ped, "pedestrian", character="ralphYellow", texture="ralphYellow.png",
						pos=yellowPos, scale="10", hpr="90 0 0")
path = generateRandomPath(randomLocations[locationList[2]][0], randomLocations[locationList[2]][1], 90)
commands = ET.SubElement(yellow, "commands", start_time="0").text=path

brown = ET.SubElement(ped, "pedestrian", character="ralphBrown", texture="ralphBrown.png",
						pos=brownPos, scale="10", hpr="90 0 0")
path = generateRandomPath(randomLocations[locationList[3]][0], randomLocations[locationList[3]][1], 90)
commands = ET.SubElement(brown, "commands", start_time="0").text=path

white = ET.SubElement(ped, "pedestrian", character="ralphWhite", texture="ralphWhite.png",
						pos=whitePos, scale="10", hpr="90 0 0")
path = generateRandomPath(randomLocations[locationList[4]][0], randomLocations[locationList[4]][1], 90)
commands = ET.SubElement(white, "commands", start_time="0").text=path

red = ET.SubElement(ped, "pedestrian", character="ralph", texture="ralph.png",
						pos=redPos, scale="10", hpr="90 0 0")
path = generateRandomPath(randomLocations[locationList[5]][0], randomLocations[locationList[5]][1], 90)
commands = ET.SubElement(red, "commands", start_time="0").text=path

tree = ET.ElementTree(ped)
tree.write("../config/sample/pedestrians" + str(sys.argv[1]) + ".xml")
