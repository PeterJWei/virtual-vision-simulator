#!/bin/bash

iterations=$1
starterPath=../config/sample/pedestrians
extension=".xml"
while ((iterations > 0))
do
	python generatePedestrianXML.py $iterations
	newPath="$starterPath$iterations$extension"
	echo $newPath
	cp $newPath ../config/sample/pedestrians.xml
	gtimeout 250 python 3D_Simulator.py -d ../config/sample/ &
	sleep 5
	((iterations = iterations-1))
	gtimeout 245 python Peter_client.py -p 9099 -s &&
	echo $iterations
done