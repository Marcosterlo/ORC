Code repository of Advanced Optimization-Based Robot Control

Comando docker per runnare l'environment: 
'''
sudo docker run  -v /tmp/.X11-unix/:/tmp/.X11-unix/ -v /home/marco/Desktop/ORC:/home/student/mint --name ubuntu_bash --env="DISPLAY=$DISPLAY" --privileged -p 127.0.0.1:7000:7000 --shm-size 2g --rm -i -t --user=student     --workdir=/home/student andreadelprete/orc23:marco bash
'''

Fix dependencies per fare funzionare spyder:

Dal comando docker sostituire come environmental variable --v "DISPLAY=$DISPLAY"

runnare poi

'''
sudo apt update
sudo apt install pyqt5-dev-tools python3-pyqt5.qtsvg python3-pyqt5-qtwebengine
pip install spyder
'''

Per fare un commit locale dell'immagine di docker aprire un altro terminale mentre sta runnando docker, con '''sudo docker ps''' prendere l'id del container e poi eseguire
'''
sudo docker commit __id del container__ andreadelprete/orc23:marco
'''
