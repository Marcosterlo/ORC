Code repository of Advanced Optimization-Based Robot Control

Comando docker per runnare l'environment: 

'''
sudo docker run  -v /tmp/.X11-unix/:/tmp/.X11-unix/ -v /home/marco/Desktop/ORC:/home/student/mint --name ubuntu_bash --env="DISPLAY=$DISPLAY" --privileged -p 127.0.0.1:7000:7000 --shm-size 2g --rm -i -t --user=student     --workdir=/home/student andreadelprete/orc23:marco bash
'''

Fix dependencies per fare funzionare spyder:

Dal comando docker sostituire come environmental variable --env "DISPLAY=$DISPLAY"

runnare poi

Per fare un commit locale dell'immagine di docker aprire un altro terminale mentre sta runnando docker, con '''sudo docker ps''' prendere l'id del container e poi eseguire
'''
sudo docker commit ubuntu_bash andreadelprete/orc23:marco
'''

Per aggiungere vscode:
'''sudo apt update && sudo apt-get install wget gpg &&
wget -qO- https://packages.microsoft.com/keys/microsoft.asc | gpg --dearmor > packages.microsoft.gpg && 
sudo install -D -o root -g root -m 644 packages.microsoft.gpg /etc/apt/keyrings/packages.microsoft.gpg &&
sudo sh -c 'echo "deb [arch=amd64,arm64,armhf signed-by=/etc/apt/keyrings/packages.microsoft.gpg] https://packages.microsoft.com/repos/code stable main" > /etc/apt/sources.list.d/vscode.list' &&
rm -f packages.microsoft.gpg &&
sudo apt install apt-transport-https &&
sudo apt update &&
sudo apt install code &&
sudo apt install terminator ranger vim
'''
