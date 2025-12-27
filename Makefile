PROJECT_NAME=NetBelief
TOPOLOGIES_URL=https://topology-zoo.org/files/archive.zip
TOPOLOGIES_DIR=./topologies
PYTHON_TOPOLOGIES_DIR=./pytopo
TOPOLOGIES_ARCHIVE_FILE=./archive.zip

# GraphML conversion stuff
PARSER_REPO=https://github.com/uniba-ktr/assessing-mininet.git
PARSER_DIR=./assessing-mininet
PARSER_REPO_PATH=$(PARSER_DIR)/parser/GraphML-Topo-to-Mininet-Network-Generator.py
PARSER_PATH=scripts.graphml-converter

# Serious business
download-topologies:
	@if [ ! -f $(TOPOLOGIES_ARCHIVE_FILE) ]; then \
		echo "Downloading the Topology Zoo archive...."; \	
		curl -L -o $(TOPOLOGIES_ARCHIVE_FILE) $(TOPOLOGIES_URL); \
	else \
		echo "Archive $(TOPOLOGIES_ARCHIVE_FILE) already exists."; \
	fi

	@if [ ! -d $(TOPOLOGIES_DIR) ]; then \
		echo "Creating directory $(TOPOLOGIES_DIR)..."; \
		mkdir -p $(TOPOLOGIES_DIR); \
	fi

	@echo "Extracting topologies to $(TOPOLOGIES_DIR)" && \
		unzip -q $(TOPOLOGIES_ARCHIVE_FILE) -d $(TOPOLOGIES_DIR);


install:
	@uv sync

	@echo "Downloading GraphML parser"
	@if [ ! -d $(PARSER_DIR) ]; then \
		git clone $(PARSER_REPO); \
		rm -fr $(PARSER_DIR)/.git; \
		cp $(PARSER_REPO_PATH) $(PARSER_PATH) && uv run 2to3 -w $(PARSER_PATH)
	fi

	@echo "Installing networking tools"
	@if ! command -v wireshark &> /dev/null; then sudo pacman -S wireshark-qt; fi
	@if ! command -v sipp &> /dev/null; then yay -S sipp; fi
	@if ! command -v tcpdump &> /dev/null; then sudo pacman -S tcpdump; fi

graph2py:
	@echo "Converting the GraphML topology to Python"
	@uv run -m $(PARSER_PATH) -f $(TOPOLOGIES_DIR)/$(topo).graphml --output $(PYTHON_TOPOLOGIES_DIR)/$(out).py

	@echo "Porting the Python2 code to Python3"
	@uv run 2to3 -w $(PYTHON_TOPOLOGIES_DIR)/$(out).py
	@rm $(PYTHON_TOPOLOGIES_DIR)/*.bak

run-topology:
	@sudo mn --custom $(topo) --topo generated --link=TCLink --controller=remote


# terminates all processes related to Mininet to allow running further simulations
clear-mininet:
	@sudo mn -c

# Ubuntu VM related stuff
vmdk-2-qcow2:
	@qemu-img convert -f vmdk -O qcow2 $(source) $(dest) 

list-virt-networks:
	@sudo virsh net-list --all

set-network-to-autostart:
	@sudo virsh net-autostart default

change-vm-keyboard:
	@sudo dpkg-reconfigure keyboard-configuration
	@echo "You must reboot the VM"

install-ui:
	@sudo apt update && sudo apt upgrade && sudo apt install ubuntu-desktop-minimal
	