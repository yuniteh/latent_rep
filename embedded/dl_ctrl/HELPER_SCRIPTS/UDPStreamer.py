import socket
import threading
import time
import os
import numpy as np
import pcepy.can as can
import pcepy.pce as pce
if os.name == 'nt':
    host = '127.0.0.1'
else:
    host = '192.168.20.2'
    
# Define port number and create socket connection.
port = 27015
# Define the number of bytes to be read.
sockbyte_read = 256
# Define pause time.
sockpause_active = 0.025
sockpause_idle = 1
# Define socket timeout value (seconds).
socktimeout = 1
# Establish global connection bool.
connection = False
# Bool to indicate when data is streaming.
streaming = False


#######################################################################################################################
# Function    : stringFormatter(args)
# args        : dataArray, a numpy row-based array: dataType, integer = %d, float with 3dp = %.3f.
# Description : This function formats numpy vectors into a single-line string with semicolon separated values.
#######################################################################################################################
def stringFormatter(dataArray, dataType):
    # Check for data shape. Column based vectors must be transposed.
    if (dataArray.shape[0] == 1):
        tmpData = np.transpose(dataArray)
    else:
        tmpData = dataArray
    # Format data.
    formattedString = ';'.join([dataType % num for num in tmpData])
    return formattedString


#######################################################################################################################
# Function    : GrabData(args)
# args        : sock, connection socket.
# Description : This function grabs data over the socket and handles errors if the socket timesout.
#######################################################################################################################
def GrabData(sock):
    data = ""
    global connection
    # Try to read a single byte. If it fails due to a timeout, close connection.
    try:
        data = sock.recv(sockbyte_read)
    except socket.timeout:
        print("Communication timeout. Closing connection.")
        connection = False
    # Return message.
    return data


#######################################################################################################################
# Function    : ArraySizes(args)
# args        : none.
# Description : This function returns, in a semicolon deliminated string, the size of each array of interest.
#######################################################################################################################
def ArraySizes():
    # Determine and format the number of classes (based off size of N_T) and channel size (based of CHAN_MAV)
    class_SIZE = str(pce.get_var('N_T').to_np_array().shape[1])
    chan_SIZE = str(len(pce.get_var('CHAN_MAV').to_np_array()))
    # Format array sizes into concatenated string.
    arraySize = ";" + class_SIZE + "," + chan_SIZE
    # Return string.
    return arraySize

    
#######################################################################################################################
# Function    : readwrite(args)
# args        : type, type of connection; read or write.
# Description : This function either performs a read or write from the socket based on the type input.
#######################################################################################################################
def readwrite(type):
    while connection:
        # Sleep so CAPS can keep up.
        time.sleep(sockpause_active)
        # Send
        if type == 'send':
            sendData()
        # Receive
        if type == 'read':
            receiveData()
    

############################################################################
## SEND THREAD
############################################################################
def ThreadJob_A():
    global connection
    # Loop indefinitely.
    while 1:
        # If connection isn't active, sleep.
        if not connection:
            time.sleep(sockpause_idle)
        # Continuous loop while connection active.
        while connection:
            readwrite('send')

        
############################################################################
## SEND FUNCTION
############################################################################
def sendData():
    global connection
    # Get class prediction (CLASS_EST) and whether a new class has just been trained (NEW_CLASS).
    class_out = pce.get_var('CLASS_EST')
    class_new = pce.get_var('NEW_CLASS')
    # Get proportional control information and format into comma deliminated string. Shape[1] of N_T is used to get the number of modes (typically 7).
    class_pc = ';'.join(['%.5f' % num2 for num2 in [pce.get_var('PROP_CONTROL')[num] for num in range(0, pce.get_var('N_T').to_np_array().shape[1])]])
    # Get the total number of patterns (N_C), number of repetition (N_R), and temporary number of patterns (N_T).
    ntot = stringFormatter(pce.get_var('N_C').to_np_array(), '%d')
    nreps = stringFormatter(pce.get_var('N_R').to_np_array(), '%d')
    npats = stringFormatter(pce.get_var('N_T').to_np_array(), '%d')
    # Get position calibration flag
    pos_flag = pce.get_var('PD_FLAG')
    
    # If data is streaming, send data, otherwise send 'nil'.
    if streaming:
        # Create single length string to transmit.
        # Send calibrated arm positions if required.
        if pce.get_var('SEND_PD') == 1:
            # Get calibrated arm positions and rotations
            pos = stringFormatter(pce.get_var('POS').to_np_array().flatten(),'%.1f')
            rot = stringFormatter(pce.get_var('ROT').to_np_array().flatten(),'%d')
            sendString = "C_OUT=" + str(class_out) + ",POS=" + pos + ",ROT=" + rot
            pce.set_var('SEND_PD',0)
        else:
            sendString = "C_OUT=" + str(class_out) + ",C_NEW=" + str(class_new) + ",C_PC=" + str(class_pc) + ",N_C=" + ntot + ",N_R=" + nreps + ",N_T=" + npats + ",P_FLAG=" + str(pos_flag)
    else:
        sendString = "nil"

    # Try and send message, if it fails print message.
    try:
        sock.sendto(sendString, (addr))
    except:
        connection = False
        print("Error in message transmission")


############################################################################
## RECEIVE THREAD
############################################################################
def ThreadJob_B():
    global connection
    # Loop indefinitely.
    while 1:
        # If connection isn't active, sleep.
        if not connection:
            time.sleep(sockpause_idle)
        # Continuous loop while connection active.
        while connection:
            readwrite('read')


############################################################################
## RECEIVE FUNCTION
############################################################################
def receiveData():
    # Get global variables.
    global streaming
    # Get data from embedded system.
    data = GrabData(sock);

    # HEARTBEAT
    # Message pushed frequently to indicate the connection is still live.
    if data == "HeartBeat":
        print("HeartBeat")
        
    # DISCONNECT
    # If the message 'Disconnect' is sent, then close connection.
    elif data == "Disconnect":
        sock.close()
        print("Connection Closed")

    # CAN
    # If the message contains 'can.send' a CAN message is being sent to be pushed to the embedded system.
    elif "can|send" in data:
        print("Sending CAN message")
        # NID is the first hex number of the can.send string
        nid = data.split('|')[2]
        # Set NID, priority, and mode.
        canmessenger.set_nid(int(nid,16)) 
        canmessenger.set_priority(0x00)
        canmessenger.set_mode(0x01)
        # Bits in string format
        bits_str = data.split('|')[3:10]
        
        # Get CAN data into a presentable form.
        candata = np.zeros((5,), dtype = np.int)
        # Put bits in candata, converted from string to int
        for iDigit in range(0,len(bits_str)):
            candata[iDigit] = int(bits_str[iDigit],16)                
        try:
            # Attempt to send data.
            canmessenger.set_data(candata[0], candata[1], candata[2], candata[3], candata[4])
            canmessenger.send()
            # Succeeded in sending message. Print to screen and send success handshake.
            print("     CAN send value sent: " + candata)
            sock.sendto('ACKCON=1:' + data, (addr))
        except:
            # Failed to send message. Print to screen and send failed handshake.
            print("     Error: CAN send value not sent")
            sock.sendto('ACKCON=0:' + data, (addr))
            pass

    # PCE
    # If the message contains 'pce.set', a PCE message is being sent to be set on the embedded system.
    elif "pce|set" in data:
        print("Sending PCE message")
        # Bits in string format
        bits_str = data.split('|')

        # If the data contains 'cmd', then this is a command instruction.
        if "cmd" in data:
            try:
                # Attempt to sent the message to the embedded system.
                pce.send_cmd(int(bits_str[3]))
                # If the command was 3 (start streaming), then set bool to true.
                if int(bits_str[3]) == 3:
                    streaming = True
                # If the command was 4 (stop streaming), then set bool to false.
                if int(bits_str[3]) == 4:
                    streaming = False
                # Succeeded in sending message. Print to screen and send success handshake.
                print("     PCE cmd value sent: " + bits_str[3])
                sock.sendto('ACKCON=1:' + data, (addr))
            except:
                # Failed to send message. Print to screen and send failed handshake.
                print("     Error: PCE cmd value not sent")
                sock.sendto('ACKCON=0:' + data, (addr))
                pass

        # If the data contains 'var', then this is a variable on the embedded system to be updated.
        elif "var" in data:
            try:
                # todo: improve this method to determine if a string or int is to be sent.
                if "NAME" in bits_str[3]:
                    pce.set_var(bits_str[3], bits_str[4])
                elif ("POS" in bits_str[3]) or ("ROT" in bits_str[3]):
                    current = pce.get_var(bits_str[3]).to_np_array()
                    for i in range(3):
                        current[int(bits_str[4]),i] = float(bits_str[i+5])
                    pce.set_var(bits_str[3], current)
                else:
                    pce.set_var(bits_str[3], int(bits_str[4]))
                # Succeeded in sending message. Print to screen and send success handshake.
                print("     PCE var value sent: " + bits_str[4])
                sock.sendto('ACKCON=1:' + data, (addr))
            except:
                # Failed to send message. Print to screen and send failed handshake.
                print("     Error: PCE var value not sent")
                sock.sendto('ACKCON=0:' + data, (addr))
                pass

    # Update log
    # If the message contains 'log.set', a new line of progress is written to a log file.
    elif "log|set" in data:
        print("Updating log file")
        try:
            # Break data at the full stops.
            splitString = data.split('|')
            # The third block contains the location, the fourth the data.
            logLoc = splitString[2] + ".txt"
            logDat = splitString[3]
            # Write data to file ('a' for 'append', '+' creates the file if it doesn't exist).
            with open(logLoc, 'a+') as f:
                f.write(logDat + '\n')
            # Succeeded in sending message. Print to screen and send success handshake.
            print("     Log updated")
            sock.sendto('ACKCON=1:' + data, (addr))
        except:
            # Failed to send message. Print to screen and send failed handshake.
            print("     Error: Log not updated")
            sock.sendto('ACKCON=0:' + data, (addr))
            pass


#######################################################################################################################
## MAIN
#######################################################################################################################  
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
# Set socket options. This allows for port/addresses to be reused (in theory).
sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
# Bind the socket to the address specified above.
sock.bind((host, port))
# Create the threads to run the reading and writing.
threading.Thread(target=ThreadJob_A).start()
threading.Thread(target=ThreadJob_B).start()

# Loop indefinitely.
while 1:
    # Print message to indicate ready for connection.
    print("UDPStreamer: Ready to Connect")
    # Set initial timeout to none so that recvfrom can hang until client communicates
    sock.settimeout(None)
    # Grab packet from client. This line will hang until client sends a message.
    data, addr = sock.recvfrom(sockbyte_read)
    # Set a message back to client to indicate the server is awake with array sizes concatenated.
    sock.sendto("ALIVE" + ArraySizes(), (addr))
    # Set timeout so that subsequent data grabs will timeout if client dies.
    sock.settimeout(socktimeout)
    # Print message to indicate the socket is set up.
    print("UDPStreamer: Connected to " + str(addr))
    # Once connected, set connection to true.
    connection = True
    
    # Loop until connection is no longer true.
    while connection:
        # Sleep for a second so that the system isn't overloaded.
        time.sleep(1)
        pass
