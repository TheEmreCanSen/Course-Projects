# -*- coding: utf-8 -*-
"""
Created on Thu Dec  8 22:36:49 2022

@author: Emre
"""
import sys
import socket
# from threading import Thread, Lock
import threading 
import signal
import time
from time import sleep
import select


# Constants
DATA_SIZE = 1022
HEADER_SIZE = 2
IP = "127.0.0.1"
FILENAME = "received.png"

# Cmd args
PATH= str(sys.argv[1]) # file path
SEND_PORT = int(sys.argv[2])
N = int(sys.argv[3]) # window size
TIMEOUT = int(sys.argv[4]) # in ms
TIMEOUT = TIMEOUT/1000 # in s

PATH= 'asd' # file path
SEND_PORT = 310
N = 100 # window size
TIMEOUT = 120 # in ms
TIMEOUT = TIMEOUT/1000 # in s

bufferAck=[]

CLIENT_ACK_PORT=SEND_PORT

clientSendingSocket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

def chunks(message, n):
    for i in range(0, len(message), n):
        yield message[i:i+n]
        
class AckThread(threading.Thread):
    def __init__(self, sock,seq_num, data0):
        threading.Thread.__init__(self)
        self.ack_listener_socket = sock
        self.seq_num=seq_num
        self.data0=data0
        
    def run(self):
        global expectedAckNo, receivedAckNo, data, LastReceivedAckNo
        while True:
            try:
                self.ack_listener_socket.sendto(self.data0, (IP, SEND_PORT))
                sleep(TIMEOUT)
            except ack==(self.seq_num+1):
                break

threadLock = threading.Lock()             
expectedAckNo=0
receivedAckNo=0
sentSeqNo = 0
LastReceivedAckNo=[]
checkList=[]

packetList = []

fd = open("PATH", 'rb')
completeFile = fd.read()
fd.close()

packet_no = 1

for chunk in chunks(completeFile, DATA_SIZE):
    packet_no_byte=packet_no.to_bytes((packet_no.bit_length() + 7) // 4, 'big')
    packet_no_byte=packet_no_byte[len(packet_no_byte)-2:len(packet_no_byte)]
    packet = packet_no_byte + chunk
    packetList.append(packet)
    packet_no += 1

maxSequenceNo = packet_no-1
threads=[]
sentList=[]
ackList=[]
buffered=False

while receivedAckNo < maxSequenceNo:
    while (sentSeqNo - receivedAckNo < N-1) and buffered==False:
        if sentSeqNo == maxSequenceNo:
            break
        if sentSeqNo not in sentList:
            sentList.append(sentSeqNo)
            data = packetList[sentSeqNo]
            t = AckThread(clientSendingSocket,sentSeqNo,data)
            t.start()
            threads.append(t)
            sentSeqNo += 1
            
           
    if receivedAckNo != sentSeqNo:
        buffered=True
    else:
        buffered=False
    received_data, address=clientSendingSocket.recvfrom(2)
    ack= int.from_bytes(received_data[:HEADER_SIZE], byteorder="big")
    if ack not in ackList:
        ackList.append(ack)
        ackList.sort()
        receivedAckNo=receivedAckNo+1
    
last_packet=b'\x00\x00'
data=last_packet
t = AckThread(clientSendingSocket,sentSeqNo,data)
t.start()
threads.append(t)

x=True
while x:
    last_ack, address=clientSendingSocket.recvfrom(2)
    last_ack_sequence_number = int.from_bytes(last_ack[:HEADER_SIZE], byteorder="big")
    if last_ack_sequence_number==0:
        break

for t in threads: 
    t.join() 

clientSendingSocket.close()


print('File was sent.')