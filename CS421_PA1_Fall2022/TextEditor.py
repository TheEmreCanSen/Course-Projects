# -*- coding: utf-8 -*-
"""
Created on Tue Nov  1 15:56:09 2022

@author: Emre
"""

import sys
import socket
import re

HOST = sys.argv[1]  # The server's hostname or IP address
PORT = int(sys.argv[2])  # The port used by the server
# HOST = "127.0.0.1"
# PORT = 31
updt_val=0


if __name__ == "__main__":
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.connect((HOST, PORT)) #connect to server
        while True:
            user_input=int(input("Press 1 to Enter a Username:\n Press 2 to Enter a Password:\n Press 3 to Enable Write Mode:\n Press 4 to  Enable Append Mode:\n Press 5 to Exit:\n"))
            if user_input==1:
                username_input=input("Type your username:")
                un_str="USER"+" "+str(username_input) +"\r\n"
                s.sendall(un_str.encode())
                data0=s.recv(1024)
                print(f"Received {data0!r}")
            elif user_input==2:
                pass_input=input("Type your password:")
                pass_str="PASS"+" "+str(pass_input) +"\r\n"
                s.sendall(pass_str.encode())
                data1=s.recv(1024)
                print(f"Received {data1!r}")
            elif user_input==3:                
                updt_str="UPDT"+" "+str(updt_val+1) +"\r\n"
                s.sendall(updt_str.encode())
                updt_data=s.recv(1024)
                updt_val=int(re.search(r'\d+', updt_data.decode()).group())
                line_input=int(input("Type your line to write:"))
                wrte_input=input("Type your text to write:")
                wrte_str="WRTE"+" "+str(updt_val)+" "+str(line_input)+" "+ str(wrte_input)+ "\r\n"
                s.sendall(wrte_str.encode())
                data2 = s.recv(1024)
                updt_val=updt_val+1
                print(f"Received {data2!r}")
            elif user_input==4:
                updt_str="UPDT"+" "+str(updt_val+1) +"\r\n"
                s.sendall(updt_str.encode())
                updt_data=s.recv(1024)
                updt_val=int(re.search(r'\d+', updt_data.decode()).group())
                apnd_input=input("Type your text to append:")
                apnd_str="APND"+" "+str(updt_val)+" "+ str(apnd_input)+ "\r\n"
                s.sendall(apnd_str.encode())
                data3 = s.recv(1024)
                updt_val=updt_val+1
                print(f"Received {data3!r}")
            elif user_input==5:
                s.sendall(b"EXIT\r\n")
                data4= s.recv(1024)
                print(f"Received {data4!r}")
                break
            else:
                print("Please enter a valid number.")
        
# with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
#     s.connect((HOST, PORT))
#     s.sendall(b"USER bilkentstu\r\n")
#     data = s.recv(1024)
#     s.sendall(b"PASS cs421f2022\r\n")
#     data2 = s.recv(1024)
#     string0="UPDT"+" "+str(updt_val+1) +"\r\n"
#     s.sendall(string0.encode())
#     data3 = s.recv(1024)
#     updt_val=int(chr(data3[3]))
#     string1="APND"+" "+str(updt_val)+" "+"'Train of Thought'\r\n"
#     s.sendall(string1.encode())
#     data4 = s.recv(1024)
#     # s.sendall(b"WRTE 4 1 The Dark Side of the Moon\r\n")
#     # data5= s.recv(1024)
#     # s.sendall(b"EXIT\r\n")
    
# print(f"Received {data!r}")
# print(f"Received {data2!r}")
# print(f"Received {data3!r}")
# print(f"Received {data4!r}")
# print(f"Received {data5!r}")