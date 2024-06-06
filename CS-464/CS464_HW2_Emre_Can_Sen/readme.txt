For question 1, some of the later lines are commented in order to run part 2. These commented lines can be switched
in order to run part 3. It is asumed that the data file is in the same folder as the .py file. 

Most of the question 2 wasn't implemented due to poor time management :( 

cs464_lab02 is the first question

cs464_hw2.2 is the second question


IP = "127.0.0.1"

clientSendingSocket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
clientSendingSocket.bind((IP, 5))

last_packet=b'\x00\x00'
clientSendingSocket.sendto(last_packet, (IP, 5))

