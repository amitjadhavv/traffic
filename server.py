import socket
import jpysocket

def server_program():
    # get the hostname
    host = '192.168.100.19'
    port = 8080
    classes={0:'general caution',1:'Turn left ahead',2:'Turn right ahead',3:'Road Work',4:'stop',
 5: 'Right-of-way at intersection',
 6: 'Wild animals crossing',
 7: 'Children crossing',
 8: 'No passing',
 9: 'No vehicles',
 10: 'Vehicles over 3.5 tons prohibited',
 11: 'No entry',
 12: 'General caution',
 13: 'Dangerous curve left',
 14: 'Dangerous curve right',
 15: 'Double curve',
 16: 'Bumpy road',
 17: 'Road narrows on the right',
 18: 'Speed limit 80 kilometer per hour',
 19: 'Keep left',
 20: 'Roundabout mandatory',
 21: 'Speed limit 50 kilometer per hour',
 22: 'Speed limit 70 kilometer per hour',
 23: 'Ahead only',
 24: 'Go straight or right',
 25: 'Go straight or left',
 26: 'Keep right',
 99:"bye"
 }
    # initiate port no above 1024

    server_socket = socket.socket()  # get instance
    # look closely. The bind() function takes tuple as argument
    server_socket.bind((host, port))  # bind host address and port together

    # configure how many client the server can listen simultaneously
    server_socket.listen(2)
    conn, address = server_socket.accept()  # accept new connection
    print("Connection from: " + str(address))
    while True:
        # receive data stream. it won't accept data packet greater than 1024 bytes
        
        data = int(input(' -> '))
        msgsend = jpysocket.jpyencode(classes[data])
        conn.send(msgsend)  # send data to the client
        if data == 99:
            conn.close()  # close the connection

if __name__ == '__main__':
    server_program()
