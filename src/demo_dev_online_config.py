import yarp
import numpy as np

from functions.utilities_vt import *
from functions.config_vt import *

# Initialize YARP
yarp.Network.init()

class VisualTargetDetection(yarp.RFModule):
    def configure(self, rf):
        
        # GPU
        num_gpu = rf.find("num_gpu").asInt32()
        num_gpu_start = rf.find("num_gpu_start").asInt32()
        print('Num GPU: %d, GPU start: %d' % (num_gpu, num_gpu_start))
        init_gpus(num_gpu, num_gpu_start)    
        
        # Command port
        self.cmd_port = yarp.Port()
        self.cmd_port.open('/vtd/command:i')
        print('{:s} opened'.format('/vtd/command:i'))
        self.attach(self.cmd_port)
        
        # Input port and buffer for rgb image
        # Create the port and name it
        self.in_port_human_image = yarp.BufferedPortImageRgb()
        self.in_port_human_image.open('/vtd/image:i')
        # Create numpy array to receive the image 
        self.in_buf_human_array = np.ones((IMAGE_HEIGHT, IMAGE_WIDTH, 3), dtype=np.uint8)
        self.in_buf_human_image = yarp.ImageRgb()
        self.in_buf_human_image.resize(IMAGE_WIDTH, IMAGE_HEIGHT)
        # Wrap YARP image around the array
        self.in_buf_human_image.setExternal(self.in_buf_human_array.data, self.in_buf_human_array.shape[1],
                                            self.in_buf_human_array.shape[0])
        print('{:s} opened'.format('/vtd/image:i'))
        
        
        # Input port for openpose data
        self.in_port_human_data = yarp.BufferedPortBottle()
        self.in_port_human_data.open('/vtd/data:i')
        print('{:s} opened'.format('/vtd/data:i'))
        
        # Output port for bboxes
        self.out_port_human_image = yarp.Port()
        self.out_port_human_image.open('/vtd/image:o')
        self.out_buf_human_array = np.ones((IMAGE_HEIGHT, IMAGE_WIDTH, 3), dtype=np.uint8)
        self.out_buf_human_image = yarp.ImageRgb()
        self.out_buf_human_image.resize(IMAGE_WIDTH, IMAGE_HEIGHT)
        self.out_buf_human_image.setExternal(self.out_buf_human_array.data, self.out_buf_human_array.shape[1],
                                             self.out_buf_human_array.shape[0])
        print('{:s} opened'.format('/vtd/image:o'))

        # Propag input image
        self.out_port_propag_image = yarp.Port()
        self.out_port_propag_image.open('/vtd/propag:o')
        self.out_buf_propag_image_array = np.ones((IMAGE_HEIGHT, IMAGE_WIDTH, 3), dtype=np.uint8)
        self.out_buf_propag_image = yarp.ImageRgb()
        self.out_buf_propag_image.resize(IMAGE_WIDTH, IMAGE_HEIGHT)
        self.out_buf_propag_image.setExternal(self.out_buf_propag_image_array.data, self.out_buf_propag_image_array.shape[1],
                                             self.out_buf_propag_image_array.shape[0])
        print('{:s} opened'.format('/vtd/propag:o'))

        # Output port for the selection
        self.out_port_prediction = yarp.Port()
        self.out_port_prediction.open('/vtd/pred:o')
        print('{:s} opened'.format('/vtd/pred:o'))
        
        self.human_image = np.ones((IMAGE_HEIGHT, IMAGE_WIDTH, 3), dtype=np.uint8)

        return True
    
    # Respond to a message
    def respond(self, command, reply):
        if command.get(0).asString() == 'quit':
            # command: quit
            #self.TRAIN = 0
            self.cleanup()
            reply.addString('Quit command sent')
        return True
    
    def cleanup(self):
        print('Cleanup function')
        self.in_port_human_image.close()
        self.in_port_human_depth.close()
        self.in_port_human_data.close()
        self.out_port_human_image.close()
        self.out_port_propag_image.close()
        self.out_port_propag_depth.close()
        self.out_port_prediction.close()
        self.out_port_state.close()
        self.cmd_port.close()
        return True
    
    # Called after a quit command (Does nothing)
    def interruptModule(self):
        print('Interrupt function')
        self.in_port_human_image.close()
        self.in_port_human_data.close()
        self.out_port_human_image.close()
        self.out_port_propag_image.close()
        self.out_port_prediction.close()
        self.out_port_state.close()
        self.cmd_port.close()
        return True

    # Desired period between successive calls to updateModule()
    def getPeriod(self):
        return 0.001
    
    # Update module
    def updateModule(self):

if __name__ == '__main__':
    rf = yarp.ResourceFinder()
    
    # Run module
    manager = VisualTargetDetection()
    manager.configure(rf)