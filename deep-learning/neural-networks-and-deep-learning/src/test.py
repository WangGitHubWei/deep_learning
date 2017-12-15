'''
Created on 2017年12月15日

@author: wangwei
'''

import mnist_loader
import network
training_data, validation_data, test_data = mnist_loader.load_data_wrapper()

# net = network.Network([784,30,10])  # 有30个隐藏层
# net.SGD(training_data,30,10,3.0,test_data=test_data)
"""
Epoch 0: 8184 / 10000
Epoch 1: 8327 / 10000
Epoch 2: 8447 / 10000
Epoch 3: 9331 / 10000
Epoch 4: 9348 / 10000
Epoch 5: 9375 / 10000
Epoch 6: 9394 / 10000
Epoch 7: 9403 / 10000
Epoch 8: 9435 / 10000
Epoch 9: 9423 / 10000
Epoch 10: 9463 / 10000
Epoch 11: 9488 / 10000
Epoch 12: 9501 / 10000
Epoch 13: 9474 / 10000

"""
net = network.Network([784,100,10]) # 有100个隐藏层
net.SGD(training_data,30,10,3.0,test_data=test_data)
"""
Epoch 0: 5655 / 10000
Epoch 1: 6574 / 10000
Epoch 2: 6662 / 10000
Epoch 3: 6697 / 10000
Epoch 4: 6700 / 10000
Epoch 5: 6754 / 10000
Epoch 6: 6761 / 10000
Epoch 7: 6765 / 10000
Epoch 8: 6781 / 10000
Epoch 9: 6781 / 10000
Epoch 10: 6776 / 10000
Epoch 11: 6782 / 10000
Epoch 12: 6783 / 10000
Epoch 13: 6798 / 10000
Epoch 14: 6813 / 10000
Epoch 15: 6793 / 10000
Epoch 16: 6818 / 10000
Epoch 17: 6820 / 10000
Epoch 18: 6817 / 10000
Epoch 19: 6827 / 10000
Epoch 20: 6818 / 10000
Epoch 21: 6826 / 10000
Epoch 22: 6817 / 10000
Epoch 23: 6831 / 10000
Epoch 24: 6844 / 10000
Epoch 25: 6868 / 10000
Epoch 26: 6850 / 10000
Epoch 27: 6877 / 10000
Epoch 28: 6918 / 10000

"""

