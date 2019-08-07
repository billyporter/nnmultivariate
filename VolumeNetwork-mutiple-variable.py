# This is code to train neural network included in publication
# A Deep Learning Approach to Selecting Representative Time Steps for Time-Varying Multivariate Data
# Accepted to IEEE Vis 2019, the #1 data visualization conference as ranked by Google.



import torch.nn as nn
import torch
import torch.autograd as autograd
from torch.nn import init
from torch.nn.modules import conv, Linear
from torch.nn.modules.utils import _pair
import torch.nn.functional as F
import torch.optim as optim
from torch.nn import init
import time
import argparse
from torch.utils.data import DataLoader
from torch.autograd import Variable
import numpy as np
from torch.utils.data import DataLoader
from torch.optim import lr_scheduler

# hyperparameters

parser = argparse.ArgumentParser(description='PyTorch Implementation of VolumeNet')
parser.add_argument('--lr', type=float, default=1e-4, metavar='LR',
                    help='learning rate')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--batch_size', type=int, default=1, metavar='N',
                    help='input batch size for training')
parser.add_argument('--epochs', type=int, default=100, metavar='N',
                    help='number of epochs to train (default: 5000)')




args = parser.parse_args()
print(not args.no_cuda)
print(torch.cuda.is_available())
args.cuda = not args.no_cuda and torch.cuda.is_available()
kwargs = {'num_workers': 30, 'pin_memory': True} if args.cuda else {}



#path where data is stored
data_path = '/afs/crc.nd.edu/user/w/wporter2/my_DL/Flownet/Data/mixfrac/'
#path where result is stored
result_path = '/afs/crc.nd.edu/user/w/wporter2/my_DL/Flownet/Result/mixfrac/mult/'

# dataset definition
Dataset = 'mixfrac'

# path where the model is stored
model_path = '/afs/crc.nd.edu/user/w/wporter2/my_DL/Flownet/Result/mixfrac/G1_'+Dataset
path = ['/afs/crc.nd.edu/user/w/wporter2/my_DL/Flownet/Data/mixfrac/mixfrac_','/afs/crc.nd.edu/user/w/wporter2/my_DL/Flownet/Data/HR/hr_bicubic_','/afs/crc.nd.edu/user/w/wporter2/my_DL/Flownet/Data/combustion_chi/chi_bicubic_']



if Dataset == 'Vortex': # folder
	name = 'vorts' #  data name
	o = [64,64,64] #  resolution
elif Dataset == 'mixfrac':
	name = 'mixfrac'
	o = [480//4,720//4,120//4]
elif Dataset == 'Supernova':
	name = 'Supernova'
	o = [108//2,108//2,108//2]
elif Dataset == 'Y_OH':
	name = 'jet_Y_OH'
	o = [480//4,720//4,120//4]
elif Dataset == 'Hurricane':
	name = 'hurricane'
	o = [500//4,500//4,100//4]
# reshape the 1D vector into 3D volume
# 120 x 180 x 30

def reshape(data,x_dim,y_dim,z_dim):
	d = np.zeros((x_dim,y_dim,z_dim))
	for z in range(0,z_dim):
		for y in range(0,y_dim):
			for x in range(0,x_dim):
				idx = x+y*x_dim+z*x_dim*y_dim
				d[x][y][z] = data[idx]
	return d


#### residual block #############


def BuildResidualBlock(channels,dropout,kernel,depth,bias):
	#channels: the number of feature map we will use in this block
	# droput : True/False. whether we use dropout to prevent overfitting
	# kernelk : kernel size
	# depth: the number of Conv layers we use in this block
	# bias: True/False. whether we will use bias. O = Wx+b if bias = True else O = Wx


	layers = []
	for i in range(int(depth)):
		layers += [nn.Conv3d(channels,channels,kernel_size=kernel,padding=kernel//2,bias=bias),
		           nn.BatchNorm3d(channels),
		           nn.ReLU(True)]
		if dropout:
			layers += [nn.Dropout(0.5)]
	layers += [nn.Conv3d(channels,channels,kernel_size=kernel,padding=kernel//2,bias=bias),
		       nn.BatchNorm3d(channels)
		       ]
	return nn.Sequential(*layers)

# Ex. ResidualBlock(channels=64,dropout=False,kernel=5,bias=False,depth=3,bias=False)
class ResidualBlock(nn.Module):
	def __init__(self,channels,dropout,kernel,depth,bias):
		super(ResidualBlock,self).__init__()
		self.block = BuildResidualBlock(channels,dropout,kernel,depth,bias)

	def forward(self,x):
		out = x+self.block(x)
		return out

# using kaiming initizalition to intialize weights
def weights_init_kaiming(m):
	classname = m.__class__.__name__
	if classname.find("Conv")!=-1:
		init.kaiming_uniform_(m.weight.data)
	elif classname.find("Linear")!=-1:
		init.kaiming_uniform_(m.weight.data)
	elif classname.find("BatchNorm")!=-1:
		init.normal_(m.weight.data, 1.0, 0.02)
		init.constant_(m.bias.data, 0.0)

# using Guassian initizalition to intialize weights

def weights_init_normal(m):
	classname = m.__class__.__name__
	if classname.find("Conv")!=-1:
		init.normal_(m.weight.data, 0.0, 0.01)
	elif classname.find("Linear")!=-1:
		init.normal_(m.weight.data, 0.0, 0.01)
	elif classname.find("BatchNorm")!=-1:
		init.normal_(m.weight.data, 1.0, 0.02)
		init.constant_(m.bias.data, 0.0)



# define different versions of GANLoss. It will not be used if we use Autoencoder
# def GANLoss(nn.Module):
# 	def __init__(self,loss_type):
# 		super(GANLoss,self).__init__()
# 		self.loss_type = loss_type
# 		if loss_type == 'BCE':
# 			self.loss = nn.BCEWithLogitsLoss()
# 		elif loss_type == 'MSE':
# 			self.loss = nn.MSELoss()
# 		elif loss_type == 'Hinge':
# 			self.loss = nn.ReLU(True)
#
# 	 def forward(self,data,label):
# 	 	if self.loss_type == 'BCE' or self.loss_type == 'MSE':
# 	 		return self.loss(data,label)
# 	 	elif self.loss_type == 'Hinge':
# 	 		if torch.sum(label) == 0:
# 	 			return self.loss(1.0+label)
# 	 		else:
# 	 			return self.loss(1.0-label)


# Conv neural network
class VolumeNet(nn.Module):
	def __init__(self):
		super(VolumeNet,self).__init__()
		# nn.Conv3d(C1,C2,4,stride=2,padding=1)   downsample the input by 2
		# nn.ConvTranspose3d(C1,C2,4,stride=2,padding=1)   upscale the input by 2
		down = [nn.Conv3d(3,64,kernel_size=(5,5,2),stride=(3,3,2),padding=(1,1,5)),  # 40 x 60 x 20
		        nn.ReLU(True),
		        ResidualBlock(channels=64,dropout=False,kernel=5,depth=3,bias=False),
		        nn.Conv3d(64,128,kernel_size=(4,4,2),stride=(2,2,2),padding=(1,1,5)), # 20 x 30 x 15
		        nn.ReLU(True),
		        ResidualBlock(channels=128,dropout=False,kernel=5,depth=3,bias=False),
		        nn.Conv3d(128,256,kernel_size=(4,2,2),stride=(2,2,2),padding=(1,5,5)), # 10 x 20 x 10
		        nn.ReLU(True),
		        ResidualBlock(channels=256,dropout=False,kernel=5,depth=3,bias=False),
		        nn.Conv3d(256,512,kernel_size=(5,8,5),stride=(3,6,3),padding=(1,1,1)), # 3 x 3 x 3
		        nn.ReLU(True),
		        ResidualBlock(channels=512,dropout=False,kernel=5,depth=3,bias=False),
		        nn.Conv3d(512,1024,kernel_size=5, stride=3, padding=1), # 1 x 1 x 1
		        nn.ReLU(True)
		        ]
		up = [  nn.ConvTranspose3d(1024,512,5, stride=3, padding=1),  # 3 x 3 x 3
		        nn.ReLU(True),
		        ResidualBlock(channels=512,dropout=False,kernel=5,depth=3,bias=False),
		        nn.ConvTranspose3d(512,256,5,stride=2,padding=2), # 5 x 5 x 5
		        nn.ReLU(True),
		        ResidualBlock(channels=256,dropout=False,kernel=5,depth=3,bias=False),
		        nn.ConvTranspose3d(256,128,(4,6,4),stride=(5,7,2),padding=(2,2,1)), # 20 x 30 x 10
		        nn.ReLU(True),
		        ResidualBlock(channels=128,dropout=False,kernel=5,depth=3,bias=False),
		        nn.ConvTranspose3d(128,64,(5,5,3),stride=(3,3,2),padding=(1,1,3)), # 60 x 90 x 15
		        nn.ReLU(True),
		        ResidualBlock(channels=64,dropout=False,kernel=5,depth=3,bias=False),
		        nn.ConvTranspose3d(64,3,4,stride=2,padding=1), # 120 x 180 x 30
		        nn.Tanh()
		      ]
		self.down = nn.Sequential(*down)
		self.up = nn.Sequential(*up)


	def forward(self,x):
		# x: the input
		feature = self.down(x)
		volume = self.up(feature)
		return feature,volume



# training function
def train(G,epochs,train_loader):

	# G : the model we need to train
	# epochs: the number of epochs for training
	#train_loader : a data structure to store the training data


	# define optimizer
	optimizer = optim.Adam(G.parameters(), lr=args.lr)

	# define loss function
	# for L1 loss, criterion = nn.L1Loss()
	criterion = nn.MSELoss()
	for itera in range(1,epochs+1):
		loss = 0
		print("========================")
		print(itera)
		x = time.time()
		for batch_idx,d in enumerate(train_loader): # get data from train_loader
			if args.cuda:
				d = d.cuda()
			d = Variable(d)
			# begin to train

			#clear gradient in optimizer
			optimizer.zero_grad()

			# get result from VolumeNet
			feature,volume = G(d)

			# compute loss
			L2 = criterion(volume,d)

			# update parameters in VolumeNet
			L2.backward()
			optimizer.step()

			loss += L2.mean().item()


		y = time.time()
		print("Time = "+str(y-x)) # print time
		print("L2 = "+str(loss)) # print loss
		if itera%100==0 or itera==1:
			torch.save(G,result_path+'G_Vortex_'+str(itera)+'.pth')
		if itera%10==0 or itera==1:
			test(G,'0046',itera)
		adjust_learning_rate(optimizer, itera);


# optimize learning rate
def adjust_learning_rate(optimizer,epoch):
	# optimizer: a class for optimizing the weights in neural netwrok
	# epoch: how frequent we decay the learning rate


    """Sets the learning rate to the initial LR decayed by 10 every 50 epochs"""
    lr = args.lr * (0.1 ** (epoch // 50))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr



def test(G,index,itera):
	# G: the model we will use
	# index: this demenstrates which datum we will use for testing. This datum should not be in the training data. For example, we use [1,2,..,30] for training, then the index should be in
	#[31,32,33,...,]
	# itera: the ith iteration, which uses for naming the data. so that we can compare the results in different iterations.
    criterion = nn.MSELoss()
    low = np.zeros((3,o[0],o[1],o[2]))

    for i in range(3):
    	data = np.fromfile(path[i]+str(index)+'.dat',dtype='<f')
    	data = 2*(data-np.min(data))/(np.max(data)-np.min(data))-1
    	low[i] = reshape(data,o[0],o[1],o[2])
    '''
    read mixfrac, hr and chi variables at the same time step and store them into low

    '''

    low = low.reshape(1,3,o[0],o[1],o[2])
    low = torch.FloatTensor(low)
    if args.cuda:
    	low = low.cuda()
    with torch.no_grad():
    	feature,volume = G(low)
    loss = criterion(volume,low).item()
    print("L2 (test) = "+str(loss))
    f = []
    v = []
    # write feature
    _,C,_,_,_ = feature.size()
    for i in range(0,C):
    	f.append(feature.data[0][i][0][0][0])
    f = np.asarray(f,dtype='<f')
    if itera%100 == 0:
        f.tofile(result_path+name+'feature-'+str(index)+'.dat',format='<f')
    #write volume
    N = ['mixfrac','hr','chi']
    for t in range(0,3):
	    for z in range(0,o[2]):
	    		for y in range(0,o[1]):
	    			for x in range(0,o[0]):
	    				v.append(volume.data[0][t][x][y][z])
    v = np.asarray(v,dtype='<f')
    if itera%100 == 0:
        v.tofile(result_path+N[t]+'volume-'+str(index)+'.dat',format='<f')



# function for generating features after network is trained
def generatefeatures(model_path,start,end):
	'''
	model path: the path where you store the model (.pth file)
	start: the starting time step where we aim to produce the feature 
	end: the ending time step where we aim to produce the feature
	'''
	G = torch.load(model_path,map_location=lambda storage, loc:storage)
	if args.cuda:
		G.cuda()

	for i in range(start,end+1):
		low = np.zeros((3,o[0],o[1],o[2]))
		for a in range(3):
			data = np.fromfile(path[a]+inttostring(i)+'.dat',dtype='<f')
			data = 2*(data-np.min(data))/(np.max(data)-np.min(data))-1
			low[a] = reshape(data,o[0],o[1],o[2])

		low = low.reshape(1,3,o[0],o[1],o[2])
		low = torch.FloatTensor(low)


		if args.cuda:
			low = low.cuda()
		with torch.no_grad():
			feature,volume = G(low)
		f = []
		_,C,_,_,_ = feature.size()
		for j in range(0,C):
			f.append(feature.data[0][j][0][0][0])
		f = np.asarray(f,dtype='<f')
		f.tofile(result_path+name+'feature-'+inttostring(i)+'.dat',format='<f')


# transform a int type to a string type
def inttostring(i):
	if i<10:
		return str(0)+str(0)+str(0)+str(i)
	elif i<100:
		return str(0)+str(0)+str(i)
	elif i<1000:
		return str(0)+str(i)
	else:
		return str(i)

# main function for reading in datam reshapingm and training
def main():
	AE = VolumeNet()
	if args.cuda:
		AE.cuda()
	AE.apply(weights_init_kaiming)
	d = [i for i in range(1,45)] #define the training samples Note the number of samples can be changed, it dependents on the what data set we use
	data = np.zeros((len(d),3,o[0],o[1],o[2]))
	path = ['/afs/crc.nd.edu/user/w/wporter2/my_DL/Flownet/Data/mixfrac/mixfrac_','/afs/crc.nd.edu/user/w/wporter2/my_DL/Flownet/Data/HR/hr_bicubic_','/afs/crc.nd.edu/user/w/wporter2/my_DL/Flownet/Data/combustion_chi/chi_bicubic_'] ## define the paths where we store mixfrac, hr and chi variables, respectively
	for i in range(0,len(d)):
		### read mixfrax variable
		d = np.fromfile(path[0]+str(inttostring(i+1))+'.dat',dtype='<f')
		data[i][0] = reshape(d,o[0],o[1],o[2])
		data[i][0] = 2*(data[i][0])/(255.0-0.0)-1 # normalize the data into [-1,1]
		### read hr variable
		d = np.fromfile(path[1]+str(inttostring(i+1))+'.dat',dtype='<f')
		data[i][1] = reshape(d,o[0],o[1],o[2])
		data[i][1] = 2*(data[i][1])/(255.0-0.0)-1 # normalize the data into [-1,1]
		### read chi variable
		d = np.fromfile(path[2]+str(inttostring(i+1))+'.dat',dtype='<f')
		data[i][2] = reshape(d,o[0],o[1],o[2])
		data[i][2] = 2*(data[i][2])/(255.0-0.0)-1 # normalize the data into [-1,1]
	data = torch.FloatTensor(data)
	train_loader = DataLoader(dataset=data,batch_size=args.batch_size, shuffle=True, **kwargs)
	train(AE,args.epochs,train_loader)

	# after network is trained use this line to generate features
	# generatefeatures('/afs/crc.nd.edu/user/w/wporter2/my_DL/Flownet/Path/G_mult.pth', 1,122)

main()